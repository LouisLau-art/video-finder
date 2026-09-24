#!/usr/bin/env python3
"""05_api.py — 视频检索 FastAPI 接口服务.

把 03_search.py 的检索能力封装为标准 HTTP 接口，供 Web 前端跨域调用。

接口:
    POST /api/search          文本搜素材（视频级结果）(请求体见 SearchRequest)
    POST /api/search-by-image 以图搜素材（multipart 表单: image 文件 + 可选 query/top_k；
                              或 JSON: {image_base64, query?, top_k?})
    GET  /api/frames/{name}   帧图片静态预览 (流式返回 jpg)
    GET  /api/health          健康检查 {"status": "ok", "frames_count": N}

复用:
    encoder 加载 (load_encoder) 与查询预处理 (resolve_query_for_encoder)
    均从 03_search.py 动态导入，保证与 CLI 一致。

站点私有配置（NAS 前缀/路径映射等）来自仓库外的 site.json
（见 scripts/site_config.py），缺失则以降级空配置运行。

向量库:
    --db 显式指定则用之；缺省按顺序探测存在者:
      1) data/local-runtime/index_full/cnclip
      2) data/local-runtime/eval_chroma/cnclip
      3) chroma_db

用法:
    uv run python scripts/05_api.py                       # 默认探测库，端口 8000
    uv run python scripts/05_api.py --db chroma_db --port 8000
    uv run python scripts/05_api.py --db data/local-runtime/eval_chroma/cnclip
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
except ImportError as e:  # noqa: BLE001 — 缺依赖时给安装提示而非堆栈
    raise SystemExit(
        "[error] 缺少 fastapi 依赖，请先安装:\n"
        "    uv pip install fastapi uvicorn\n"
        f"  (原始错误: {e})"
    )

ROOT = Path(__file__).resolve().parents[1]

# 默认探测顺序：全量索引库（index_full/cnclip）优先，其次评测库，最后本地 chroma_db
DEFAULT_DB_CANDIDATES = [
    "data/local-runtime/index_full/cnclip",
    "data/local-runtime/eval_chroma/cnclip",
    "chroma_db",
]
# 帧图片目录探测顺序：全量帧目录优先（10 万帧规模下 find_frame_file 仍是逐目录
# 直接 join + stat，不做整目录扫描，故不随帧数退化；旧目录保留做兼容兜底）
DEFAULT_FRAMES_CANDIDATES = [
    "data/local-runtime/frames_full",
    "data/local-runtime/brand2_frames",
    "frames",
]

# 站点私有配置（NAS 前缀/映射表等）全部来自仓库外的 site.json，
# 缺失则给空值/空 map（降级运行，不影响本地最小链路）。
def _load_site_nas() -> dict[str, Any]:
    """读 site.json 的 nas section；失败返回空默认。"""
    try:
        path = Path(__file__).resolve().parent / "site_config.py"
        spec = importlib.util.spec_from_file_location("site_config", str(path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        nas = mod.load().get("nas", {}) or {}
        return nas if isinstance(nas, dict) else {}
    except Exception:
        return {}


_SITE_NAS = _load_site_nas()
NAS_PREFIX = str(_SITE_NAS.get("prefix", "") or "")
NAS_HOST = str(_SITE_NAS.get("smb_host", "") or "")
SYNOLOGY_WEB_BASE = str(_SITE_NAS.get("synology_web_base", "") or "")
_raw_path_map = _SITE_NAS.get("path_map", {}) or {}
NAS_PATH_MAP: dict[str, str] = (
    {str(k): str(v) for k, v in _raw_path_map.items()}
    if isinstance(_raw_path_map, dict) else {}
)

# 以图搜图：上传大小上限 15MB；允许的图片后缀；JSON base64 兼容键名
MAX_IMAGE_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_BASE64_KEYS = ("image_base64", "imageBase64", "image", "base64")


def _env_int(name: str, default: int, minimum: int) -> int:
    """读取整数环境变量；配置异常时回退到安全默认值。"""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, minimum: float) -> float:
    """读取浮点环境变量；配置异常时回退到安全默认值。"""
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量，支持常见的 0/1 开关写法。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# 视频级检索的候选扩展参数：20 倍/200 条能在常见 top_k=20 时一次覆盖
# 足够多的素材，又避免把整个帧库拉回；4 次倍增与 500ms 预算限制极端查询。
VIDEO_LEVEL_OVERSAMPLE_FACTOR = _env_int("VIDEO_LEVEL_OVERSAMPLE_FACTOR", 20, 1)
VIDEO_LEVEL_MIN_CANDIDATES = _env_int("VIDEO_LEVEL_MIN_CANDIDATES", 200, 1)
VIDEO_LEVEL_MAX_EXPANSIONS = _env_int("VIDEO_LEVEL_MAX_EXPANSIONS", 4, 0)
VIDEO_LEVEL_EXPANSION_BUDGET_MS = _env_float(
    "VIDEO_LEVEL_EXPANSION_BUDGET_MS", 500.0, 0.0
)
VIDEO_LEVEL_CANDIDATE_EXPANSION_ENABLED = _env_bool(
    "VIDEO_LEVEL_CANDIDATE_EXPANSION_ENABLED", True
)

# 关键词通道默认开启；关闭后保留工单 01 的纯语义响应契约。
KEYWORD_CHANNEL_ENABLED = _env_bool("VIDEO_FINDER_KEYWORD_ENABLED", True)
# RRF 只使用名次，不把两路分数混在一起。
RRF_K = 60

# 视频 ID -> 站点内网完整路径（随 site.json 下发，本仓库不存真实值）

# 由 main()/--参数写入的运行时配置（ensure_state 懒加载时读取）
APP_DB = ""
APP_COLLECTION = "frames"
APP_FRAMES_DIRS: list[str] = []
APP_MODEL: str | None = None
APP_TRANSLATE = False

# 懒加载单例
_M3 = None
_ENCODER = None
_ENCODER_NAME = ""
_CHROMA_COL = None
_CHROMA_COUNT = 0

# 视频级文本索引：只在内存中保存素材文件名与一个可回退的代表帧元数据。
_KEYWORD_INDEX: dict[str, dict[str, Any]] | None = None
_KEYWORD_INDEX_COUNT: int | None = None


# ---------- 复用 03_search ----------

def load_search_module():
    """动态导入 03_search.py（带数字前缀无法直接 import）."""
    global _M3
    if _M3 is None:
        path = Path(__file__).resolve().parent / "03_search.py"
        spec = importlib.util.spec_from_file_location("search_cli", str(path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _M3 = mod
    return _M3


def resolve_db(db: str | None) -> str:
    """--db 显式值优先；缺省探测候选目录中第一个“像向量库”的（有 chroma.sqlite3）。"""
    if db:
        return db
    for cand in DEFAULT_DB_CANDIDATES:
        if (ROOT / cand / "chroma.sqlite3").exists():
            return cand
        # 兼容：只有 encoder.json 也算半个库（启动时会报 collection 缺失）
        if (ROOT / cand / "encoder.json").exists():
            return cand
    # 都不存在：返回首选，让启动报错信息明确
    return DEFAULT_DB_CANDIDATES[0]


def resolve_frames_dirs(frames_dir: str | None) -> list[str]:
    """--frames-dir 显式值优先；缺省返回所有存在的候选目录。"""
    if frames_dir:
        return [frames_dir]
    return [c for c in DEFAULT_FRAMES_CANDIDATES if (ROOT / c).is_dir()]


def ensure_state():
    """懒加载 encoder + chroma collection（幂等，多请求复用）。"""
    global _ENCODER, _ENCODER_NAME, _CHROMA_COL, _CHROMA_COUNT
    if _ENCODER is not None and _CHROMA_COL is not None:
        # 索引作业可能持续写入；每请求只读取廉价 count，变化时同步候选上限。
        try:
            current_count = int(_CHROMA_COL.count())
        except Exception:
            current_count = _CHROMA_COUNT
        if current_count != _CHROMA_COUNT:
            _CHROMA_COUNT = current_count
        return _ENCODER, _CHROMA_COL
    import chromadb

    m3 = load_search_module()
    encoder = m3.load_encoder(APP_DB, APP_MODEL)
    client = chromadb.PersistentClient(path=APP_DB)
    try:
        col = client.get_collection(APP_COLLECTION)
    except Exception:
        raise RuntimeError(
            f"collection 不存在: {APP_COLLECTION}（db={APP_DB}），先跑 02_embed_index.py"
        )
    n = col.count()
    if n == 0:
        raise RuntimeError(f"索引为空: {APP_DB}，先跑 02_embed_index.py")
    _ENCODER = encoder
    _ENCODER_NAME = encoder.name
    _CHROMA_COL = col
    _CHROMA_COUNT = n
    print(f"[api] 就绪 db={APP_DB} collection={APP_COLLECTION} "
          f"frames={n} encoder={encoder.name}")
    return _ENCODER, _CHROMA_COL


# ---------- 纯函数（便于测试） ----------

def format_time(t: float) -> str:
    """秒 -> 'MM:SS.d'，如 15.0 -> '00:15.0'，42.5 -> '00:42.5'。"""
    t = max(float(t), 0.0)
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:04.1f}"


def build_result(rank: int, score: float, meta: dict[str, Any]) -> dict[str, Any]:
    """代表帧命中 -> 前端契约的 result 项。"""
    score = max(min(float(score), 1.0), 0.0)
    video_id = str(meta.get("video_id", ""))
    video_name = f"{video_id}.mp4"
    t = float(meta.get("time", 0.0) or 0.0)
    frame_path = str(meta.get("frame_path", ""))
    filename = Path(frame_path).name

    # NAS 路径：全量索引的 metadata 自带 share+relpath 时直拼最准；
    # 旧评测库无该字段，走站点下发的路径映射表；最后兜底拼一级目录。
    share = str(meta.get("share") or "")
    relpath = str(meta.get("relpath") or "")
    if share and relpath:
        nas_path = f"{share}/{relpath}"
    elif NAS_PREFIX:
        nas_path = NAS_PATH_MAP.get(video_id, f"{NAS_PREFIX}/{video_name}")
    else:
        # 无站点配置：只给映射表命中，否则退化为纯文件名（不含内网信息）
        nas_path = NAS_PATH_MAP.get(video_id, video_name)

    # 站点文件管理器的深链协议：直接定位展开所在文件夹
    # 路径需为双重 URL 编码（%252F...），且定位到所在文件夹（带末尾斜杠）；
    # 未配置基地址时置空字符串。
    if SYNOLOGY_WEB_BASE:
        folder_path = "/" + str(Path(nas_path).parent).replace("\\", "/") + "/"
        double_encoded_folder = urllib.parse.quote(urllib.parse.quote(folder_path, safe=""), safe="")
        synology_web_url = f"{SYNOLOGY_WEB_BASE}/index.cgi?launchApp=SYNO.SDS.App.FileStation3.Instance&launchParam=openfile%3D{double_encoded_folder}"
    else:
        synology_web_url = ""
    full_nas_uri = f"{NAS_HOST}/{nas_path}" if NAS_HOST else ""
    return {
        "rank": rank,
        "score": round(score, 4),
        "match_percentage": f"{round(score * 100)}%",
        "video_id": video_id,
        "video_name": video_name,
        "time_seconds": t,
        "time_formatted": format_time(t),
        "duration": "--:--",
        "resolution": "1080P",
        "frame_image_url": f"/api/frames/{filename}",
        "nas_path": nas_path,
        "full_nas_uri": full_nas_uri,
        "synology_web_url": synology_web_url,
    }


def _metadata_time(meta: dict[str, Any]) -> float:
    """读取帧时刻；异常元数据按 0 处理，保证排序仍可执行。"""
    try:
        return float(meta.get("time", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stable_meta_key(meta: dict[str, Any]) -> str:
    """把元数据转成与字典插入顺序无关的稳定排序键。"""
    return json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str)


_TYPED_FILENAME_PATTERNS = (
    re.compile(r"^\d+$"),
    re.compile(r"^\d{4,}[_-][A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*$", re.IGNORECASE),
    re.compile(
        r"^(?:OUT|IN|VID|CLIP|RAW|DSC|C)[_-]?\d{3,}"
        r"(?:[_-][A-Za-z0-9]+)*$",
        re.IGNORECASE,
    ),
    re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE),
)


def _filename_stem(meta: dict[str, Any]) -> str:
    """只取元数据中的文件名部分，不把父目录加入关键词文本。"""
    raw_name = str(meta.get("video_name") or "")
    if not raw_name:
        raw_name = str(meta.get("relpath") or meta.get("video_path") or "")
    raw_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    return Path(raw_name).stem if raw_name else ""


def _is_typed_filename(filename: str) -> bool:
    """类型化文件名只允许精确匹配，避免短数字片段误伤其它素材。"""
    value = str(filename or "").strip()
    return bool(
        value and any(pattern.fullmatch(value) for pattern in _TYPED_FILENAME_PATTERNS)
    )


def _is_meaningful_parent_segment(segment: str) -> bool:
    """保留含可读字符的目录段；剔除纯数字、哈希和无字母符号段。"""
    value = str(segment or "").strip()
    if not value or value.isdigit():
        return False
    if re.fullmatch(r"[0-9a-f]{16,}", value, re.IGNORECASE):
        return False
    # 不以数字开头判断；「2024年」「2026春季」等含可读字符的段应保留。
    return any(char.isalpha() for char in value)


def _meaningful_parent_segments(meta: dict[str, Any]) -> tuple[str, ...]:
    """从 relpath/video_path 拆出父目录片段，不把完整路径作为匹配文本。"""
    raw_path = str(meta.get("relpath") or "")
    if not raw_path:
        raw_path = str(meta.get("video_path") or "")
    parts = [
        part.strip()
        for part in raw_path.replace("\\", "/").split("/")
        if part.strip() not in {"", ".", ".."}
    ]
    share = str(meta.get("share") or "").strip()
    if share and parts and parts[0] == share:
        parts = parts[1:]
    if len(parts) <= 1:
        return ()
    parents: list[str] = []
    seen: set[str] = set()
    for part in parts[:-1]:
        if _is_meaningful_parent_segment(part) and part not in seen:
            seen.add(part)
            parents.append(part)
    return tuple(parents)


def _build_keyword_index(metas: list[Any]) -> dict[str, dict[str, Any]]:
    """从帧元数据构建素材级文本索引，并保留一个稳定的回退帧。"""
    index: dict[str, dict[str, Any]] = {}
    for raw_meta in metas:
        if not isinstance(raw_meta, dict):
            continue
        meta = dict(raw_meta)
        video_id = str(meta.get("video_id") or "")
        filename = _filename_stem(meta)
        if not video_id or not filename:
            continue
        parents = _meaningful_parent_segments(meta)
        meta_key = (_metadata_time(meta), _stable_meta_key(meta))
        text_key = (filename, parents, _stable_meta_key(meta))
        current = index.get(video_id)
        if current is None:
            index[video_id] = {
                "filename": filename,
                "parents": parents,
                "text_key": text_key,
                "typed": _is_typed_filename(filename),
                "meta": meta,
                "meta_key": meta_key,
            }
            continue
        if text_key < current["text_key"]:
            current["filename"] = filename
            current["parents"] = parents
            current["text_key"] = text_key
            current["typed"] = _is_typed_filename(filename)
        if meta_key < current["meta_key"]:
            current["meta"] = meta
            current["meta_key"] = meta_key
    return index


def _ensure_keyword_index(col: Any, current_count: int) -> dict[str, dict[str, Any]]:
    """按集合条数变化刷新内存文本索引；无变化时直接复用。"""
    global _KEYWORD_INDEX, _KEYWORD_INDEX_COUNT
    if not KEYWORD_CHANNEL_ENABLED:
        return {}
    count = int(current_count)
    if _KEYWORD_INDEX is not None and _KEYWORD_INDEX_COUNT == count:
        return _KEYWORD_INDEX
    try:
        payload = col.get(include=["metadatas"])
    except Exception:
        # 关键词索引失败时仍保留语义通道可用性；保持旧 count，下一次请求继续尝试刷新。
        if _KEYWORD_INDEX is not None:
            return _KEYWORD_INDEX
        _KEYWORD_INDEX = {}
        _KEYWORD_INDEX_COUNT = count
        return _KEYWORD_INDEX
    metas = payload.get("metadatas", []) if isinstance(payload, dict) else []
    _KEYWORD_INDEX = _build_keyword_index(list(metas or []))
    _KEYWORD_INDEX_COUNT = count
    print(
        "[api] 关键词索引刷新 "
        f"frames={count} materials={len(_KEYWORD_INDEX)}"
    )
    return _KEYWORD_INDEX


def keyword_search(
    query: str, index: dict[str, dict[str, Any]]
) -> list[tuple[str, str]]:
    """在文件名和有意义父目录片段中做确定性匹配。"""
    needle = str(query or "").strip()
    if not needle:
        return []
    matches: list[tuple[str, str]] = []
    for video_id, entry in index.items():
        filename = str(entry.get("filename") or "")
        if not filename:
            continue
        fragments: list[tuple[str, bool]] = [
            (filename, bool(entry.get("typed"))),
        ]
        fragments.extend(
            (str(parent), False) for parent in entry.get("parents", ())
        )
        hit_fragments: list[str] = []
        for fragment, is_filename in fragments:
            if not fragment:
                continue
            hit = (
                needle == fragment
                if is_filename
                else needle in fragment
            )
            if hit:
                hit_fragments.append(fragment)
        if hit_fragments:
            matched_text = min(
                hit_fragments,
                key=lambda text: (
                    0 if text == needle else 1,
                    len(text),
                    text,
                ),
            )
            matches.append((video_id, matched_text))
    matches.sort(key=lambda item: (
        0 if item[1] == needle else 1,
        len(item[1]),
        item[1],
        item[0],
    ))
    return matches


def reciprocal_rank_fusion(*rankings: Any, k: int = RRF_K) -> list[Any]:
    """纯函数 RRF：输入各通道名次，输出按名次贡献融合后的 ID 顺序。"""
    if len(rankings) == 1 and rankings and isinstance(rankings[0], (list, tuple)):
        first = rankings[0]
        if not first or isinstance(first[0], (list, tuple)):
            rankings = tuple(first)
    scores: dict[Any, float] = {}
    first_order: dict[Any, int] = {}
    denominator_base = max(1, int(k))
    for ranking in rankings:
        seen: set[Any] = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            if item not in first_order:
                first_order[item] = len(first_order)
            scores[item] = scores.get(item, 0.0) + 1.0 / (denominator_base + rank)
    return sorted(scores, key=lambda item: (-scores[item], first_order[item]))


def _aggregate_video_level(
    metas: list[Any],
    dists: list[Any],
    ids: list[Any],
    top_k: int,
) -> list[tuple[int, float, dict[str, Any]]]:
    """把帧级命中聚合为视频级结果。

    先对全部帧命中按「距离、帧时刻、稳定键」重排，消除向量库在同分
    命中时的返回顺序差异；再按素材的最佳帧名次聚合，最后只保留前
    ``top_k`` 个素材。代表帧与素材顺序都不依赖字典或不稳定排序。
    """
    hits: list[dict[str, Any]] = []
    for i, (meta_value, distance_value) in enumerate(zip(metas, dists)):
        if meta_value is None:
            continue
        meta = dict(meta_value)
        hits.append({
            "distance": float(distance_value),
            "time": _metadata_time(meta),
            "stable_key": _stable_meta_key(meta),
            "frame_id": str(ids[i]) if i < len(ids) else "",
            "meta": meta,
        })

    hits.sort(key=lambda hit: (
        hit["distance"],
        hit["time"],
        hit["stable_key"],
        hit["frame_id"],
    ))

    # 用重排后的帧名次计算素材最佳名次，代表帧也直接从排序后的组内首项取得。
    by_video: dict[str, list[dict[str, Any]]] = {}
    for frame_rank, hit in enumerate(hits, 1):
        hit["frame_rank"] = frame_rank
        video_id = str(hit["meta"].get("video_id", ""))
        by_video.setdefault(video_id, []).append(hit)

    videos: list[tuple[int, str, dict[str, Any]]] = []
    for video_id, frames in by_video.items():
        representative = min(
            frames,
            key=lambda hit: (
                hit["distance"],
                hit["time"],
                hit["stable_key"],
                hit["frame_id"],
            ),
        )
        best_frame_rank = min(hit["frame_rank"] for hit in frames)
        videos.append((best_frame_rank, video_id, representative))

    videos.sort(key=lambda item: (item[0], item[1]))
    limit = max(1, int(top_k))
    results: list[tuple[int, float, dict[str, Any]]] = []
    for rank, (_best_frame_rank, _video_id, representative) in enumerate(
        videos[:limit], 1
    ):
        score = 1.0 - float(representative["distance"]) / 2.0
        results.append((rank, score, representative["meta"]))
    return results


def find_frame_file(filename: str) -> Path | None:
    """在配置的帧目录中按 basename 查找图片；防目录穿越。"""
    name = Path(filename).name  # 剥掉任何 ../ 前缀
    if not name or name in (".", ".."):
        return None
    for d in APP_FRAMES_DIRS:
        base = ROOT / d if not Path(d).is_absolute() else Path(d)
        p = base / name
        if p.is_file():
            return p
    return None


def count_frames() -> int:
    """所有帧目录下的图片总数（健康检查用，不碰模型/向量库）。"""
    total = 0
    for d in APP_FRAMES_DIRS:
        base = ROOT / d if not Path(d).is_absolute() else Path(d)
        if base.is_dir():
            total += sum(
                1 for p in base.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
    return total


def decode_image_bytes(raw: bytes):
    """上传字节 -> PIL RGB 图；非法图片抛 ValueError（调用方转 400）。"""
    from PIL import Image

    if not raw:
        raise ValueError("图片内容为空")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"图片过大（{len(raw)} 字节），上限 {MAX_IMAGE_BYTES} 字节")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # 提前触发损坏文件报错，避免懒加载漏检
        return img.convert("RGB")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"无法解析为图片: {e}")


def decode_base64_image(s: str) -> bytes:
    """兼容 data-URL 前缀的 base64 -> 字节；非法抛 ValueError。"""
    s = (s or "").strip()
    if not s:
        raise ValueError("base64 图片字符串为空")
    if "," in s and s.startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"base64 解码失败: {e}")


def search_by_embedding(
    vec,
    k_want: int,
    label: str,
    keyword_query: str | None = None,
) -> dict[str, Any]:
    """共用检索：特征向量 -> 视频级前端契约响应体。

    vec: 1xD 或 D 维向量（list / numpy 均可）；label: 响应 data.query。
    """
    import numpy as np

    t0 = time.perf_counter()
    try:
        _encoder, col = ensure_state()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)

    requested_k = max(1, int(k_want))
    try:
        collection_count = max(1, int(col.count()))
    except Exception:
        collection_count = max(1, int(_CHROMA_COUNT))
    target_k = min(requested_k, collection_count)

    # 常见查询只需一次过取；候选不足时按倍数扩展，避免全库捞取。
    if VIDEO_LEVEL_CANDIDATE_EXPANSION_ENABLED:
        n_results = min(
            collection_count,
            max(
                target_k * VIDEO_LEVEL_OVERSAMPLE_FACTOR,
                VIDEO_LEVEL_MIN_CANDIDATES,
            ),
        )
    else:
        # 关闭扩展时保留单次 top_k 帧查询行为，聚合逻辑本身不变。
        n_results = target_k
    n_results = max(1, int(n_results))

    expansion_started = time.perf_counter()
    attempts = 0
    aggregated: list[tuple[int, float, dict[str, Any]]] = []
    stop_reason = "enough"

    while True:
        attempts += 1
        res = col.query(
            query_embeddings=arr.tolist(), n_results=n_results,
            include=["metadatas", "distances"],
        )
        metas = (res.get("metadatas") or [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or []
        ids = (res.get("ids") or [[]])[0] or []
        aggregated = _aggregate_video_level(
            list(metas), list(dists), list(ids), requested_k
        )

        if len(aggregated) >= target_k:
            stop_reason = "enough"
            break
        if not VIDEO_LEVEL_CANDIDATE_EXPANSION_ENABLED:
            stop_reason = "disabled"
            break
        if n_results >= collection_count:
            stop_reason = "hard_cap"
            break
        if attempts > VIDEO_LEVEL_MAX_EXPANSIONS:
            stop_reason = "max_expansions"
            break
        elapsed_budget_ms = (time.perf_counter() - expansion_started) * 1000
        if elapsed_budget_ms >= VIDEO_LEVEL_EXPANSION_BUDGET_MS:
            stop_reason = "budget"
            break

        next_n = min(collection_count, max(n_results + 1, n_results * 2))
        if next_n <= n_results:
            stop_reason = "hard_cap"
            break
        n_results = next_n

    if attempts > 1:
        # 不记录查询文本，避免把用户输入写入服务日志。
        print(
            "[api] 视频级候选扩展完成 "
            f"attempts={attempts} n_results={n_results} "
            f"unique_videos={len(aggregated)} reason={stop_reason}"
        )

    results = []
    for rank, score, meta in aggregated:
        results.append(build_result(rank, score, meta))

    # 关键词通道只服务文字检索；以图搜图不传 keyword_query，保持工单 01 行为。
    if str(keyword_query or "").strip() and KEYWORD_CHANNEL_ENABLED:
        keyword_index = _ensure_keyword_index(col, collection_count)
        keyword_matches = keyword_search(str(keyword_query), keyword_index)
        keyword_ids = [video_id for video_id, _matched_text in keyword_matches]
        keyword_text_by_id = {
            video_id: matched_text for video_id, matched_text in keyword_matches
        }
        semantic_by_id = {
            str(row["video_id"]): row for row in results
        }
        fused_ids = reciprocal_rank_fusion(
            [str(row["video_id"]) for row in results], keyword_ids
        )
        fused_results: list[dict[str, Any]] = []
        for rank, video_id in enumerate(fused_ids[:requested_k], 1):
            if video_id in semantic_by_id:
                row = dict(semantic_by_id[video_id])
            else:
                entry = keyword_index.get(video_id)
                if not entry:
                    continue
                # 关键词独有素材没有语义分数；RRF 只看名次，保留 0 分契约。
                row = build_result(rank, 0.0, entry["meta"])
            row["rank"] = rank
            if video_id in keyword_text_by_id:
                row["match_type"] = (
                    "both" if video_id in semantic_by_id else "keyword"
                )
                row["matched_text"] = keyword_text_by_id[video_id]
            else:
                row["match_type"] = "semantic"
                row["matched_text"] = ""
            fused_results.append(row)
        results = fused_results

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "code": 0,
        "message": "success",
        "data": {
            "query": label,
            "total": len(results),
            "elapsed_ms": elapsed_ms,
            "encoder": _ENCODER_NAME,
            "results": results,
        },
    }


# ---------- FastAPI ----------

class SearchRequest(BaseModel):
    """POST /api/search 请求体（对齐前端 shared/api.interface.ts）。

    前端可能额外带 filter 等字段：extra=ignore 直接丢弃，保证向前兼容。
    """

    model_config = {"extra": "ignore", "populate_by_name": True}

    query: str = Field(..., description="搜索文本（中文自然语言）")
    top_k: int = Field(default=20, ge=1, le=100, description="返回前 K 个素材")
    # 兼容 topk 别名写法
    topk: int | None = Field(default=None, ge=1, le=100, exclude=True)


app = FastAPI(title="视频检索 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Web 前端直接跨域调用
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "frames_count": count_frames()}


@app.get("/api/frames/{filename}")
def get_frame(filename: str):
    p = find_frame_file(filename)
    if p is None:
        raise HTTPException(status_code=404, detail=f"帧不存在: {filename}")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/api/search")
def search(req: SearchRequest) -> dict[str, Any]:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query 不能为空")
    k_want = req.top_k if req.topk is None else req.topk
    t_start = time.perf_counter()  # 全链路计时起点（含文本编码）

    try:
        encoder, _col = ensure_state()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    m3 = load_search_module()
    resolved, _translated = m3.resolve_query_for_encoder(
        q, _ENCODER_NAME, translate=APP_TRANSLATE
    )
    q_emb = encoder.encode_texts([resolved])
    out = search_by_embedding(q_emb[0], k_want, q, keyword_query=q)
    out["data"]["elapsed_ms"] = int((time.perf_counter() - t_start) * 1000)
    return out


def _clamp_top_k(v: Any, default: int) -> int:
    try:
        return max(1, min(int(v), 100))
    except (TypeError, ValueError):
        return default


@app.post("/api/search-by-image")
async def search_by_image(request: Request) -> dict[str, Any]:
    """以图搜图：multipart 表单上传优先，JSON base64 兼容。

    A. multipart/form-data: image=<文件 jpg/png/webp/bmp>，
       query=<可选辅助文字>，top_k=<返回条数，默认 10>。
    B. application/json: {image_base64|image: <base64，可带 data-URL 前缀>，
       query?, top_k?}。
    响应结构与 /api/search 完全一致，data.query 为 "[以图搜图] {filename}"。
    """
    ctype = request.headers.get("content-type", "")
    image_bytes: bytes | None = None
    filename = "upload"
    text_hint: str | None = None
    k_want = 10
    t_start = time.perf_counter()  # 全链路计时起点（含模型懒加载与图像编码）

    if "multipart" in ctype:
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"表单解析失败: {e}")
        upload = form.get("image")
        if upload is None:
            raise HTTPException(status_code=400, detail="缺少 image 文件字段")
        filename = getattr(upload, "filename", None) or "upload"
        try:
            image_bytes = await upload.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"读取上传文件失败: {e}")
        hint = form.get("query")
        if hint is not None:
            text_hint = str(hint).strip() or None
        if form.get("top_k") is not None:
            k_want = _clamp_top_k(form.get("top_k"), 10)
        elif form.get("topk") is not None:
            k_want = _clamp_top_k(form.get("topk"), 10)
    elif "json" in ctype:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON 解析失败: {e}")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON 请求体必须为对象")
        b64 = next((body.get(k) for k in IMAGE_BASE64_KEYS if body.get(k)), None)
        if b64 is None:
            raise HTTPException(
                status_code=400,
                detail=f"缺少图片字段（任一: {', '.join(IMAGE_BASE64_KEYS)}）",
            )
        try:
            image_bytes = decode_base64_image(str(b64))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        filename = str(body.get("filename") or body.get("name") or "upload")
        hint = body.get("query") or body.get("text")
        if hint is not None:
            text_hint = str(hint).strip() or None
        k_want = _clamp_top_k(body.get("top_k", body.get("topk", 10)), 10)
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type 须为 multipart/form-data（image 文件上传）"
                   "或 application/json（image_base64）",
        )

    if image_bytes is None:
        raise HTTPException(status_code=400, detail="未收到图片数据")
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in ALLOWED_IMAGE_SUFFIXES:
        print(f"[api][warn] 以图搜图非常规后缀 {suffix!r}（文件 {filename!r}），仍尝试解码")
    try:
        img = decode_image_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        _encoder, _col = ensure_state()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not hasattr(_encoder, "encode_images"):
        raise HTTPException(
            status_code=500, detail=f"当前 encoder（{_ENCODER_NAME}）不支持图像编码"
        )
    img_emb = _encoder.encode_images([img])

    label = f"[以图搜图] {Path(filename).name}"
    if text_hint:
        # 可选辅助文字：仅日志记录 + 拼入 query 标签，不改变图像检索语义
        print(f"[api] 以图搜图附带文字: {text_hint!r}")
        label += f" + {text_hint}"
    out = search_by_embedding(img_emb[0], k_want, label)
    out["data"]["elapsed_ms"] = int((time.perf_counter() - t_start) * 1000)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="视频检索 FastAPI 服务（复用 03_search 检索逻辑）")
    ap.add_argument("--db", default=None,
                    help="向量库路径；缺省自动探测 "
                         f"({' / '.join(DEFAULT_DB_CANDIDATES)})")
    ap.add_argument("--collection", default="frames")
    ap.add_argument("--frames-dir", default=None,
                    help="帧图片目录；缺省自动探测 "
                         f"({' / '.join(DEFAULT_FRAMES_CANDIDATES)})")
    ap.add_argument("--model", default=None,
                    choices=["siglip2", "cnclip", "openclip", "dummy"],
                    help="强制指定 encoder，默认读 <db>/encoder.json")
    ap.add_argument("--translate", action="store_true",
                    help="启用 opus-mt 中文翻译层（仅 siglip2/openclip 有效，默认关闭）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global APP_DB, APP_COLLECTION, APP_FRAMES_DIRS, APP_MODEL, APP_TRANSLATE
    APP_DB = resolve_db(args.db)
    APP_COLLECTION = args.collection
    APP_FRAMES_DIRS = resolve_frames_dirs(args.frames_dir)
    APP_MODEL = args.model
    APP_TRANSLATE = args.translate

    print(f"[api] db={APP_DB} collection={APP_COLLECTION} "
          f"frames_dirs={APP_FRAMES_DIRS} model={APP_MODEL or 'auto'} "
          f"translate={APP_TRANSLATE}")
    print(f"[config] site.json: nas={'on' if (NAS_HOST or NAS_PREFIX or NAS_PATH_MAP) else 'off'}, "
          f"path_map={len(NAS_PATH_MAP)}")
    if not APP_FRAMES_DIRS:
        print("[warn] 未找到帧图片目录，/api/frames 将全部 404")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
