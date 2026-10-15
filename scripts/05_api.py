#!/usr/bin/env python3
"""05_api.py — 视频检索 FastAPI 接口服务.

把 03_search.py 的检索能力封装为标准 HTTP 接口，供 Web 前端跨域调用。

接口:
    POST /api/search          文本搜素材（视频级结果）(请求体见 SearchRequest)
    POST /api/search-by-image 以图搜素材（multipart 表单: image 文件 + 可选 query/top_k；
                              或 JSON: {image_base64, query?, top_k?})
    POST /api/feedback        意见收集（搜索词/结果/原因/描述 append 到 JSONL）
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
import mimetypes
import os
import re
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

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

# 意见收集（/api/feedback）落盘路径：JSONL append。
# 生产环境用 VF_FEEDBACK_PATH 指向部署目录之外（rsync 代码同步不会误删）；
# 本地/测试默认 data/feedback/feedback.jsonl。
FEEDBACK_PATH = Path(
    os.environ.get("VF_FEEDBACK_PATH") or (ROOT / "data" / "feedback" / "feedback.jsonl")
)
FEEDBACK_REASONS = ("wrong_result", "missing_asset", "other")
_FEEDBACK_MAX_QUERY = 200
_FEEDBACK_MAX_RESULT_ID = 200
_FEEDBACK_MAX_DETAIL = 2000

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

# 元数据分页批次保持在 SQLite 变量上限以下；500 条是安全性与吞吐量的折中。
METADATA_BATCH_SIZE = _env_int("VIDEO_FINDER_METADATA_BATCH_SIZE", 500, 1)
# 索引增长期间最多每 30 秒重建一次；查询先使用旧索引，后续查询完成刷新。
KEYWORD_INDEX_REBUILD_MIN_INTERVAL = _env_float(
    "VIDEO_FINDER_KEYWORD_INDEX_REBUILD_INTERVAL", 30.0, 0.0
)

# 关键词通道默认开启；关闭后保留工单 01 的纯语义响应契约。
KEYWORD_CHANNEL_ENABLED = _env_bool("VIDEO_FINDER_KEYWORD_ENABLED", True)
# RRF 只使用名次，不把语义通道与关键词通道的分数混在一起。
RRF_K = 60
# 关键词通道提权系数：专有名词查询下提升关键词通道贡献，避免命中被语义噪音挤出。
KEYWORD_BOOST_WEIGHT = _env_float("VIDEO_FINDER_KEYWORD_BOOST_WEIGHT", 1.25, 1.0)
# 选择性门控：仅当关键词命中素材数占全库比例 <= 该比例时提权。
KEYWORD_SELECTIVE_RATIO_MAX = _env_float(
    "VIDEO_FINDER_KEYWORD_SELECTIVE_RATIO_MAX", 0.005, 0.0
)
# 选择性门控的绝对下限命中数，避免小素材库时比例阈值过于苛刻。
KEYWORD_SELECTIVE_MIN_HITS = _env_int("VIDEO_FINDER_KEYWORD_SELECTIVE_MIN_HITS", 40, 1)

# 台词（ASR）通道：transcripts 集合名 / 候选数 / RRF 权重均可用环境变量覆盖。
# 新通道保守起步：权重默认 1.0、不做选择性提权；集合不存在时完全退化为两路融合。
ASR_COLLECTION_NAME = (
    (os.getenv("VIDEO_FINDER_ASR_COLLECTION") or "").strip() or "transcripts"
)
ASR_CANDIDATE_LIMIT = _env_int("VIDEO_FINDER_ASR_CANDIDATES", 50, 1)
ASR_CHANNEL_WEIGHT = _env_float("VIDEO_FINDER_ASR_WEIGHT", 1.0, 0.0)

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

# 台词（ASR）集合懒加载单例：集合不存在时留 None（不缓存失败结果，
# 便于服务运行期间 ASR 全量补齐后自动生效）。
_TRANSCRIPTS_COL = None
_TRANSCRIPTS_CLIENT = None

# 视频级文本索引：只在内存中保存素材文件名与一个可回退的代表帧元数据。
_KEYWORD_INDEX: dict[str, dict[str, Any]] | None = None
_KEYWORD_INDEX_COUNT: int | None = None
_KEYWORD_INDEX_LAST_REFRESH_AT: float | None = None

# 图片删除下架集合（ticket-04-part2）：state_images.json 中 deleted_pending
# （缓冲期内）的 image_key 不出现在搜索结果里。集合懒加载 + 节流刷新，
# 与关键词索引同模式；state 文件缺失/损坏时为空集合（不影响检索）。
_DELETED_IMAGE_KEYS: set[str] = set()
_DELETED_KEYS_LAST_REFRESH_AT: float | None = None
_DELETED_KEYS_MTIME: float | None = None

# 元数据 path 字段推导出的额外帧/原图目录（find_frame_file 兜底候选，
# 随关键词索引刷新而更新；固定候选 APP_FRAMES_DIRS 永远优先）。
_EXTRA_FRAME_DIRS: list[str] = []

# 健康检查的图片计数缓存：(collection_count, images_count)。
_IMAGES_COUNT_CACHE: tuple[int, int] | None = None


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


def _ensure_transcripts_col():
    """懒加载 ASR 台词集合（同 APP_DB 下的 transcripts）；不可用时返回 None。

    台词通道是可选增强：集合尚未创建（ASR 全量还没跑）、库不可达或读取
    异常时一律返回 None，绝不影响既有语义/关键词检索。失败结果不缓存，
    服务运行期间 06c 补齐集合后下一次查询即可自动生效。
    """
    global _TRANSCRIPTS_CLIENT, _TRANSCRIPTS_COL
    if _TRANSCRIPTS_COL is not None:
        return _TRANSCRIPTS_COL
    if not str(APP_DB or "").strip():
        return None
    if _TRANSCRIPTS_CLIENT is None:
        try:
            import chromadb
            # 同路径多 client 由 chromadb 支持；与 ensure_state 的 client 相互独立。
            _TRANSCRIPTS_CLIENT = chromadb.PersistentClient(path=APP_DB)
        except Exception:
            return None
    try:
        col = _TRANSCRIPTS_CLIENT.get_collection(ASR_COLLECTION_NAME)
    except Exception:
        return None
    _TRANSCRIPTS_COL = col
    try:
        count = int(col.count())
    except Exception:
        count = -1
    print(f"[api] 台词通道就绪 collection={ASR_COLLECTION_NAME} segments={count}")
    return col


# ---------- 纯函数（便于测试） ----------

def format_time(t: float) -> str:
    """秒 -> 'MM:SS.d'，如 15.0 -> '00:15.0'，42.5 -> '00:42.5'。"""
    t = max(float(t), 0.0)
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:04.1f}"


def build_result(rank: int, score: float, meta: dict[str, Any]) -> dict[str, Any]:
    """代表帧命中 -> 前端契约的 result 项。

    media_type=="image" 走图片分支：时间戳字段为占位，新增尺寸字段，
    其余字段（NAS 路径/深链/帧图 URL）与视频行共用同一套契约。
    """
    score = max(min(float(score), 1.0), 0.0)
    video_id = str(meta.get("video_id", ""))
    is_image = str(meta.get("media_type") or "") == "image"
    if is_image:
        # 图片条目 video_name 已带真实扩展名（06b 入库契约），
        # 不能像视频一样硬拼 .mp4。
        video_name = str(meta.get("video_name") or f"{video_id}.png")
    else:
        video_name = f"{video_id}.mp4"
    t = float(meta.get("time", 0.0) or 0.0)
    frame_path = str(meta.get("frame_path", ""))
    # 图片行的封面优先用索引时生成的网页缩略图（thumb_path）：原图在 NAS 上，
    # 而公网服务器不挂 NAS（ADR-0001），直接服务原图会 404。缺失时回退原图名。
    thumb_path = str(meta.get("thumb_path", "")) if is_image else ""
    filename = Path(thumb_path or frame_path).name

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

    if is_image:
        width = _metadata_int(meta, "width")
        height = _metadata_int(meta, "height")
        resolution = (
            f"{width}x{height}" if width > 0 and height > 0 else ""
        )
        return {
            "rank": rank,
            "score": round(score, 4),
            "match_percentage": f"{round(score * 100)}%",
            "video_id": video_id,
            "video_name": video_name,
            "media_type": "image",
            "time_seconds": 0.0,
            "time_formatted": "--:--",
            "duration": "--:--",
            "resolution": resolution,
            "width": width,
            "height": height,
            "frame_image_url": f"/api/frames/{filename}",
            "nas_path": nas_path,
            "full_nas_uri": full_nas_uri,
            "synology_web_url": synology_web_url,
        }
    return {
        "rank": rank,
        "score": round(score, 4),
        "match_percentage": f"{round(score * 100)}%",
        "video_id": video_id,
        "video_name": video_name,
        "media_type": "video",
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


def _metadata_int(meta: dict[str, Any], key: str) -> int:
    """读取整型元数据字段；缺失/异常按 0 处理（图片宽高等可选字段用）。"""
    try:
        return max(0, int(float(meta.get(key, 0) or 0)))
    except (TypeError, ValueError):
        return 0


def _identity(meta: dict[str, Any]) -> str:
    """素材的全局唯一身份：图片条目用 image_key，其余优先 video_key。

    图片与视频同库混存，video_id（=文件名 stem）在两类素材间大量撞名；
    image_key 与 video_key 同构（同哈希算法），聚合时天然互不串身份。
    历史集合（评测库、单测假数据）没有 key 字段时回退 video_id，
    与旧集合行为一致。
    """
    if str(meta.get("media_type") or "") == "image":
        return str(meta.get("image_key") or "")
    return str(meta.get("video_key") or meta.get("video_id") or "")


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


def iter_collection_metadata(
    col: Any,
    expected_count: int | None = None,
    batch_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    """按固定 offset 分页读取元数据，避免一次查询生成超大 SQL 参数集。

    Chroma 的 offset 分页顺序保持原样，不对单页二次排序；调用方以传入的
    count 作为快照边界，避免索引增长时无限追新数据。
    """
    size = max(1, int(batch_size or METADATA_BATCH_SIZE))
    offset = 0
    while expected_count is None or offset < expected_count:
        limit = size if expected_count is None else min(size, expected_count - offset)
        payload = col.get(include=["metadatas"], limit=limit, offset=offset)
        raw_metas = payload.get("metadatas") if isinstance(payload, dict) else None
        metas = raw_metas if isinstance(raw_metas, list) else []
        if not metas:
            break
        for raw_meta in metas:
            if isinstance(raw_meta, dict):
                yield dict(raw_meta)
        returned = len(metas)
        offset += returned
        if returned < limit:
            break


def _build_keyword_index(
    metas: Iterable[Any],
    frame_dirs_out: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """从帧元数据构建素材级文本索引，并保留一个稳定的回退帧。

    ``frame_dirs_out`` 传入时顺带收集元数据 path 字段的父目录集合
    （供 /api/frames 兜底服务图片原图），避免二次全量扫描。
    """
    index: dict[str, dict[str, Any]] = {}
    for raw_meta in metas:
        if not isinstance(raw_meta, dict):
            continue
        meta = dict(raw_meta)
        identity = _identity(meta)
        filename = _filename_stem(meta)
        if not identity or not filename:
            continue
        if frame_dirs_out is not None:
            frame_dirs_out.update(_metadata_frame_dirs(meta))
        parents = _meaningful_parent_segments(meta)
        meta_key = (_metadata_time(meta), _stable_meta_key(meta))
        text_key = (filename, parents, _stable_meta_key(meta))
        current = index.get(identity)
        if current is None:
            index[identity] = {
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
    """按条数变化刷新索引，并用最小间隔抑制增长期间的重复重建。

    条数变化后的查询在间隔内继续使用旧索引；间隔满足后的下一次查询
    重建当前快照，因此是最终一致而不是强一致。
    """
    global _KEYWORD_INDEX, _KEYWORD_INDEX_COUNT, _KEYWORD_INDEX_LAST_REFRESH_AT
    if not KEYWORD_CHANNEL_ENABLED:
        return {}
    count = int(current_count)
    if _KEYWORD_INDEX is not None and _KEYWORD_INDEX_COUNT == count:
        return _KEYWORD_INDEX
    clock = getattr(time, "monotonic", time.perf_counter)
    now = clock()
    if (
        _KEYWORD_INDEX_LAST_REFRESH_AT is not None
        and now - _KEYWORD_INDEX_LAST_REFRESH_AT < KEYWORD_INDEX_REBUILD_MIN_INTERVAL
    ):
        return _KEYWORD_INDEX if _KEYWORD_INDEX is not None else {}
    _KEYWORD_INDEX_LAST_REFRESH_AT = now
    try:
        frame_dirs: set[str] = set()
        metas = iter_collection_metadata(
            col,
            expected_count=count,
            batch_size=METADATA_BATCH_SIZE,
        )
        new_index = _build_keyword_index(metas, frame_dirs_out=frame_dirs)
    except Exception:
        # 关键词索引失败时仍保留语义通道可用性；旧索引和旧 count 保持不变，
        # 下一次查询在节流间隔结束后继续尝试刷新。
        return _KEYWORD_INDEX if _KEYWORD_INDEX is not None else {}
    _KEYWORD_INDEX = new_index
    _KEYWORD_INDEX_COUNT = count
    _register_metadata_frame_dirs(frame_dirs)
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


def _keyword_channel_weight(hit_count: int, material_count: int) -> float:
    """按查询选择性决定关键词通道权重。

    命中素材数占整个素材库比例很小的查询（专有名词）→ 提权，避免其命中被
    语义噪音挤出默认结果；
    命中很多素材的泛化词（如高频目录名）→ 保持 1.0，避免关键词淹没语义通道。
    """
    try:
        hits = int(hit_count)
        materials = int(material_count)
    except (TypeError, ValueError):
        return 1.0
    if hits <= 0:
        return 1.0
    proportional_limit = int(materials * KEYWORD_SELECTIVE_RATIO_MAX)
    limit = max(KEYWORD_SELECTIVE_MIN_HITS, proportional_limit)
    if hits <= limit:
        return KEYWORD_BOOST_WEIGHT
    return 1.0


def reciprocal_rank_fusion(
    *rankings: Any,
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[Any]:
    """纯函数 RRF：输入各通道名次，输出按名次贡献融合后的 ID 顺序。

    ``weights`` 可选地为每一路通道提供贡献系数，权重缺省或长度不足时按
    1.0 处理；不传时行为与旧实现逐位一致。
    """
    if len(rankings) == 1 and rankings and isinstance(rankings[0], (list, tuple)):
        first = rankings[0]
        if not first or isinstance(first[0], (list, tuple)):
            rankings = tuple(first)
    scores: dict[Any, float] = {}
    first_order: dict[Any, int] = {}
    denominator_base = max(1, int(k))
    for index, ranking in enumerate(rankings):
        if weights is not None and index < len(weights):
            weight = float(weights[index])
        else:
            weight = 1.0
        seen: set[Any] = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            if item not in first_order:
                first_order[item] = len(first_order)
            scores[item] = scores.get(item, 0.0) + weight / (denominator_base + rank)
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
        video_id = _identity(hit["meta"])
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


def _transcript_search(
    query_vec: Any,
    col_transcripts: Any,
    top_n: int,
) -> list[tuple[str, float, dict[str, Any]]]:
    """台词向量通道：查询台词集合并聚合到素材级，返回 [(identity, score, meta)]。

    同一素材的多段台词只保留距离最小（最佳）的一段作为素材代表，identity
    去重后按距离升序输出；距离 -> 分数换算沿用 _aggregate_video_level 的
    约定 ``score = 1.0 - distance / 2.0``。

    集合为 None、为空或查询异常时返回空列表，绝不影响其它通道。
    """
    if col_transcripts is None:
        return []
    try:
        import numpy as np
        arr = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        count = int(col_transcripts.count())
    except Exception:
        return []
    if count <= 0:
        return []
    n_results = max(1, min(int(top_n), count))
    try:
        res = col_transcripts.query(
            query_embeddings=arr.tolist(),
            n_results=n_results,
            include=["metadatas", "distances"],
        )
    except Exception:
        return []

    raw_metas = (res.get("metadatas") or [[]])[0] or []
    dists = (res.get("distances") or [[]])[0] or []
    ids = (res.get("ids") or [[]])[0] or []

    # 先按「距离、台词时刻、稳定键、文档 ID」重排，消除向量库的返回顺序差异；
    # 再按重排后的顺序取每个素材的首段（即最佳段），素材顺序即最佳段名次。
    hits: list[dict[str, Any]] = []
    for i, meta_value in enumerate(raw_metas):
        if meta_value is None:
            continue
        meta = dict(meta_value)
        if not _identity(meta):
            continue
        try:
            distance = float(dists[i])
        except (IndexError, TypeError, ValueError):
            distance = 2.0
        hits.append({
            "distance": distance,
            "time": _metadata_time(meta),
            "stable_key": _stable_meta_key(meta),
            "doc_id": str(ids[i]) if i < len(ids) else "",
            "meta": meta,
        })
    hits.sort(key=lambda hit: (
        hit["distance"],
        hit["time"],
        hit["stable_key"],
        hit["doc_id"],
    ))

    results: list[tuple[str, float, dict[str, Any]]] = []
    seen: set[str] = set()
    for hit in hits:
        identity = _identity(hit["meta"])
        if identity in seen:
            continue
        seen.add(identity)
        score = 1.0 - hit["distance"] / 2.0
        results.append((identity, score, hit["meta"]))
    return results


def _metadata_frame_dirs(meta: dict[str, Any]) -> set[str]:
    """单条元数据 -> 帧图/原图所在目录集合（find_frame_file 兜底候选）。"""
    path_value = str(meta.get("frame_path") or meta.get("image_path") or "")
    if not path_value:
        return set()
    path = Path(path_value.replace("\\", "/"))
    if path.is_absolute():
        parent = path.parent
    elif path_value.startswith("~"):
        parent = path.expanduser().parent
    else:
        parent = (ROOT / path).parent
    try:
        return {str(parent.resolve())}
    except OSError:
        return set()


def _register_metadata_frame_dirs(dirs: Iterable[str]) -> None:
    """把元数据推导出的目录并入帧目录候选（幂等、去重、排序稳定）。

    候选集无变化时跳过赋值，避免每次搜索都触发无谓的重建。
    """
    global _EXTRA_FRAME_DIRS
    merged = set(_EXTRA_FRAME_DIRS)
    for d in dirs:
        if d:
            merged.add(str(d))
    new_dirs = sorted(merged)
    if new_dirs != _EXTRA_FRAME_DIRS:
        _EXTRA_FRAME_DIRS = new_dirs


def find_frame_file(filename: str) -> Path | None:
    """在配置的帧目录中按 basename 查找图片；防目录穿越。

    目录候选 = APP_FRAMES_DIRS（固定列表，优先）+ 元数据里出现过的
    path 字段父目录（兜底，覆盖图片原图所在目录）。全程只 join
    basename，调用方传入的任何路径成分都会被剥掉。
    """
    name = Path(filename).name  # 剥掉任何 ../ 前缀
    if not name or name in (".", ".."):
        return None
    candidates = list(APP_FRAMES_DIRS) + list(_EXTRA_FRAME_DIRS)
    for d in candidates:
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


def count_indexed_images() -> int:
    """索引中的图片条目数（健康检查可选字段）。

    只复用已加载的 collection（检索发生过至少一次后才有值），不触发
    模型/库懒加载；按库条数缓存，索引未变化时零成本返回。冷启动或
    库不可达时返回 0，绝不影响 /api/health 可用性。
    """
    col = _CHROMA_COL
    if col is None:
        return 0
    global _IMAGES_COUNT_CACHE
    try:
        total = int(col.count())
    except Exception:
        return _IMAGES_COUNT_CACHE[1] if _IMAGES_COUNT_CACHE else 0
    if _IMAGES_COUNT_CACHE is not None and _IMAGES_COUNT_CACHE[0] == total:
        return _IMAGES_COUNT_CACHE[1]
    try:
        images = sum(
            1
            for meta in iter_collection_metadata(col, expected_count=total)
            if str(meta.get("media_type") or "") == "image"
        )
    except Exception:
        return _IMAGES_COUNT_CACHE[1] if _IMAGES_COUNT_CACHE else 0
    _IMAGES_COUNT_CACHE = (total, images)
    return images


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
    # 懒加载分页取数窗口：offset 翻页时需要多取候选才能切片到位。
    # 无分页（offset=0）时 fetch_k == requested_k，行为与旧版逐位一致。
    fetch_k = min(100, requested_k * 2)
    try:
        collection_count = max(1, int(col.count()))
    except Exception:
        collection_count = max(1, int(_CHROMA_COUNT))
    # 候选扩展逻辑以"期望返回条数"（requested_k）定目标；fetch_k 放大
    # 只影响首查 n_results，避免 offset 翻页触发多余的扩展查询。
    target_k = min(requested_k, collection_count)

    # 常见查询只需一次过取；候选不足时按倍数扩展，避免全库捞取。
    # 懒加载分页：fetch_k > requested_k 时，首查取大窗口但仍以 requested_k 为目标，
    # 保证常见路径（如 top_k=3/offset=0）保持单次查询行为不变。
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
        n_results = requested_k
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
    identities: list[str] = []
    semantic_by_identity: dict[str, dict[str, Any]] = {}
    for rank, score, meta in aggregated:
        row = build_result(rank, score, meta)
        results.append(row)
        identity = _identity(meta)
        identities.append(identity)
        semantic_by_identity[identity] = row
        # 语义命中行本身也登记帧/原图目录，关键词独有图片行（无语义分）
        # 与关键词索引刷新失败时的兜底都靠这一步保证 /api/frames 可用。
        _register_metadata_frame_dirs(_metadata_frame_dirs(meta))

    # 关键词与台词通道都只服务文字检索；以图搜图不传 keyword_query，保持工单 01 行为。
    text_query = str(keyword_query or "").strip()
    keyword_active = bool(text_query) and KEYWORD_CHANNEL_ENABLED

    # 台词通道：集合不存在（ASR 全量还没跑）或查询异常时静默跳过，不影响既有检索。
    transcript_hits: list[tuple[str, float, dict[str, Any]]] = []
    if text_query:
        transcript_hits = _transcript_search(
            arr[0], _ensure_transcripts_col(), ASR_CANDIDATE_LIMIT
        )
    transcript_ids = [identity for identity, _score, _meta in transcript_hits]
    transcript_score_by_id = {
        identity: score for identity, score, _meta in transcript_hits
    }
    transcript_meta_by_id = {
        identity: meta for identity, _score, meta in transcript_hits
    }

    if keyword_active or transcript_ids:
        keyword_index = (
            _ensure_keyword_index(col, collection_count) if keyword_active else {}
        )
        keyword_ids: list[str] = []
        keyword_text_by_id: dict[str, str] = {}
        kw_weight = 1.0
        if keyword_active:
            keyword_matches = keyword_search(text_query, keyword_index)
            keyword_ids = [identity for identity, _matched_text in keyword_matches]
            keyword_text_by_id = {
                identity: matched_text for identity, matched_text in keyword_matches
            }
            kw_weight = _keyword_channel_weight(len(keyword_ids), len(keyword_index))

        # 语义 + 关键词 + 台词三路 RRF；台词通道不可用时通道序列与旧版逐位一致。
        rankings: list[list[str]] = [identities]
        weights: list[float] = [1.0]
        if keyword_active:
            rankings.append(keyword_ids)
            weights.append(kw_weight)
        if transcript_ids:
            rankings.append(transcript_ids)
            weights.append(ASR_CHANNEL_WEIGHT)
        fused_ids = reciprocal_rank_fusion(*rankings, k=RRF_K, weights=tuple(weights))

        fused_results: list[dict[str, Any]] = []
        for rank, identity in enumerate(fused_ids[:fetch_k], 1):
            semantic_row = semantic_by_identity.get(identity)
            keyword_text = keyword_text_by_id.get(identity)
            transcript_meta = transcript_meta_by_id.get(identity)
            # 台词命中优先以台词段为代表：时间给台词起始秒、摘要给台词原句；
            # 例外：语义+关键词双命中（历史 "both" 语义）保持不变。
            if transcript_meta is not None and not (
                semantic_row is not None and keyword_text is not None
            ):
                row = build_result(
                    rank,
                    transcript_score_by_id.get(identity, 0.0),
                    transcript_meta,
                )
                # 台词段没有帧图（frame_path 属于帧图）；封面留空，契约字段仍存在。
                row["frame_image_url"] = ""
                match_type = "transcript"
                matched_text = str(transcript_meta.get("text") or "")
            elif semantic_row is not None:
                row = dict(semantic_row)
                if keyword_text is not None:
                    match_type, matched_text = "both", keyword_text
                else:
                    match_type, matched_text = "semantic", ""
            else:
                entry = keyword_index.get(identity)
                if not entry:
                    continue
                # 关键词独有素材没有语义分数；RRF 只看名次，保留 0 分契约。
                row = build_result(rank, 0.0, entry["meta"])
                match_type, matched_text = "keyword", keyword_text or ""
            row["rank"] = rank
            row["match_type"] = match_type
            row["matched_text"] = matched_text
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
    # 懒加载分页：从全部候选结果的第 offset 条开始切片（0 起始）
    offset: int = Field(default=0, ge=0, le=10000, description="结果偏移量（翻页用）")


class FeedbackRequest(BaseModel):
    """POST /api/feedback 请求体（意见收集）。

    全部字段容错：缺失/超长/非法 reason 都会被规范化而不是报错
    （见 normalize_feedback）。前端可能额外带字段：extra=ignore 丢弃。
    """

    model_config = {"extra": "ignore"}

    query: str = Field(default="", description="当时的搜索词（可为空）")
    result_id: str = Field(
        default="", description="被反馈结果的素材身份（video_key/image_key，可空）"
    )
    reason: str = Field(
        default="other", description="wrong_result | missing_asset | other"
    )
    detail: str = Field(default="", description="用户补充描述（可为空）")


def _clip_feedback_text(value: Any, limit: int) -> str:
    """转字符串、去首尾空白、按上限截断（防御式，不抛错）。"""
    text = "" if value is None else str(value)
    return text.strip()[:limit]


def normalize_feedback(body: FeedbackRequest) -> dict[str, Any]:
    """规范化反馈记录：截断超长、归一化 reason、补北京时间戳。"""
    reason = _clip_feedback_text(body.reason, 32).lower()
    if reason not in FEEDBACK_REASONS:
        reason = "other"
    return {
        "ts": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "query": _clip_feedback_text(body.query, _FEEDBACK_MAX_QUERY),
        "result_id": _clip_feedback_text(body.result_id, _FEEDBACK_MAX_RESULT_ID),
        "reason": reason,
        "detail": _clip_feedback_text(body.detail, _FEEDBACK_MAX_DETAIL),
    }


def write_feedback(record: dict[str, Any], path: Path | None = None) -> Path:
    """把一条反馈 append 到 JSONL（自动建父目录）；返回实际写入路径。"""
    target = Path(path) if path is not None else FEEDBACK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with target.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return target


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
    return {
        "status": "ok",
        "frames_count": count_frames(),
        "images_count": count_indexed_images(),
    }


@app.get("/api/frames/{filename}")
def get_frame(filename: str):
    p = find_frame_file(filename)
    if p is None:
        raise HTTPException(status_code=404, detail=f"帧不存在: {filename}")
    # 视频帧图均为 jpg；该端点同时服务图片原图（png/webp 等），按后缀给 MIME。
    media_type, _ = mimetypes.guess_type(p.name)
    return FileResponse(str(p), media_type=media_type or "image/jpeg")


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
    results = out.get("data", {}).get("results", [])
    # 图片删除下架过滤（ticket-04-part2）：deleted_pending 缓冲期内的图片行剔除，
    # rank 重排后再分页。集合为空（未跑 GC / state 缺失）时逐位不走过滤。
    results = filter_deleted_rows(results, ensure_deleted_image_keys())
    window, has_more = paginate_results(results, req.offset, k_want)
    out["data"]["results"] = window
    out["data"]["total"] = len(window)  # 语义恢复为"本页条数"（契约：total=results 长度）
    out["data"]["offset"] = req.offset
    out["data"]["has_more"] = has_more
    out["data"]["total_fetched"] = len(results)
    out["data"]["elapsed_ms"] = int((time.perf_counter() - t_start) * 1000)
    return out


def _clamp_top_k(v: Any, default: int) -> int:
    try:
        return max(1, min(int(v), 100))
    except (TypeError, ValueError):
        return default


def load_deleted_image_keys(state_path: Path | str) -> set[str]:
    """读 state_images.json，返回 deleted_pending（缓冲期内下架）的 image_key 集合。

    文件不存在/损坏返回空集合：GC 从未跑过或 state 异常时，检索行为与旧版一致。
    """
    p = Path(state_path)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — state 损坏不能让检索挂掉
        print(f"[api] 图片 state 读取失败（下架过滤跳过）: {e}")
        return set()
    if not isinstance(data, dict):
        return set()
    return {
        str(k) for k, v in data.items()
        if isinstance(v, dict) and v.get("status") == "deleted_pending"
    }


def _deleted_image_keys_refresh_interval() -> int:
    return _env_int("VIDEO_FINDER_DELETED_REFRESH_SEC", 60, 1)


def ensure_deleted_image_keys() -> set[str]:
    """下架集合懒加载 + 节流刷新（与 ensure_keyword_index 同模式）。

    state 文件 mtime 变化（GC 刚跑过）时立即刷新，否则至少 60s 轮询一次。
    """
    global _DELETED_IMAGE_KEYS, _DELETED_KEYS_LAST_REFRESH_AT, _DELETED_KEYS_MTIME
    state_path = _images_state_path()
    if state_path is None:
        return _DELETED_IMAGE_KEYS
    try:
        mtime = state_path.stat().st_mtime
    except OSError:
        return _DELETED_IMAGE_KEYS
    now = time.time()
    stale = (
        _DELETED_KEYS_LAST_REFRESH_AT is None
        or now - _DELETED_KEYS_LAST_REFRESH_AT >= _deleted_image_keys_refresh_interval()
    )
    if stale or mtime != _DELETED_KEYS_MTIME:
        _DELETED_IMAGE_KEYS = load_deleted_image_keys(state_path)
        _DELETED_KEYS_LAST_REFRESH_AT = now
        _DELETED_KEYS_MTIME = mtime
    return _DELETED_IMAGE_KEYS


def filter_deleted_rows(rows: list[dict[str, Any]],
                        deleted_keys: set[str]) -> list[dict[str, Any]]:
    """剔除下架图片行；rank 重排连续。集合为空时原样返回（逐位一致）。"""
    if not deleted_keys:
        return rows
    kept = [
        r for r in rows
        if not (str(r.get("media_type")) == "image"
                and str(r.get("image_key") or "") in deleted_keys)
    ]
    if len(kept) == len(rows):
        return rows
    for rank, r in enumerate(kept, 1):
        r["rank"] = rank  # 原地改 rank 保持对象身份（调用方可能持有引用）
    return kept


def _images_state_path() -> Path | None:
    """state_images.json 路径：与 APP_DB 同级的 index_full/ 下（部署形态一致）。"""
    if not APP_DB:
        return None
    # APP_DB = <base>/index_full/cnclip；state 在 <base>/index_full/ 下
    p = Path(APP_DB).parent / "state_images.json"
    return p


def paginate_results(
    results: list[dict[str, Any]], offset: int, k_want: int
) -> tuple[list[dict[str, Any]], bool]:
    """结果切片：返回 (窗口内结果, 是否还有更多)。offset 越界时返回空列表。"""
    window = results[offset : offset + k_want]
    has_more = len(results) > offset + k_want
    return window, has_more


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


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    """意见收集：把用户反馈（搜索词/结果/原因/描述）append 到 JSONL。

    字段问题一律规范化（截断/归一）而非 400；仅落盘失败返回 500。
    """
    record = normalize_feedback(req)
    try:
        write_feedback(record)
    except OSError as e:
        print(f"[api][error] 反馈落盘失败: {e}")
        raise HTTPException(status_code=500, detail="反馈落盘失败，请稍后重试")
    print(
        f"[api] feedback: reason={record['reason']} "
        f"query={record['query'][:50]!r} result_id={record['result_id'][:40]!r}"
    )
    return {"code": 0, "message": "ok"}


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
