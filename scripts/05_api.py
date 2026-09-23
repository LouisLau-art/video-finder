#!/usr/bin/env python3
"""05_api.py — 视频检索 FastAPI 接口服务.

把 03_search.py 的检索能力封装为标准 HTTP 接口，供 Web 前端跨域调用。

接口:
    POST /api/search          文本搜视频帧 (请求体见 SearchRequest)
    POST /api/search-by-image 以图搜图 (multipart 表单: image 文件 + 可选 query/top_k；
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
    """单条 chroma 命中 -> 前端契约的 result 项。"""
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


def search_by_embedding(vec, k_want: int, label: str) -> dict[str, Any]:
    """共用检索：特征向量 -> 前端契约响应体（文本/以图搜图共用）。

    vec: 1xD 或 D 维向量（list / numpy 均可）；label: 响应 data.query。
    """
    import numpy as np

    t0 = time.perf_counter()
    try:
        _encoder, col = ensure_state()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    k = max(1, min(int(k_want), _CHROMA_COUNT))
    res = col.query(
        query_embeddings=arr.tolist(), n_results=k,
        include=["metadatas", "distances"],
    )
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    results = []
    for i, (m, d) in enumerate(zip(metas, dists), 1):
        score = 1.0 - float(d) / 2.0  # chroma cosine distance ∈ [0,2]
        results.append(build_result(i, score, dict(m)))

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
    top_k: int = Field(default=20, ge=1, le=100, description="返回前 K 个结果")
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
    out = search_by_embedding(q_emb[0], k_want, q)
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
