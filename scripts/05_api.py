#!/usr/bin/env python3
"""05_api.py — 视频检索 FastAPI 接口服务.

把 03_search.py 的检索能力封装为标准 HTTP 接口，供平台前端平台前端跨域调用。

接口:
    POST /api/search          文本搜视频帧 (请求体见 SearchRequest)
    GET  /api/frames/{name}   帧图片静态预览 (流式返回 jpg)
    GET  /api/health          健康检查 {"status": "ok", "frames_count": N}

复用:
    encoder 加载 (load_encoder) 与查询预处理 (resolve_query_for_encoder)
    均从 03_search.py 动态导入，保证与 CLI 一致。

向量库:
    --db 显式指定则用之；缺省按顺序探测存在者:
      1) data/local-runtime/eval_chroma/cnclip
      2) chroma_db

用法:
    uv run python scripts/05_api.py                       # 默认探测库，端口 8000
    uv run python scripts/05_api.py --db chroma_db --port 8000
    uv run python scripts/05_api.py --db data/local-runtime/eval_chroma/cnclip
"""
from __future__ import annotations

import argparse
import importlib.util
import time
import urllib.parse
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
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

# 默认探测顺序：中文原生 cnclip 索引优先，其次本地 chroma_db
DEFAULT_DB_CANDIDATES = [
    "data/local-runtime/eval_chroma/cnclip",
    "chroma_db",
]
# 帧图片目录探测顺序
DEFAULT_FRAMES_CANDIDATES = [
    "data/local-runtime/brand2_frames",
    "frames",
]

NAS_PREFIX = "share-a/0video"
NAS_HOST = "smb://nas.example.invalid"
SYNOLOGY_WEB_BASE = "http://nas.example.invalid:5000"

# 真实 NAS 相对路径映射表（从 NAS 真实目录树快照提取，覆盖当前全部 20 个评测素材）
NAS_PATH_MAP: dict[str, str] = {
    "clip-redvest": "share-a/0video/clip-redvest.mp4",
    "race-a": "share-a/0video/品牌/race-a.mp4",
    "clip-1010": "share-a/0video/品牌/clip-1010.mp4",
    "clip-story": "share-a/0video/folder-2026/clip-story.mp4",
    "clip-htc": "share-a/0video/品牌/documentary/clip-htc.mp4",
    "clip-steady": "share-a/0video/品牌/culture/clip-steady.mp4",
    "clip-newyear": "share-a/0video/品牌/culture/clip-newyear.mp4",
    "clip-rotate": "share-a/0video/folder-3d/folder-3d-proj/素材/clip-rotate.mp4",
    "item-yarn-a": "share-a/0video/products/0.2026/26item-y3D/yarn-item/item-yarn-a.mp4",
    "item-pants": "share-a/0video/products/0.2026/item-pants/item-pants.mp4",
    "clip-training-mix": "share-a/0video/products/0.2026/spring-training/clip-training-mix.mp4",
    "clip-person-a": "share-a/0video/products/0.2026/7月brand-runners-assets/clip-person-a.mp4",
    "clip-together": "share-a/0video/品牌/culture/folder-expo/screen/clip-together.mp4",
    "item-cap-a": "share-a/0video/products/0.2026/6.2item-cap/item-cap/item-cap-a.mp4",
    "item-tee-a": "share-a/0video/products/0.2026/26AW/加厚item-y圆领T恤 升级版/item-tee-a.mp4",
    "item-vest-a": "share-a/0video/products/0.2026/26AW/dir-vest-a/item-vest-a.mp4",
    "item-vest-b": "share-a/0video/products/0.2026/26AW/dir-vest-b/item-vest-b.mp4",
    "item-vest-c": "share-a/0video/products/0.2026/26AW/item-xdir-vest-c2.0/item-vest-c.mp4",
    "item-coat-a": "share-a/0video/products/0.2026/26AW/item-xdir-coat2.0/item-coat-a.mp4",
    "item-zip-a": "share-a/0video/products/0.2026/26AW/item-zdir-zip3.0/item-zip-a.mp4",
}

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
    """单条 chroma 命中 -> 前端平台契约的 result 项。"""
    score = max(min(float(score), 1.0), 0.0)
    video_id = str(meta.get("video_id", ""))
    video_name = f"{video_id}.mp4"
    t = float(meta.get("time", 0.0) or 0.0)
    frame_path = str(meta.get("frame_path", ""))
    filename = Path(frame_path).name

    # 优先查真实 NAS 完整全路径映射，无映射则兜底拼一级目录
    nas_path = NAS_PATH_MAP.get(video_id, f"{NAS_PREFIX}/{video_name}")

    # 群晖 DSM 标准深链协议：直接拉起 File Station 并自动定位展开所在文件夹
    # 路径需为双重 URL 编码（%252F...），且定位到所在文件夹（带末尾斜杠）
    folder_path = "/" + str(Path(nas_path).parent).replace("\\", "/") + "/"
    double_encoded_folder = urllib.parse.quote(urllib.parse.quote(folder_path, safe=""), safe="")
    synology_web_url = f"{SYNOLOGY_WEB_BASE}/index.cgi?launchApp=SYNO.SDS.App.FileStation3.Instance&launchParam=openfile%3D{double_encoded_folder}"
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
        "full_nas_uri": f"{NAS_HOST}/{nas_path}",
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


# ---------- FastAPI ----------

class SearchRequest(BaseModel):
    """POST /api/search 请求体（对齐前端平台 shared/api.interface.ts）。

    前端平台前端可能额外带 filter 等字段：extra=ignore 直接丢弃，保证向前兼容。
    """

    model_config = {"extra": "ignore", "populate_by_name": True}

    query: str = Field(..., description="搜索文本（中文自然语言）")
    top_k: int = Field(default=20, ge=1, le=100, description="返回前 K 个结果")
    # 兼容 topk 别名写法
    topk: int | None = Field(default=None, ge=1, le=100, exclude=True)


app = FastAPI(title="视频检索 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 平台前端平台网页直接跨域调用
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

    t0 = time.perf_counter()
    try:
        encoder, col = ensure_state()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    m3 = load_search_module()
    resolved, _translated = m3.resolve_query_for_encoder(
        q, _ENCODER_NAME, translate=APP_TRANSLATE
    )
    q_emb = encoder.encode_texts([resolved])
    k = max(1, min(int(k_want), _CHROMA_COUNT))
    res = col.query(
        query_embeddings=q_emb.tolist(), n_results=k,
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
            "query": q,
            "total": len(results),
            "elapsed_ms": elapsed_ms,
            "encoder": _ENCODER_NAME,
            "results": results,
        },
    }


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
    if not APP_FRAMES_DIRS:
        print("[warn] 未找到帧图片目录，/api/frames 将全部 404")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
