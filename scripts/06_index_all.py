#!/usr/bin/env python3
"""06_index_all.py — 全量 NAS 视频批量索引管线（抽帧 -> CN-CLIP -> Chroma）。

把 --roots 下的全部视频抽帧成 10 万级帧并灌入 Chroma，要求断点续跑、可并行、
先小样验证。抽帧复用 scripts/01_extract.process_video（本文件不改它），
编码复用 scripts/02_embed_index.build_encoder（默认 cnclip 中文原生）。

布局（与评测库 eval_chroma/<encoder> 结构对齐）:
    --frames-dir  帧落盘目录（默认 data/local-runtime/frames_full）
    --manifest    明细（默认 <frames-dir>/manifest.jsonl，增量 append）
    --db          索引根目录（默认 data/local-runtime/index_full）
        <db>/state.json       断点状态（video_key -> {status,mtime,size,frames}）
        <db>/<encoder>/       Chroma PersistentClient 路径（默认 <db>/cnclip，
                              正好是 05_api 默认探测的首选 db）

唯一命名：video_key = md5(f"{share}/{relpath}")[:8]，
    帧文件 f"{video_key}_{stem}_{t:.1f}.jpg"，chroma id f"{video_key}_{stem}@{t:.1f}"，
    避免不同目录同名视频撞名。

并行：multiprocessing 池跑抽帧（ffmpeg/场景切分，CPU 密集，默认 3 worker，
    每 worker os.nice(10) 限速）；主进程按 64 帧一批 encode+upsert
   （torch.set_num_threads(6)），单 worker 模型只加载一次。

用法:
    .venv/bin/python scripts/06_index_all.py --dry-run            # 只统计，不写盘
    .venv/bin/python scripts/06_index_all.py --limit 3 \\         # 小样验证
        --frames-dir /tmp/idx_smoke/frames \\
        --manifest /tmp/idx_smoke/manifest.jsonl --db /tmp/idx_smoke/chroma
    .venv/bin/python scripts/06_index_all.py                       # 全量（先小样！）
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

VIDEO_EXTS = {".mp4", ".mov", ".m4v"}  # 大小写不敏感

# 代码依赖目录剪枝（精确小写匹配）：视频管线无需进入，CIFS 上枚举极慢
# （实测：个人目录下的 sd-webui 整套 site-packages 让遍历卡死，约 3s/次响应）。
# 隐藏目录（.git/.venv 等）已由“隐藏段”规则覆盖，此处只列非隐藏的代码目录。
PRUNE_DIRNAMES = {
    "site-packages", "node_modules", "__pycache__", "venv",
    ".tox", ".eggs", "__pypackages__", "pip-wheel-metadata",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    # 群晖系统目录：缩略图缓存（每个目录一个，是目录数的主要水分）、
    # 回收站（多为已删重复文件，且只读挂载下常 Permission denied）、快照
    "@eadir", "#recycle", "#snapshot",
}

# 本地挂载点 -> 共享名：由仓库外的 site.json 的 shares 派生；
# 无配置时为空，share_of 兜底用目录名（不影响本地最小链路）。
def _load_share_map() -> dict:
    """shares 列表 -> {mount: name}；失败返回空 dict。"""
    try:
        spec = importlib.util.spec_from_file_location(
            "site_config", str(SCRIPTS / "site_config.py"))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        out: dict = {}
        for s in mod.load().get("shares", []) or []:
            if isinstance(s, dict) and s.get("mount") and s.get("name"):
                out[str(s["mount"]).rstrip("/")] = str(s["name"])
        return out
    except Exception:
        return {}


SHARE_MAP = _load_share_map()


def _default_roots() -> list[str]:
    """--roots 默认值：site.json 的 shares 挂载点列表；无配置时退回本地默认。"""
    if SHARE_MAP:
        return sorted(SHARE_MAP)
    return ["/mnt/nas"]

DEFAULT_FRAMES_DIR = "data/local-runtime/frames_full"
DEFAULT_DB = "data/local-runtime/index_full"
DEFAULT_COLLECTION = "frames"

# 连续 0 帧/失败熔断阈值（疑似 NAS 掉线时中止，避免空转写脏 state）
MAX_CONSECUTIVE_EMPTY = 20


# ---------- 发现 ----------

def share_of(root: str) -> str:
    """挂载点 -> 共享名；未知根目录回退用其 basename。"""
    return SHARE_MAP.get(root.rstrip("/"), Path(root).name)


def discover_videos(roots: list[str], include_dir: list[str], exclude_dir: list[str],
                    min_size_mb: float) -> list[dict]:
    """递归发现视频。返回 [{video_path, share, relpath, stem, size, mtime}]，按体积升序。

    跳过：.fcpbundle 目录（含其子树）、任何隐藏路径段（以 . 开头）、
    代码依赖目录（site-packages/node_modules/__pycache__ 等）、
    小于 min-size 的文件、非 mp4/mov/m4v 后缀。
    include/exclude：relpath（posix）的子串包含匹配，可重复。
    """
    min_bytes = int(min_size_mb * 1024 * 1024)
    out: list[dict] = []
    n_dirs = 0
    for root in roots:
        r = Path(root)
        if not r.is_dir():
            print(f"[warn] 根目录不存在跳过: {root}", file=sys.stderr)
            continue
        share = share_of(root)
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: print(
                f"[warn] 遍历失败 {e.filename}: {e.strerror}", file=sys.stderr)):
            n_dirs += 1
            if n_dirs % 500 == 0:
                print(f"[scan] 已扫 {n_dirs} 目录，已发现 {len(out)} 视频…",
                      file=sys.stderr, flush=True)
            # 修剪：.fcpbundle 包 + 隐藏目录 + 代码依赖目录（原地改 dirnames 才生效）
            dirnames[:] = [d for d in dirnames
                           if not d.endswith(".fcpbundle") and not d.startswith(".")
                           and d.lower() not in PRUNE_DIRNAMES]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                if p.suffix.lower() not in VIDEO_EXTS:
                    continue
                try:
                    rel = p.relative_to(r).as_posix()
                except ValueError:
                    continue
                if any(seg.startswith(".") for seg in rel.split("/")):
                    continue
                if include_dir and not any(s in rel for s in include_dir):
                    continue
                if any(s in rel for s in exclude_dir):
                    continue
                try:
                    st = p.stat()
                except OSError as e:
                    print(f"[warn] stat 失败跳过 {p}: {e}", file=sys.stderr)
                    continue
                if st.st_size < min_bytes:
                    continue
                out.append({"video_path": str(p), "share": share, "relpath": rel,
                            "stem": p.stem, "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda v: (v["size"], v["relpath"]))
    return out


def video_key_of(share: str, relpath: str) -> str:
    return hashlib.md5(f"{share}/{relpath}".encode("utf-8")).hexdigest()[:8]


def check_roots_readable(roots: list[str]) -> list[str]:
    """启动前置检查：每个 --roots 根目录必须可读且非空。

    返回错误信息列表（空表示全过）。NAS 掉线时挂载点常表现为：
    不存在 / 不可列目录 / 空目录，任一命中都应拒绝开工，避免空转写脏 state。
    """
    errs: list[str] = []
    for root in roots:
        p = Path(root)
        if not p.is_dir():
            errs.append(
                f"[error] 根目录不可用: {root}（不存在或不是目录）。"
                "NAS 可能未挂载，请先运行 scripts/mount_nas.sh 确认挂载后再跑。")
            continue
        try:
            entries = os.listdir(root)
        except OSError as e:
            errs.append(
                f"[error] 根目录不可读: {root}（{type(e).__name__}: {e}）。"
                "可能是 NAS 掉线/权限丢失，请先运行 scripts/mount_nas.sh 确认挂载后再跑。")
            continue
        if not entries:
            errs.append(
                f"[error] 根目录为空: {root}（列目录成功但无任何条目）。"
                "NAS 可能未挂载或挂载点被遮挡，请先运行 scripts/mount_nas.sh 确认挂载后再跑。")
    return errs


# ---------- worker（抽帧，只在子进程跑；模块动态加载，避免 pickle 传模块对象） ----------


def _worker_init(nice: int) -> None:
    try:
        os.nice(int(nice))
    except Exception as e:  # noqa: BLE001 — nice 失败不阻断抽帧
        print(f"[warn] os.nice({nice}) 失败: {e}", file=sys.stderr)


def _load_extract_module():
    spec = importlib.util.spec_from_file_location(
        "mod_01_extract", str(SCRIPTS / "01_extract.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _extract_job(job: dict) -> dict:
    """抽一支视频：staging 抽帧 -> 重命名搬运 -> 返回 manifest 行。

    job: {video_path, share, relpath, video_key, stem, frames_dir, fps, max_per_scene}
    return: {video_key, ok, rows|error, frames}
    """
    vkey = job["video_key"]
    stem = job["stem"]
    frames_dir = Path(job["frames_dir"])
    staging = frames_dir / f".staging_{vkey}"
    try:
        m01 = _load_extract_module()
        frames_dir.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        # 复用 01_extract.process_video（不改它）：先落 staging，避免同名 stem 并行相撞
        raw_rows = m01.process_video(
            Path(job["video_path"]), staging, job["fps"], job["max_per_scene"])
        rows: list[dict] = []
        for r in raw_rows:
            t = float(r["time"])
            dst_name = f"{vkey}_{stem}_{t:.1f}.jpg"
            dst = frames_dir / dst_name
            shutil.move(str(r["frame_path"]), str(dst))
            rows.append({"video_key": vkey, "video_id": stem,
                         "share": job["share"], "relpath": job["relpath"],
                         "video_path": job["video_path"],
                         "frame_path": str(dst.as_posix()), "time": t})
        rows.sort(key=lambda x: x["time"])
        shutil.rmtree(staging, ignore_errors=True)
        if not rows:
            # 0 帧时轻量探测：区分「真无画面」与「文件不可读（NAS 抖动/掉线）」。
            # stat 失败 -> 记 failed（续跑自动重试，不污染 state）；
            # stat 正常 -> 真无画面，保持 done+0帧，并打 empty_confirmed 供续跑跳过。
            try:
                os.stat(job["video_path"])
            except OSError as e:
                return {"video_key": vkey, "ok": False,
                        "error": f"unreadable: {type(e).__name__}: {e}",
                        "rows": [], "frames": 0}
            return {"video_key": vkey, "ok": True, "rows": rows, "frames": 0,
                    "empty_confirmed": True}
        return {"video_key": vkey, "ok": True, "rows": rows, "frames": len(rows)}
    except Exception as e:  # noqa: BLE001 — 单视频异常由主进程记 failed，不中断
        shutil.rmtree(staging, ignore_errors=True)
        return {"video_key": vkey, "ok": False, "error": f"{type(e).__name__}: {e}",
                "rows": [], "frames": 0}


# ---------- 状态 / manifest ----------

def load_state(db_dir: Path) -> dict:
    p = db_dir / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] state.json 损坏，从空开始（旧文件改名保留）: {e}",
                  file=sys.stderr)
            p.rename(p.with_suffix(".json.corrupt"))
    return {}


def save_state(db_dir: Path, state: dict) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    tmp = db_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.rename(db_dir / "state.json")


def load_manifest_keys(manifest: Path) -> set[str]:
    keys: set[str] = set()
    if manifest.exists():
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.add(json.loads(line).get("video_key", ""))
                except Exception:
                    continue
    return keys


def manifest_rows_for_done_frames(video: dict, vkey: str, stem: str,
                                  frames_dir: Path) -> list[dict]:
    """已 done 但 manifest 缺行时，按落盘帧文件名重建 manifest 行（不重抽）。"""
    prefix = f"{vkey}_{stem}_"
    rows: list[dict] = []
    for p in sorted(frames_dir.glob(f"{vkey}_*.jpg")):
        name = p.name
        if not name.startswith(prefix) or not name.endswith(".jpg"):
            continue
        try:
            t = float(name[len(prefix):-4])
        except ValueError:
            continue
        rows.append({"video_key": vkey, "video_id": stem,
                     "share": video["share"], "relpath": video["relpath"],
                     "video_path": video["video_path"],
                     "frame_path": str(p.as_posix()), "time": t})
    rows.sort(key=lambda x: x["time"])
    return rows


# ---------- 主流程 ----------

def fmt_gb(n: int) -> str:
    return f"{n / (1024 ** 3):.1f}GB"


def main() -> int:
    ap = argparse.ArgumentParser(description="全量 NAS 视频批量索引（抽帧->CN-CLIP->Chroma）")
    ap.add_argument("--roots", nargs="+", default=_default_roots())
    ap.add_argument("--include-dir", action="append", default=[],
                    help="只收录 relpath 含该子串的视频（可重复）")
    ap.add_argument("--exclude-dir", action="append", default=[],
                    help="排除 relpath 含该子串的视频（可重复）")
    ap.add_argument("--min-size-mb", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    ap.add_argument("--manifest", default=None, help="默认 <frames-dir>/manifest.jsonl")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（体积升序，小样验证用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    ap.add_argument("--force", action="store_true", help="忽略断点状态，全部重跑")
    ap.add_argument("--max-consecutive-empty", type=int, default=MAX_CONSECUTIVE_EMPTY,
                    help="连续 0 帧/失败达到该数即熔断停机（疑似 NAS 掉线），默认 20")
    # 以下为管线必需但任务未列出的配套旋钮（均取与现有脚本一致的默认值）：
    ap.add_argument("--model", default="cnclip",
                    choices=["siglip2", "cnclip", "openclip", "dummy"])
    ap.add_argument("--batch-size", type=int, default=64, help="每批 encode+upsert 帧数")
    ap.add_argument("--fps", type=float, default=1.0, help="回退模式帧率（透传 01）")
    ap.add_argument("--max-per-scene", type=int, default=3, help="每场景最多帧数（透传 01）")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    manifest = Path(args.manifest) if args.manifest else frames_dir / "manifest.jsonl"
    db_dir = Path(args.db)

    # 启动前置检查（dry-run 同样执行：它也要遍历目录，根不可用时直接退出）。
    root_errs = check_roots_readable(args.roots)
    if root_errs:
        for e in root_errs:
            print(e, file=sys.stderr)
        return 1

    t_start = time.time()
    videos = discover_videos(args.roots, args.include_dir, args.exclude_dir,
                             args.min_size_mb)
    for v in videos:
        v["video_key"] = video_key_of(v["share"], v["relpath"])
    if args.limit and args.limit > 0:
        videos = videos[:args.limit]

    total_bytes = sum(v["size"] for v in videos)
    if args.dry_run:
        from collections import Counter
        bydir: Counter[str] = Counter()
        bydir_n: Counter[str] = Counter()
        for v in videos:
            top = v["relpath"].split("/")[0] if "/" in v["relpath"] else "."
            d = f"{v['share']}/{top}"
            bydir[d] += v["size"]
            bydir_n[d] += 1
        print(f"[dry-run] 视频 {len(videos)} 个，总体积 {fmt_gb(total_bytes)} "
              f"({total_bytes} 字节)，平均 {total_bytes / max(len(videos), 1) / 1024 / 1024:.1f}MB/个")
        print("[dry-run] 按顶层目录体积 top10:")
        for d, b in bydir.most_common(10):
            print(f"    {fmt_gb(b):>9}  {bydir_n[d]:>5}个  {d}")
        if videos:
            print("[dry-run] 体积最小 3 个（--limit 3 将选中它们）:")
            for v in videos[:3]:
                print(f"    {v['size'] / 1024 / 1024:8.1f}MB  {v['share']}/{v['relpath']}")
        return 0

    if not videos:
        print("未发现视频，先检查 --roots / --include-dir。")
        return 1

    state = load_state(db_dir)
    manifest_keys = load_manifest_keys(manifest)
    man_f = None

    def append_manifest(rows: list[dict]) -> None:
        """整视频粒度去重 append：该 video_key 已记则整批跳过。"""
        nonlocal man_f
        if not rows or rows[0]["video_key"] in manifest_keys:
            return
        if man_f is None:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            man_f = manifest.open("a", encoding="utf-8")
        for r in rows:
            man_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            manifest_keys.add(r["video_key"])
        man_f.flush()

    # 断点分类
    todo: list[dict] = []
    n_skip = 0
    skip_frames = 0
    for v in videos:
        vkey = v["video_key"]
        st = state.get(vkey)
        # 续跑跳过三条件：状态 done、无“待 upsert”尾巴（崩溃残留必须重跑补向量）、
        # 帧文件齐全；失败/缺帧/强制都重跑。
        if st and st.get("status") == "done" and not st.get("pending_upsert") \
                and not args.force:
            names = st.get("frames") or []
            if not names:
                # 0 帧 done 默认重跑（all([])==True 会永久跳过，此处显式规避）；
                # 只有打了 empty_confirmed（真无画面）才跳过，避免无限重试纯色卡视频。
                if st.get("empty_confirmed"):
                    n_skip += 1
                    continue
                todo.append(v)
                continue
            if all((frames_dir / n).is_file() for n in names):
                n_skip += 1
                skip_frames += len(names)
                if vkey not in manifest_keys:
                    # 帧齐但 manifest 缺行（如 manifest 被删）：重建行，不重抽不重编码
                    append_manifest(manifest_rows_for_done_frames(
                        v, vkey, v["stem"], frames_dir))
                continue
        todo.append(v)
    # --force 时先把待重跑 video_key 的旧 manifest 行滤掉，避免重复行越积越多
    if args.force and todo:
        redo = {v["video_key"] for v in todo}
        if manifest.exists():
            lines = manifest.read_text(encoding="utf-8").splitlines()
            kept = [l for l in lines
                    if json.loads(l).get("video_key") not in redo] if lines else []
            if len(kept) != len(lines):
                manifest.write_text("\n".join(kept) + ("\n" if kept else ""),
                                    encoding="utf-8")
            manifest_keys = load_manifest_keys(manifest)

    print(f"[plan] 发现 {len(videos)} 个视频（{fmt_gb(total_bytes)}），"
          f"待处理 {len(todo)}，跳过 {n_skip}（已索引 {skip_frames} 帧）")
    if not todo:
        if man_f:
            man_f.close()
        print("[done] 无待处理视频，全部已索引。")
        return 0

    # 主进程：torch 限线程 + 一次性加载 encoder（worker 子进程不碰模型）
    import torch
    torch.set_num_threads(6)
    importlib_spec = importlib.util.spec_from_file_location(
        "mod_02_embed_index", str(SCRIPTS / "02_embed_index.py"))
    assert importlib_spec is not None and importlib_spec.loader is not None
    m02 = importlib.util.module_from_spec(importlib_spec)
    importlib_spec.loader.exec_module(m02)  # type: ignore[union-attr]
    encoder = m02.build_encoder(args.model)

    import chromadb
    from PIL import Image
    chroma_path = str(db_dir / args.model)  # <db>/<encoder>，与 eval 布局一致
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_or_create_collection(
        name=args.collection, metadata={"hnsw:space": "cosine"})

    jobs = [{"video_path": v["video_path"], "share": v["share"], "relpath": v["relpath"],
             "video_key": v["video_key"], "stem": v["stem"],
             "frames_dir": str(frames_dir),
             "fps": args.fps, "max_per_scene": args.max_per_scene} for v in todo]
    by_key = {v["video_key"]: v for v in todo}

    def flush_batch(pending: list[tuple[str, list[dict]]]) -> tuple[int, list[str]]:
        """一批 rows -> encode+upsert。返回 (成功帧数, [成功video_key])；整批失败则记 failed。"""
        flat: list[dict] = [r for _, rs in pending for r in rs]
        if not flat:
            return 0, [k for k, _ in pending]
        try:
            imgs: list[Image.Image] = []
            keep: list[dict] = []
            for r in flat:
                p = Path(r["frame_path"])
                if not p.is_file():
                    print(f"[warn] 帧文件丢失跳过: {p}", file=sys.stderr)
                    continue
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    keep.append(r)
                except Exception as e:
                    print(f"[warn] 读图失败跳过 {p}: {e}", file=sys.stderr)
            if not keep:
                return 0, [k for k, _ in pending]
            embs = encoder.encode_images(imgs)
            ids = [f"{r['video_key']}_{r['video_id']}@{r['time']:.1f}" for r in keep]
            metas = [{"video_id": r["video_id"], "video_name": r["video_id"] + ".mp4",
                      "time": float(r["time"]), "frame_path": str(r["frame_path"]),
                      "video_path": str(r["video_path"]),
                      "share": r["share"], "relpath": r["relpath"]} for r in keep]
            docs = [f"{r['video_id']} @ {r['time']:.1f}s" for r in keep]
            col.upsert(ids=ids, embeddings=embs.tolist(),
                       metadatas=metas, documents=docs)
            return len(keep), [k for k, _ in pending]
        except Exception as e:  # noqa: BLE001 — 整批失败，主进程记录后继续
            print(f"[error] encode/upsert 整批失败（{len(flat)} 帧）: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            for k, _ in pending:
                v = by_key[k]
                state[k] = {"status": "failed", "mtime": v["mtime"], "size": v["size"],
                            "frames": [], "error": f"batch: {type(e).__name__}: {e}"}
            save_state(db_dir, state)
            return 0, []

    n_done = 0
    n_failed = 0
    cum_frames = 0
    pending: list[tuple[str, list[dict]]] = []
    pending_frames = 0
    # 连续 0 帧/失败计数（按 imap 返回顺序统计）：成功（frames>0）即清零，
    # 达到阈值即判定疑似 NAS 掉线，熔断停机。
    consecutive_empty = 0
    fused = False
    max_empty = max(int(args.max_consecutive_empty), 1)
    t0 = time.time()
    workers = max(int(args.workers), 1)
    with Pool(workers, initializer=_worker_init, initargs=(args.nice,)) as pool:
        for res in pool.imap_unordered(_extract_job, jobs):
            vkey = res["video_key"]
            v = by_key[vkey]
            done_idx = n_done + n_failed + 1
            if res.get("ok"):
                rows = res["rows"]
                append_manifest(rows)
                cum_frames += len(rows)
                pending.append((vkey, rows))
                pending_frames += len(rows)
                if pending_frames >= args.batch_size:
                    _ok_frames, ok_keys = flush_batch(pending)
                    for k in ok_keys:
                        vv = by_key[k]
                        krows = next(rs for kk, rs in pending if kk == k)
                        state[k] = {"status": "done", "mtime": vv["mtime"],
                                    "size": vv["size"],
                                    "frames": [Path(r["frame_path"]).name for r in krows]}
                        if not krows:
                            # 真无画面（worker 已 stat 确认）：打标，续跑可跳过
                            state[k]["empty_confirmed"] = True
                    save_state(db_dir, state)
                    pending = []
                    pending_frames = 0
                else:
                    # 未到一批：先记 done（帧已落盘+manifest 已记），向量在 flush 时补；
                    # pending_upsert 标记崩溃残留，下次续跑强制重跑补向量。
                    state[vkey] = {"status": "done", "mtime": v["mtime"], "size": v["size"],
                                   "frames": [Path(r["frame_path"]).name for r in rows],
                                   "pending_upsert": True}
                    if not rows:
                        state[vkey]["empty_confirmed"] = True
                    save_state(db_dir, state)
                n_done += 1
                # 熔断计数：真无画面（0 帧）与失败同等计入，成功（>0 帧）清零
                if res.get("frames", 0) > 0:
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
            else:
                state[vkey] = {"status": "failed", "mtime": v["mtime"], "size": v["size"],
                               "frames": [], "error": res.get("error", "unknown")}
                save_state(db_dir, state)
                n_failed += 1
                consecutive_empty += 1
            el = time.time() - t0
            rate = cum_frames / el if el > 0 else 0.0
            avg_per_vid = cum_frames / max(done_idx, 1)
            remain_vid = len(todo) - done_idx
            eta = (remain_vid * avg_per_vid / rate) if rate > 0 else -1
            eta_s = f"{eta / 60:.0f}分" if eta >= 0 and eta < 7200 else "--"
            print(f"[progress] {done_idx}/{len(todo)} {vkey} {v['stem'][:28]} "
                  f"{res.get('frames', 0)}帧 | 累计{cum_frames}帧/{el / 60:.1f}分 "
                  f"{rate:.1f}帧/秒 预计剩余{eta_s}"
                  + ("" if res.get("ok") else f" [FAILED] {res.get('error')}"),
                  flush=True)
            if consecutive_empty >= max_empty:
                print(f"[FUSE] 连续 {consecutive_empty} 个视频 0 帧/失败，"
                      "疑似 NAS 掉线或后端存储不可用，已中止提交新任务。",
                      file=sys.stderr, flush=True)
                print(f"[FUSE] state 已保存（done={n_done} failed={n_failed}），"
                      "确认 NAS 挂载恢复后直接重跑本命令即可续跑。",
                      file=sys.stderr, flush=True)
                fused = True
                break
    # 尾批（不足一批的剩余帧）：成功则去掉 pending_upsert 转正，失败已在内部记 failed
    if pending:
        _ok_frames, ok_keys = flush_batch(pending)
        ok_set = set(ok_keys)
        for k, krows in pending:
            if k in ok_set:
                vv = by_key[k]
                state[k] = {"status": "done", "mtime": vv["mtime"], "size": vv["size"],
                            "frames": [Path(r["frame_path"]).name for r in krows]}
                if not krows:
                    state[k]["empty_confirmed"] = True
        save_state(db_dir, state)
    if man_f:
        man_f.close()

    # encoder.json 写进 chroma 子目录（与 02/03/API 的 load_encoder 约定一致）
    (Path(chroma_path) / "encoder.json").write_text(
        json.dumps({"encoder": encoder.name, "model_id": encoder.model_id,
                    "dim": int(encoder.dim)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    el = time.time() - t_start
    rate = cum_frames / el if el > 0 else 0.0
    print(f"[summary] done={n_done} failed={n_failed} skipped={n_skip} "
          f"帧数={cum_frames}(+跳过{skip_frames}) 耗时={el / 60:.1f}分 平均帧速={rate:.1f}帧/秒")
    print(f"[summary] 帧->{frames_dir} manifest->{manifest} "
          f"chroma->{chroma_path} collection={args.collection}")
    if fused:
        return 3
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
