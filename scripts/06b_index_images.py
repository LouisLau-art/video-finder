#!/usr/bin/env python3
"""06b_index_images.py — NAS 图片批量索引管线（发现 -> CN-CLIP -> Chroma）。

把 --roots 下的静态图片（jpg/jpeg/png/webp）编码进与视频帧同一个
Chroma collection（frames），每张图一条向量，与视频帧同库混排检索。
结构与约定全部对齐 scripts/06_index_all.py（下称 06）：
编码复用 scripts/02_embed_index.build_encoder（不新写 encoder），
目录剪枝/身份键/断点 state/熔断模式均与 06 同构。

布局:
    --db      Chroma PersistentClient 路径（默认 data/local-runtime/index_full/cnclip，
              即 05_api 默认探测的首选 db；encoder.json 同样写在此处）
    --state   断点状态文件（默认 data/local-runtime/index_full/state_images.json，
              与视频 state.json 分离，schema 同构：image_key -> {status,mtime,size,frame_path}）

唯一命名（与 06 的 video_key 同构）：
    image_key = md5(f"{share}/{relpath}")[:8]
    chroma id = f"{image_key}_{stem}@0.0"（time 恒 0.0，对齐帧 id 格式）

批处理：每 batch-size 张一批，批内线程池并行 PIL 解码+缩放（--workers），
主线程整批 encode+upsert（torch.set_num_threads(6)）。

用法:
    .venv/bin/python scripts/06b_index_images.py --dry-run   # 只统计，不写盘
    .venv/bin/python scripts/06b_index_images.py --limit 200 \\
        --db /tmp/img_smoke/chroma --state /tmp/img_smoke/state_images.json
    .venv/bin/python scripts/06b_index_images.py             # 全量（先小样！）
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # 大小写不敏感

# 目录剪枝（精确小写匹配）：与 06 的 PRUNE_DIRNAMES 完全一致（来源见 06 注释）：
# 代码依赖目录、群晖系统目录（缩略图缓存 @eadir / 回收站 / 快照），
# 隐藏目录（.git/.venv 等）由"隐藏段"规则覆盖。
PRUNE_DIRNAMES = {
    "site-packages", "node_modules", "__pycache__", "venv",
    ".tox", ".eggs", "__pypackages__", "pip-wheel-metadata",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "@eadir", "#recycle", "#snapshot",
}

DEFAULT_DB = "data/local-runtime/index_full/cnclip"
DEFAULT_STATE = "data/local-runtime/index_full/state_images.json"
DEFAULT_COLLECTION = "frames"

# 连续失败熔断阈值（与 06 的 MAX_CONSECUTIVE_EMPTY 同一处置思路：
# 疑似 NAS 掉线时中止，避免空转写脏 state）
MAX_CONSECUTIVE_FAILED = 20
# 磁盘保护：写库前可用空间低于该值即停机
MIN_FREE_GB = 5.0
# 编码前最长边超过该值的图片等比缩小（省时省内存；缩略图服务端点直接用原图）
DEFAULT_MAX_SIDE = 1024


class LowDiskSpace(RuntimeError):
    """可用磁盘空间低于保护阈值，携带当前读数交给主流程安全退出。"""

    def __init__(self, free_gb, min_free_gb: float = MIN_FREE_GB):
        try:
            self.free_gb = float(free_gb)
            self.min_free_gb = float(min_free_gb)
            msg = f"可用空间 {self.free_gb:.1f}GB 低于下限 {self.min_free_gb:.1f}GB"
        except (TypeError, ValueError):
            # 允许直接传完整消息（测试桩/自定义触发场景）
            self.free_gb = 0.0
            self.min_free_gb = MIN_FREE_GB
            msg = str(free_gb)
        super().__init__(msg)


# ---------- 配置 / 共享名 ----------

def _load_share_map() -> dict:
    """shares 列表 -> {mount: name}；失败返回空 dict（与 06 同款）。"""
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


def share_of(root: str) -> str:
    """挂载点 -> 共享名；未知根目录回退用其 basename。"""
    return SHARE_MAP.get(root.rstrip("/"), Path(root).name)


# ---------- 发现 ----------

def discover_images(roots: list[str], include_dir: list[str], exclude_dir: list[str],
                    min_size_mb: float = 0.0) -> list[dict]:
    """递归发现图片。返回 [{image_path, share, relpath, name, stem, size, mtime}]，
    按体积升序（--limit 小样验证优先小文件，与 06 一致）。

    跳过：.fcpbundle 目录（含其子树）、任何隐藏路径段（以 . 开头）、
    剪枝目录（PRUNE_DIRNAMES）、0 字节文件（必损坏）、
    非 jpg/jpeg/png/webp 后缀（psd/ai/tif/heic 等源文件不入库）。
    include/exclude：relpath（posix）的子串包含匹配，可重复（与 06 语义一致）。
    """
    min_bytes = max(int(min_size_mb * 1024 * 1024), 1)  # 0 字节文件一律跳过
    out: list[dict] = []
    n_dirs = 0
    n_ext_skip = 0
    for root in roots:
        r = Path(root)
        if not r.is_dir():
            print(f"[warn] 根目录不存在跳过: {root}", file=sys.stderr)
            continue
        share = share_of(root)
        for dirpath, dirnames, filenames in os_walk(root):
            n_dirs += 1
            if n_dirs % 500 == 0:
                print(f"[scan] 已扫 {n_dirs} 目录，已发现 {len(out)} 图片…",
                      file=sys.stderr, flush=True)
            # 修剪：.fcpbundle 包 + 隐藏目录 + 剪枝目录（原地改 dirnames 才生效）
            dirnames[:] = [d for d in dirnames
                           if not d.endswith(".fcpbundle") and not d.startswith(".")
                           and d.lower() not in PRUNE_DIRNAMES]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                if p.suffix.lower() not in IMAGE_EXTS:
                    n_ext_skip += 1
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
                out.append({"image_path": str(p), "share": share, "relpath": rel,
                            "name": p.name, "stem": p.stem,
                            "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda v: (v["size"], v["relpath"]))
    print(f"[filter] 扫描 {n_dirs} 目录，图片 {len(out)} 张"
          f"（非图片扩展名跳过 {n_ext_skip} 个文件）", flush=True)
    return out


def os_walk(root: str):
    """os.walk 包装：遍历失败打 warn 继续（与 06 的 onerror 行为一致）。"""
    return os.walk(root, onerror=lambda e: print(
        f"[warn] 遍历失败 {e.filename}: {e.strerror}", file=sys.stderr))


def image_key_of(share: str, relpath: str) -> str:
    """md5(f"{share}/{relpath}")[:8]，与 06 的 video_key_of 完全同构。"""
    return hashlib.md5(f"{share}/{relpath}".encode("utf-8")).hexdigest()[:8]


def check_roots_readable(roots: list[str]) -> list[str]:
    """启动前置检查：每个 --roots 根目录必须可读且非空（与 06 同款）。"""
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


# ---------- 编码 ----------

def _resampling():
    """Pillow 新旧版本统一的 LANCZOS 重采样常量。"""
    from PIL import Image
    return getattr(Image, "Resampling", Image).LANCZOS


def open_image(path: str, max_side: int = DEFAULT_MAX_SIDE):
    """PIL 打开图片并转 RGB；最长边超过 max_side 时等比缩小（只缩不放）。

    返回 (img, 原宽, 原高)。metadata 记录原始宽高，编码用缩小后的图；
    缩略图服务端点直接用原图文件，显示不受影响。调用方负责 close。
    """
    from PIL import Image
    with Image.open(path) as source:
        img = source.convert("RGB")
    w, h = img.size
    if max_side > 0 and max(w, h) > max_side:
        img.thumbnail((max_side, max_side), _resampling())
    return img, w, h


def image_id(image_key: str, stem: str) -> str:
    """Chroma id：{image_key}_{stem}@0.0，对齐帧 id 格式、time 恒 0.0。"""
    return f"{image_key}_{stem}@0.0"


def build_meta(item: dict, width: int, height: int) -> dict:
    """构造 upsert 用的 metadata（字段集合是图片结果行的入库契约）。"""
    video_name = item.get("name") or Path(item["relpath"]).name
    return {
        "image_key": item["image_key"],
        "video_id": item["stem"],          # 兼容字段：与视频行的 video_id 对齐
        "video_name": video_name,          # 兼容字段：stem + 扩展名
        "media_type": "image",
        "time": 0.0,
        "image_path": str(item["image_path"]),
        "width": int(width),
        "height": int(height),
        "share": item["share"],
        "relpath": item["relpath"],
        "frame_path": str(item["image_path"]),  # 兼容现有契约：指向原图
    }


def document_text(meta: dict) -> str:
    """document 文本：与 06 的 f"{stem} @ {t}s" 同构，标注图片类型。"""
    return f"{meta['video_id']} @ 0.0s (image)"


def process_image(item: dict, encoder, collection, max_side: int = DEFAULT_MAX_SIDE) -> dict:
    """单图完整处理：打开 -> encode -> upsert。返回 {ok, meta, state}。

    state 条目与 06 的 state schema 同构：{status, mtime, size, frame_path}。
    """
    img, w, h = open_image(item["image_path"], max_side)
    try:
        meta = build_meta(item, w, h)
        embs = encoder.encode_images([img])
        emb0 = embs[0]
        if hasattr(emb0, "tolist"):
            emb0 = emb0.tolist()
        collection.upsert(
            ids=[image_id(item["image_key"], item["stem"])],
            embeddings=[emb0],
            metadatas=[meta],
            documents=[document_text(meta)],
        )
        return {"ok": True, "meta": meta,
                "state": {"status": "done", "mtime": item["mtime"],
                          "size": item["size"], "frame_path": item["image_path"]}}
    finally:
        img.close()


# ---------- 批处理编排 ----------

def iter_batches(items: list, batch_size: int):
    """按固定上限切分列表（与 06 同款）。"""
    size = max(int(batch_size), 1)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def run_processing(items: list, encoder=None, collection=None, process_one=None,
                   batch_size: int = 64, max_consecutive_failures: int = MAX_CONSECUTIVE_FAILED,
                   save=None, disk_check=None, workers: int = 1) -> dict:
    """批驱动编排：每批先并行准备（process_one），攒满整批 encode+upsert。

    - process_one(item, encoder, collection) -> {ok, img?, meta?, state?, error?}：
      返回 img+meta 的项进入当前批，攒满 batch_size 或批尾统一 encode+upsert；
      不返回 img 的成功项视为已完成（供测试桩与轻量路径使用）。
    - 失败（process_one 异常/返回 ok=False、或整批 encode+upsert 失败）计入
      连续失败计数；成功清零；达到 max_consecutive_failures 即熔断停机。
    - disk_check 抛 LowDiskSpace 即停机（stopped=True），不再处理后续项。
    - save：每批结束后回调 save(new_results)（本批新增的结果切片，供调用方
      增量折算断点 state 并写盘）；None 表示不回调。

    返回 {processed, done, failed, fused, stopped, results}；
    results 每项 {image_key, ok, state?/error?}，state 为该图的最终 state 条目。
    """
    batch_size = max(int(batch_size), 1)
    max_fail = max(int(max_consecutive_failures), 1)
    workers = max(int(workers), 1)
    processed = done = failed = 0
    consecutive = 0
    fused = stopped = False
    results: list[dict] = []
    saved_n = 0  # 已回调给 save 的结果数（增量切片，避免全量重放）
    open_imgs: list = []  # 已打开未 flush 的图片对象；熔断/停机时兜底释放

    def _safe(it: dict) -> dict:
        try:
            return process_one(it, encoder, collection)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001 — 单图异常按失败计数，不中断
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def flush(imgs: list, metas: list[dict], idx_range: list[int]) -> bool:
        """整批 encode+upsert；返回是否成功。图片对象无论如何都释放。"""
        try:
            embs = encoder.encode_images(imgs)  # type: ignore[union-attr]
            emb_list = embs.tolist() if hasattr(embs, "tolist") else embs
            ids = [image_id(m["image_key"], m["video_id"]) for m in metas]
            collection.upsert(  # type: ignore[union-attr]
                ids=ids, embeddings=emb_list, metadatas=metas,
                documents=[document_text(m) for m in metas])
            return True
        except Exception as e:  # noqa: BLE001 — 整批失败，按失败计数后继续
            print(f"[error] encode/upsert 整批失败（{len(imgs)} 张）: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            return False
        finally:
            for im in imgs:
                try:
                    im.close()
                except Exception:  # noqa: BLE001 — 释放失败不影响主流程
                    pass

    for batch in iter_batches(items, batch_size):
        if disk_check is not None:
            try:
                disk_check()
            except LowDiskSpace as e:
                for im in open_imgs:
                    try:
                        im.close()
                    except Exception:  # noqa: BLE001 — 释放失败不影响主流程
                        pass
                print(f"[DISK] {e}；已停止处理（processed={processed} "
                      f"done={done} failed={failed}）。state 保持上次写盘状态，"
                      "清理磁盘后重跑相同命令即可续跑。", file=sys.stderr, flush=True)
                stopped = True
                break
        # 批内并行准备（PIL 解码+缩放，线程池重叠 IO/CPU）；保序
        if workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                prepared = list(pool.map(_safe, batch))
        else:
            prepared = [_safe(it) for it in batch]

        batch_imgs: list = []
        batch_metas: list[dict] = []
        batch_ok_idx: list[int] = []
        fused_this_batch = False
        for offset, res in enumerate(prepared):
            item = batch[offset]
            key = item.get("image_key", "")
            processed += 1
            if not res.get("ok"):
                failed += 1
                consecutive += 1
                results.append({"image_key": key, "ok": False,
                                "error": res.get("error", "unknown")})
                if consecutive >= max_fail:
                    print(f"[FUSE] 连续 {consecutive} 张图片处理失败，"
                          "疑似 NAS 掉线或后端存储不可用，已中止处理。",
                          file=sys.stderr, flush=True)
                    print(f"[FUSE] state 保持已写盘内容（done={done} failed={failed}），"
                          "确认挂载恢复后直接重跑本命令即可续跑。",
                          file=sys.stderr, flush=True)
                    fused = True
                    fused_this_batch = True
                    break
                continue
            consecutive = 0
            img = res.get("img")
            meta = res.get("meta")
            if img is not None and meta is not None:
                batch_imgs.append(img)
                open_imgs.append(img)
                batch_metas.append(meta)
                batch_ok_idx.append(len(results))
                results.append({"image_key": key, "ok": True,
                                "state": res.get("state")})
            else:
                # 无 img 的成功项：已完成（不需要批 encode）
                done += 1
                results.append({"image_key": key, "ok": True,
                                "state": res.get("state")})
        # 批尾 flush（不足整批或正好满批）
        if batch_imgs:
            if flush(batch_imgs, batch_metas, batch_ok_idx):
                for i in batch_ok_idx:
                    r = results[i]
                    r["state"] = r.get("state") or {"status": "done"}
                done += len(batch_ok_idx)
            else:
                for i in batch_ok_idx:
                    results[i] = {"image_key": results[i]["image_key"], "ok": False,
                                  "error": "batch: encode/upsert 失败"}
                failed += len(batch_ok_idx)
                consecutive += len(batch_ok_idx)
                if consecutive >= max_fail:
                    fused = True
                    fused_this_batch = True
        open_imgs = [im for im in open_imgs if im not in batch_imgs]
        if save is not None:
            try:
                save(results[saved_n:])
                saved_n = len(results)
            except Exception as e:  # noqa: BLE001 — 断点写盘失败不中断处理
                print(f"[warn] state 写盘失败: {e}", file=sys.stderr)
        if fused_this_batch:
            break

    return {"processed": processed, "done": done, "failed": failed,
            "fused": fused, "stopped": stopped, "results": results}


# ---------- 状态 ----------

def _state_file(p: Path) -> Path:
    """state 路径归一：目录 -> 其下 state_images.json；带 .json 后缀 -> 文件本身。"""
    return p if p.suffix == ".json" else p / "state_images.json"


def load_state(state_path: Path) -> dict:
    """读 state 文件；路径传目录则取其下 state_images.json。
    损坏时改名保留并从空开始（与 06 同款）。"""
    p = _state_file(state_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] {p.name} 损坏，从空开始（旧文件改名保留）: {e}",
                  file=sys.stderr)
            p.rename(p.with_suffix(".json.corrupt"))
    return {}


def save_state(state_path: Path, state: dict) -> None:
    """原子写盘：tmp + rename（与 06 同款）；路径传目录则写其下 state_images.json。"""
    p = _state_file(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.rename(p)


def plan_pending(items: list[dict], state: dict, force: bool = False) -> list[dict]:
    """按 state 过滤出待处理项：done 且 mtime/size 均未变化才跳过。

    failed 不阻断（可重试）；mtime 或 size 任一变化即重处理（幂等增量）；
    force 时全部重跑。
    """
    todo: list[dict] = []
    for it in items:
        key = it.get("image_key") or image_key_of(it["share"], it["relpath"])
        it.setdefault("image_key", key)
        st = state.get(key)
        if st and not force and st.get("status") == "done" \
                and st.get("mtime") == it["mtime"] and st.get("size") == it["size"]:
            continue
        todo.append(it)
    return todo


# ---------- 磁盘保护 ----------

def check_disk_space(target_dir: Path, min_free_gb: float = MIN_FREE_GB) -> None:
    """目标目录所在盘可用空间低于阈值即抛 LowDiskSpace（写前检查）。"""
    usage = shutil.disk_usage(target_dir)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise LowDiskSpace(free_gb, min_free_gb)


# ---------- 主流程 ----------

def fmt_gb(n: int) -> str:
    return f"{n / (1024 ** 3):.1f}GB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NAS 图片批量索引（发现->CN-CLIP->Chroma，与视频帧同库）")
    ap.add_argument("--roots", nargs="+", default=_default_roots())
    ap.add_argument("--include-dir", action="append", default=[],
                    help="只收录 relpath 含该子串的图片（可重复）")
    ap.add_argument("--exclude-dir", action="append", default=[],
                    help="排除 relpath 含该子串的图片（可重复）")
    ap.add_argument("--db", default=DEFAULT_DB,
                    help="Chroma PersistentClient 路径（默认 05_api 首选 db）")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help="断点 state 文件路径（默认与视频 state 分离）")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--model", default="cnclip",
                    choices=["siglip2", "cnclip", "openclip", "dummy"])
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个（体积升序，小样验证用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    ap.add_argument("--force", action="store_true", help="忽略断点状态，全部重跑")
    ap.add_argument("--workers", type=int, default=2,
                    help="批内并行 PIL 解码线程数（图片解码较轻，默认 2）")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="每批 encode+upsert 图片数")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                    help="编码前最长边缩放阈值（0 关闭缩放），默认 1024")
    ap.add_argument("--min-size-mb", type=float, default=0.0,
                    help="小于该体积的文件跳过（0 字节文件始终跳过）")
    ap.add_argument("--max-consecutive-failures", type=int,
                    default=MAX_CONSECUTIVE_FAILED,
                    help="连续失败达到该数即熔断停机（疑似 NAS 掉线），默认 20")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                    help="写库前可用磁盘下限（GB），低于即停机，默认 5GB")
    args = ap.parse_args()
    args.batch_size = max(int(args.batch_size), 1)
    args.max_consecutive_failures = max(int(args.max_consecutive_failures), 1)
    args.min_free_gb = max(float(args.min_free_gb), 0.0)

    # 启动前置检查（dry-run 同样执行：它也要遍历目录，根不可用时直接退出）
    root_errs = check_roots_readable(args.roots)
    if root_errs:
        for e in root_errs:
            print(e, file=sys.stderr)
        return 1

    t_start = time.time()
    images = discover_images(args.roots, args.include_dir, args.exclude_dir,
                             args.min_size_mb)
    for it in images:
        it["image_key"] = image_key_of(it["share"], it["relpath"])
    if args.limit and args.limit > 0:
        images = images[:args.limit]

    total_bytes = sum(it["size"] for it in images)
    state_path = Path(args.state)
    state = load_state(state_path)
    todo = plan_pending(images, state, force=args.force)

    if args.dry_run:
        from collections import Counter
        bydir: Counter[str] = Counter()
        for it in images:
            top = it["relpath"].split("/")[0] if "/" in it["relpath"] else "."
            bydir[f"{it['share']}/{top}"] += 1
        print(f"[dry-run] 图片 {len(images)} 张，总体积 {fmt_gb(total_bytes)} "
              f"({total_bytes} 字节)，平均 "
              f"{total_bytes / max(len(images), 1) / 1024 / 1024:.2f}MB/张")
        print(f"[dry-run] state 已索引 {len(images) - len(todo)} 张，"
              f"本次待处理 {len(todo)} 张")
        print("[dry-run] 按顶层目录数量 top10:")
        for d, n in bydir.most_common(10):
            print(f"    {n:>7}张  {d}")
        if images:
            print("[dry-run] 体积最小 3 张（--limit 3 将选中它们）:")
            for it in images[:3]:
                print(f"    {it['size'] / 1024:8.1f}KB  {it['share']}/{it['relpath']}")
        return 0

    if not images:
        print("未发现图片，先检查 --roots / --include-dir。")
        return 1

    print(f"[plan] 发现 {len(images)} 张图片（{fmt_gb(total_bytes)}），"
          f"待处理 {len(todo)}，跳过 {len(images) - len(todo)}")
    if not todo:
        print("[done] 无待处理图片，全部已索引。")
        return 0

    # 主进程：torch 限线程 + 一次性加载 encoder（复用 02_embed_index，不新写）
    import torch
    torch.set_num_threads(6)
    spec = importlib.util.spec_from_file_location(
        "mod_02_embed_index", str(SCRIPTS / "02_embed_index.py"))
    assert spec is not None and spec.loader is not None
    m02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m02)  # type: ignore[union-attr]
    encoder = m02.build_encoder(args.model)

    import chromadb
    db_path = Path(args.db)
    client = chromadb.PersistentClient(path=str(db_path))
    col = client.get_or_create_collection(
        name=args.collection, metadata={"hnsw:space": "cosine"})

    by_key = {it["image_key"]: it for it in todo}

    def apply_results(results: list[dict]) -> None:
        """把批结果增量折算进 state 并原子写盘（断点安全：中断只丢当前批）。"""
        for r in results:
            key = r["image_key"]
            item = by_key.get(key)
            if item is None:
                continue
            if r.get("ok"):
                st = r.get("state") or {"status": "done"}
                st.setdefault("mtime", item["mtime"])
                st.setdefault("size", item["size"])
                st.setdefault("frame_path", item["image_path"])
                state[key] = st
            else:
                state[key] = {"status": "failed", "mtime": item["mtime"],
                              "size": item["size"], "error": r.get("error", "unknown")}
        save_state(state_path, state)

    def prepare(item: dict, enc, col_):
        """批内单项准备：打开+缩放+meta，返回 img 交给 run_processing 攒批。"""
        img, w, h = open_image(item["image_path"], args.max_side)
        meta = build_meta(item, w, h)
        return {"ok": True, "img": img, "meta": meta,
                "state": {"status": "done", "mtime": item["mtime"],
                          "size": item["size"], "frame_path": item["image_path"]}}

    def disk_check() -> None:
        check_disk_space(db_path, args.min_free_gb)

    plan = run_processing(
        todo, encoder=encoder, collection=col, process_one=prepare,
        batch_size=args.batch_size,
        max_consecutive_failures=args.max_consecutive_failures,
        save=apply_results, disk_check=disk_check, workers=args.workers)

    # 兜底：批间回调可能因熔断/停机未覆盖最后一批，统一重算落盘（幂等）
    apply_results(plan["results"])

    # encoder.json 写进 db 目录（与 02/03/API 的 load_encoder 约定一致）
    try:
        db_path.mkdir(parents=True, exist_ok=True)
        (db_path / "encoder.json").write_text(
            json.dumps({"encoder": encoder.name, "model_id": encoder.model_id,
                        "dim": int(encoder.dim)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — encoder.json 写失败不阻断索引
        print(f"[warn] encoder.json 写入失败: {e}", file=sys.stderr)

    el = time.time() - t_start
    print(f"[summary] done={plan['done']} failed={plan['failed']} "
          f"skipped={len(images) - len(todo)} 耗时={el / 60:.1f}分 "
          f"平均={plan['processed'] / max(el, 0.001):.1f}张/秒")
    print(f"[summary] chroma->{db_path} collection={args.collection} "
          f"state->{_state_file(state_path)}")
    if plan["fused"] or plan["stopped"]:
        return 3
    return 0 if plan["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
