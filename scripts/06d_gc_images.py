#!/usr/bin/env python3
"""06d_gc_images.py — 图片删除下架与 30 天缓冲回收。

背景（ticket-04-part2）：NAS 上图片被删后，索引里的对应行会变成「搜得到但打不开」。
直接物理删除风险大（误删/临时移走都要全量重建），所以：

  缺失发现 ──> deleted_pending（结果里不可见，数据仍在，可随时恢复）
       │
       └─ 连续 30 天不在 ──> 真删：Chroma 行 + 缩略图 + state 条目

规则：
- deleted_at 只在「首次发现缺失」时打时间戳；持续缺失不刷新（30 天从首次缺失起算）。
- 缓冲期内文件重新出现（误删恢复、临时移走）：清掉标记恢复 done。
- 真删只针对 media_type=image 的行，绝不触碰视频帧集合。

用法：
    .venv/bin/python scripts/06d_gc_images.py --dry-run   # 只看不动
    .venv/bin/python scripts/06d_gc_images.py             # 真跑
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

DEFAULT_STATE = "data/local-runtime/index_full/state_images.json"
DEFAULT_DB = "data/local-runtime/index_full/cnclip"
DEFAULT_THUMBS = "data/local-runtime/images_thumbs"
COLLECTION = "frames"          # 图片与视频帧同集合，靠 media_type 区分
PURGE_AFTER_DAYS = 30
PURGE_AFTER_SEC = PURGE_AFTER_DAYS * 86400


# ---------- 纯函数（测试直接覆盖） ----------

def mark_deleted(entry: dict, now: float) -> dict:
    """首次发现缺失：打 deleted_at、状态改 deleted_pending。

    deleted_at 已存在则不刷新（二次扫描保持首次时间）。
    """
    out = dict(entry)
    out["status"] = "deleted_pending"
    out.setdefault("deleted_at", now)
    return out


def mark_restored(entry: dict) -> dict:
    """文件重新出现：清掉删除标记，恢复 done。从未标记过则原样返回。"""
    out = dict(entry)
    if out.get("status") != "deleted_pending" and "deleted_at" not in out:
        return out
    out.pop("deleted_at", None)
    out["status"] = "done"
    return out


def should_purge(entry: dict, now: float) -> bool:
    """缺失满 30 天才真删；done 条目（文件在）永不删。"""
    if entry.get("status") != "deleted_pending":
        return False
    deleted_at = entry.get("deleted_at")
    if not isinstance(deleted_at, (int, float)):
        return False
    return now - deleted_at >= PURGE_AFTER_SEC


def _state_file(p: Path) -> Path:
    """state 路径归一：目录 -> 其下 state_images.json；带 .json 后缀 -> 文件本身。"""
    return p if p.suffix == ".json" else p / "state_images.json"


def _load_state(p: Path) -> dict:
    f = _state_file(p)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — state 损坏时不让 GC 崩，按空处理并告警
        print(f"[warn] state 读取失败，按空处理: {e}", file=sys.stderr)
        return {}


def _save_state(p: Path, state: dict) -> None:
    f = _state_file(p)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.rename(f)  # 原子替换，中断不损坏


def image_chroma_id(image_key: str, stem: str) -> str:
    """与 06b 的 id 格式一致：{image_key}_{stem}@0.0。"""
    return f"{image_key}_{stem}@0.0"


# ---------- 对账主流程 ----------

def reconcile_deletions(state_path: Path, collection, existing_paths: set[str],
                        now: float | None = None,
                        thumbs_dir: Path | None = None) -> dict:
    """对比 state 与现存文件，执行 标记 -> (30 天) -> 真删。

    existing_paths: NAS 上现存的图片绝对路径集合（由调用方扫描提供）。
    返回 {"marked": 新标记数, "purged": 真删数, "alive": 存活数}。
    """
    now = time.time() if now is None else now
    state = _load_state(state_path)
    stats = {"marked": 0, "purged": 0, "alive": 0}
    to_delete_ids: list[str] = []
    to_delete_keys: list[str] = []

    for key, entry in state.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") == "failed":
            continue  # 失败条目不参与 GC（重试机制负责）
        frame_path = entry.get("frame_path") or entry.get("image_path") or ""

        if frame_path and frame_path in existing_paths:
            # 文件还在（含 deleted_pending 恢复场景）
            if entry.get("status") == "deleted_pending":
                state[key] = mark_restored(entry)
            stats["alive"] += 1
            continue

        # 文件缺失
        if entry.get("status") == "deleted_pending":
            if should_purge(entry, now):
                to_delete_keys.append(key)
                stem = Path(frame_path).stem if frame_path else key
                to_delete_ids.append(image_chroma_id(key, stem))
                stats["purged"] += 1
            # 缓冲期内：保持原样（deleted_at 不刷新）
        else:
            state[key] = mark_deleted(entry, now)
            stats["marked"] += 1

    # 真删：Chroma 行 + 缩略图 + state 条目
    if to_delete_ids:
        try:
            collection.delete(ids=to_delete_ids)
        except Exception as e:  # noqa: BLE001 — delete 失败保留 state，下次重试
            print(f"[warn] Chroma delete 失败（保留 state 下次重试）: {e}",
                  file=sys.stderr)
            stats["purged"] = 0
            _save_state(state_path, state)
            return stats
        for key in to_delete_keys:
            state.pop(key, None)
        if thumbs_dir:
            for key in to_delete_keys:
                tp = Path(thumbs_dir) / f"{key}.webp"
                try:
                    tp.unlink(missing_ok=True)
                except OSError as e:
                    print(f"[warn] 缩略图删除失败 {tp}: {e}", file=sys.stderr)

    _save_state(state_path, state)
    return stats


# ---------- NAS 现存文件扫描 ----------

def _load_06b():
    spec = importlib.util.spec_from_file_location(
        "gc_images_06b", str(SCRIPTS / "06b_index_images.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def scan_existing(roots: list[str], include_dir: list[str]) -> set[str]:
    """复用 06b 的 discover_images 拿现存图片路径集合（含 include 过滤）。"""
    m = _load_06b()
    items = m.discover_images(roots, include_dir, [], 0.0)
    return {it["image_path"] for it in items}


def main() -> int:
    ap = argparse.ArgumentParser(description="图片删除下架与 30 天缓冲回收")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help="state_images.json 路径（或其目录）")
    ap.add_argument("--db", default=DEFAULT_DB, help="Chroma 库目录")
    ap.add_argument("--thumbs-dir", default=DEFAULT_THUMBS,
                    help="缩略图目录（真删时同步清理）")
    ap.add_argument("--include-dir", action="append", default=[],
                    help="只 GC 该子串的 relpath（默认与索引范围一致：摄影图目录）")
    ap.add_argument("--roots", nargs="*", default=None,
                    help="NAS 根目录（默认取 site.json 的 shares 挂载点）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    args = ap.parse_args()

    # 默认与 06b 的夜间索引范围一致：摄影图目录（两共享）
    include = args.include_dir or ["摄影图"]
    if args.roots:
        roots = args.roots
    else:
        m = _load_06b()
        roots = m.default_roots()
    if not roots:
        print("[error] 未配置 NAS roots（site.json shares 为空）", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"[gc] 扫描现存图片: roots={roots} include={include}", flush=True)
    existing = scan_existing(roots, include)
    print(f"[gc] NAS 现存 {len(existing)} 张", flush=True)

    state_path = Path(args.state)
    state = _load_state(state_path)
    if args.dry_run:
        # 干跑：只报告会发生什么
        now = time.time()
        would_mark = would_purge = 0
        for key, entry in state.items():
            if not isinstance(entry, dict) or entry.get("status") == "failed":
                continue
            fp = entry.get("frame_path") or entry.get("image_path") or ""
            if fp and fp in existing:
                continue
            if entry.get("status") == "deleted_pending":
                would_purge += 1 if should_purge(entry, now) else 0
            else:
                would_mark += 1
        print(f"[dry-run] 将新标记 {would_mark} 张，"
              f"已满 30 天待真删 {would_purge} 张（未写盘）")
        return 0

    import chromadb
    client = chromadb.PersistentClient(path=args.db)
    col = client.get_collection(name=COLLECTION)

    stats = reconcile_deletions(
        state_path=state_path, collection=col, existing_paths=existing,
        thumbs_dir=Path(args.thumbs_dir) if args.thumbs_dir else None)
    el = time.time() - t0
    print(f"[summary] 新标记下架={stats['marked']} 真删={stats['purged']} "
          f"存活={stats['alive']} 耗时={el:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
