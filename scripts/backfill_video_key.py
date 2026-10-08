#!/usr/bin/env python3
"""给现有 Chroma 集合回填 video_key 元数据字段（不重抽帧、不重编码）。

背景：README 的全局唯一命名 video_key=md5(share/relpath)[:8] 只落在帧文件名和
Chroma id 里，元数据没存，导致检索层退回用非唯一的文件名 stem 聚合，
同名不同目录的视频被合并。此脚本把 video_key 补进元数据。

用法:
    .venv/bin/python scripts/backfill_video_key.py --dry-run
    .venv/bin/python scripts/backfill_video_key.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/local-runtime/index_full/cnclip"
COLLECTION = "frames"
BATCH = 500
KEY_RE = re.compile(r"^[0-9a-f]{8}$")


def derive_key(meta: dict) -> str | None:
    """优先用 frame_path 文件名前缀（已验证 100% 可靠），否则用 share/relpath 重算。"""
    fp = os.path.basename(str(meta.get("frame_path") or ""))
    prefix = fp.split("_", 1)[0]
    if KEY_RE.fullmatch(prefix):
        return prefix
    share = str(meta.get("share") or "")
    relpath = str(meta.get("relpath") or "")
    if share and relpath:
        return hashlib.md5(f"{share}/{relpath}".encode("utf-8")).hexdigest()[:8]
    vkey = str(meta.get("video_key") or "")
    if KEY_RE.fullmatch(vkey):
        return vkey
    return None


def load_state_frames(state_file: Path, frames_dir: Path) -> dict[tuple[str, float], str]:
    """(video_key, time) -> 帧文件绝对-ish 路径，用于修复缺 frame_path 的行。"""
    import json
    if not state_file.is_file():
        return {}
    state = json.loads(state_file.read_text(encoding="utf-8"))
    out: dict[tuple[str, float], str] = {}
    for vkey, info in state.items():
        names = info.get("frames") or []
        if not isinstance(names, list):
            continue
        for name in names:
            # 帧名 {video_key}_{stem}_{t}.jpg，取最后一个 _ 后的数字为时间
            m = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.jpg$", name)
            if m:
                out[(vkey, round(float(m.group(1)), 1))] = str(frames_dir / name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--state", default=str(ROOT / "data/local-runtime/index_full/state.json"))
    ap.add_argument("--frames-dir", default=str(ROOT / "data/local-runtime/frames_full"))
    args = ap.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=args.db)
    col = client.get_collection(COLLECTION)
    total = col.count()
    print(f"集合 {COLLECTION} 共 {total} 条，dry_run={args.dry_run}")

    # 修复 retry 脚本留下缺 frame_path 的行：按 (video_key, time) 从 state 反查帧名
    frame_lookup = load_state_frames(Path(args.state), Path(args.frames_dir))
    print(f"state 帧索引: {len(frame_lookup)} 条可用于补 frame_path")

    offset = 0
    updated = 0
    skipped_has_key = 0
    fixed_path = 0
    undecidable: list[str] = []
    key_set: set[str] = set()

    while offset < total:
        payload = col.get(include=["metadatas"], limit=args.batch, offset=offset)
        ids = payload.get("ids") or []
        metas = payload.get("metadatas") or []
        if not ids:
            break
        upd_ids: list[str] = []
        upd_metas: list[dict] = []
        for i, meta in zip(ids, metas):
            meta = dict(meta or {})
            changed = False
            key = str(meta.get("video_key") or "")
            if not KEY_RE.fullmatch(key):
                dk = derive_key(meta)
                if not dk:
                    undecidable.append(i)
                    continue
                meta["video_key"] = dk
                key = dk
                changed = True
            key_set.add(key)
            # 补缺失的 frame_path（retry 脚本遗留）
            if not meta.get("frame_path"):
                t = round(float(meta.get("time") or 0.0), 1)
                fp = frame_lookup.get((key, t))
                if fp:
                    meta["frame_path"] = fp
                    fixed_path += 1
                    changed = True
            if not changed:
                skipped_has_key += 1
                continue
            upd_ids.append(i)
            upd_metas.append(meta)
        if upd_ids and not args.dry_run:
            col.update(ids=upd_ids, metadatas=upd_metas)
        updated += len(upd_ids)
        offset += len(ids)
        if offset % 20000 == 0 or offset >= total:
            print(f"  进度 {offset}/{total}  待更新 {updated}  已完整 {skipped_has_key}")

    print()
    print(f"需更新条数      : {updated}")
    print(f"已完整跳过      : {skipped_has_key}")
    print(f"其中补 frame_path: {fixed_path}")
    print(f"无法判定        : {len(undecidable)}  {undecidable[:5]}")
    print(f"涉及 video_key 数: {len(key_set)}")
    if args.dry_run:
        print("\n[dry-run] 未写入。去掉 --dry-run 实际执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
