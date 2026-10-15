#!/usr/bin/env python3
"""意见收集查看器 — 读取 feedback.jsonl 并按类型/日期汇总。

用法:
    python3 scripts/show_feedback.py                  # 汇总统计
    python3 scripts/show_feedback.py --list           # 逐条列出（最近 50 条）
    python3 scripts/show_feedback.py --list --limit 200
    python3 scripts/show_feedback.py --reason wrong_result   # 只看某类型

数据文件路径: 默认 data/feedback/feedback.jsonl，可用 VF_FEEDBACK_PATH 覆盖。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = Path(os.environ.get("VF_FEEDBACK_PATH") or (ROOT / "data" / "feedback" / "feedback.jsonl"))

REASON_LABELS = {
    "wrong_result": "结果不准",
    "missing_asset": "缺素材",
    "other": "其他",
}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 跳过坏行，不因一行损坏阻塞查看
    return records


def summarize(records: list[dict]) -> None:
    print(f"共 {len(records)} 条反馈")
    if not records:
        return
    by_reason = Counter(r.get("reason", "other") for r in records)
    print("\n按类型:")
    for reason, count in by_reason.most_common():
        label = REASON_LABELS.get(reason, reason)
        print(f"  {label:<8} {count}")
    by_date = Counter((r.get("ts") or "?")[:10] for r in records)
    print("\n按日期:")
    for date, count in sorted(by_date.items()):
        print(f"  {date}  {count}")


def list_records(records: list[dict], limit: int, reason: str | None) -> None:
    filtered = [r for r in records if not reason or r.get("reason") == reason]
    for rec in filtered[-limit:]:
        label = REASON_LABELS.get(rec.get("reason", ""), rec.get("reason", ""))
        print(f"[{rec.get('ts', '?')}] {label} | 搜索词: {rec.get('query', '')!r}")
        if rec.get("result_id"):
            print(f"    结果: {rec['result_id']}")
        if rec.get("detail"):
            print(f"    描述: {rec['detail']}")
    print(f"\n（显示 {min(limit, len(filtered))} / {len(filtered)} 条）")


def main() -> int:
    ap = argparse.ArgumentParser(description="意见收集查看器")
    ap.add_argument("--path", default=str(DEFAULT_PATH), help="feedback.jsonl 路径")
    ap.add_argument("--list", action="store_true", help="逐条列出")
    ap.add_argument("--limit", type=int, default=50, help="列出条数上限（默认 50）")
    ap.add_argument("--reason", default=None,
                    help="只看某类型: wrong_result / missing_asset / other")
    args = ap.parse_args()

    path = Path(args.path)
    records = load_records(path)
    print(f"数据文件: {path}")
    print("=" * 50)
    if args.list:
        list_records(records, args.limit, args.reason)
    else:
        summarize(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())