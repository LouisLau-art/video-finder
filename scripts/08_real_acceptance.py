#!/usr/bin/env python3
"""真实专有名称的只读 HTTP 验收工具。

真实查询和目标标识只从 gitignored 的本地用例文件读取，不写入报告。
报告只保存用例 ID、哈希前缀、命中状态、名次和命中来源类型。
脚本不导入或读取 Chroma 集合，只消费 HTTP 响应，因此不需要元数据分页。

退出码与 07_hybrid_eval.py 保持一致：
0=全部通过，1=脚本/输入/HTTP 错误，2=验收不通过，3=结果无法判定。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "eval_local/real_acceptance_cases.json"
DEFAULT_REPORT = ROOT / "eval_local/real_acceptance_report.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
VALID_MATCH_TYPES = {"semantic", "keyword", "both"}

EXIT_OK = 0
EXIT_SCRIPT_ERROR = 1
EXIT_ACCEPTANCE_FAILED = 2
EXIT_UNKNOWN = 3


class InputError(Exception):
    """输入文件或配置错误，错误码本身不包含原始输入。"""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str):  # pragma: no cover - argparse 专用分支
        self.print_usage(sys.stderr)
        self.exit(EXIT_SCRIPT_ERROR, f"{self.prog}: error: {message}\n")


def _require_gitignored(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(path)],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise InputError("path_not_gitignored")


def _digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _read_cases(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise InputError("cases_file_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError("cases_file_invalid") from exc
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise InputError("cases_empty_or_invalid")
    cases: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise InputError("case_shape_invalid")
        case_id = str(raw.get("id") or "").strip()
        query = str(raw.get("query") or "").strip()
        target = str(raw.get("target_video_id") or "").strip()
        if not case_id or not query or not target:
            raise InputError("case_fields_missing")
        if case_id in seen_ids:
            raise InputError("case_id_duplicated")
        seen_ids.add(case_id)
        cases.append({
            "id": case_id,
            "query": query,
            "target_video_id": target,
            "notes": str(raw.get("notes") or ""),
        })
    return cases


def _post_search(
    base_url: str,
    query: str,
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        {"query": query, "top_k": int(top_k)},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        return {"status": "error", "error": f"http_status_{int(exc.code)}"}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"status": "error", "error": "network_error"}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return {"status": "error", "error": "response_json_invalid"}
    if not isinstance(payload, dict):
        return {"status": "error", "error": "response_shape_invalid"}
    return {"status": "ok", "payload": payload}


def _extract_result(
    payload: dict[str, Any],
    target_video_id: str,
) -> dict[str, Any]:
    data = payload.get("data")
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return {"status": "error", "error": "results_missing"}
    for index, row in enumerate(results, 1):
        if not isinstance(row, dict) or str(row.get("video_id")) != target_video_id:
            continue
        rank_value = row.get("rank", index)
        try:
            rank = int(rank_value)
        except (TypeError, ValueError):
            return {"status": "unknown", "error": "target_rank_invalid"}
        match_type = row.get("match_type")
        if match_type not in VALID_MATCH_TYPES:
            return {"status": "unknown", "error": "match_type_invalid"}
        matched_text = row.get("matched_text")
        if not isinstance(matched_text, str):
            return {"status": "unknown", "error": "matched_text_invalid"}
        if match_type in {"keyword", "both"} and not matched_text.strip():
            return {"status": "unknown", "error": "matched_text_empty_for_keyword"}
        if match_type == "semantic" and matched_text:
            return {"status": "unknown", "error": "matched_text_unexpected_for_semantic"}
        return {
            "status": "ok",
            "hit": True,
            "target_rank": rank,
            "match_type": match_type,
            "keyword_recalled": match_type in {"keyword", "both"},
            "matched_text_nonempty": bool(matched_text.strip()),
        }
    return {
        "status": "ok",
        "hit": False,
        "target_rank": None,
        "match_type": None,
        "keyword_recalled": False,
        "matched_text_nonempty": False,
    }


def run_acceptance(
    cases: list[dict[str, str]],
    base_url: str,
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    """按顺序执行用例；单条错误不会中断其它用例。"""
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "id": case["id"],
            "query_sha256_prefix": _digest(case["query"]),
            "target_sha256_prefix": _digest(case["target_video_id"]),
        }
        response = _post_search(base_url, case["query"], top_k, timeout)
        if response.get("status") != "ok":
            row.update({
                "status": "error",
                "error": response.get("error", "request_failed"),
                "hit": False,
                "target_rank": None,
                "match_type": None,
                "keyword_recalled": False,
                "matched_text_nonempty": False,
            })
        else:
            extracted = _extract_result(response["payload"], case["target_video_id"])
            row.update({
                "status": extracted["status"],
                "error": extracted.get("error"),
                "hit": bool(extracted.get("hit", False)),
                "target_rank": extracted.get("target_rank"),
                "match_type": extracted.get("match_type"),
                "keyword_recalled": bool(extracted.get("keyword_recalled", False)),
                "matched_text_nonempty": bool(
                    extracted.get("matched_text_nonempty", False)
                ),
            })
        rows.append(row)

    error_count = sum(row["status"] == "error" for row in rows)
    unknown_count = sum(row["status"] == "unknown" for row in rows)
    passed_count = sum(
        row["status"] == "ok" and row["hit"] and row["keyword_recalled"]
        for row in rows
    )
    failed_count = len(rows) - error_count - unknown_count - passed_count
    if error_count:
        status, exit_code = "error", EXIT_SCRIPT_ERROR
    elif unknown_count:
        status, exit_code = "unknown", EXIT_UNKNOWN
    elif failed_count:
        status, exit_code = "failed", EXIT_ACCEPTANCE_FAILED
    else:
        status, exit_code = "passed", EXIT_OK
    return {
        "version": 1,
        "contains_real_names": False,
        "status": status,
        "exit_code": exit_code,
        "top_k": int(top_k),
        "case_count": len(rows),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "unknown_count": unknown_count,
        "cases": rows,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    _require_gitignored(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="真实专有名称的只读 HTTP 验收")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k < 1 or args.top_k > 100 or args.timeout <= 0:
        print(json.dumps({"status": "error", "error": "invalid_arguments"}))
        return EXIT_SCRIPT_ERROR
    try:
        _require_gitignored(args.cases)
        _require_gitignored(args.report)
        cases = _read_cases(args.cases)
    except InputError as exc:
        report = {
            "version": 1,
            "contains_real_names": False,
            "status": "error",
            "exit_code": EXIT_SCRIPT_ERROR,
            "error": exc.code,
            "case_count": 0,
            "cases": [],
        }
        try:
            _write_report(args.report, report)
        except InputError:
            pass
        print(json.dumps({
            "status": "error",
            "error": exc.code,
            "exit_code": EXIT_SCRIPT_ERROR,
        }, ensure_ascii=False))
        return EXIT_SCRIPT_ERROR

    report = run_acceptance(
        cases,
        base_url=args.base_url,
        top_k=args.top_k,
        timeout=args.timeout,
    )
    _write_report(args.report, report)
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "passed_count": report["passed_count"],
        "failed_count": report["failed_count"],
        "error_count": report["error_count"],
        "unknown_count": report["unknown_count"],
        "exit_code": report["exit_code"],
    }, ensure_ascii=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
