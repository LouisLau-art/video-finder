"""真实专有名称 HTTP 验收脚本的本地假服务测试。"""
from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/08_real_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("real_acceptance_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        query = request["query"]
        if query == "placeholder_query_semantic":
            results = [{
                "video_id": "fixture_target_semantic",
                "rank": 1,
                "match_type": "semantic",
                "matched_text": "",
            }]
        elif query == "placeholder_query_invalid":
            results = [{
                "video_id": "fixture_target_invalid",
                "rank": 1,
            }]
        elif query == "placeholder_query_miss":
            results = [{
                "video_id": "fixture_other_material",
                "rank": 1,
                "match_type": "semantic",
                "matched_text": "",
            }]
        else:
            results = [{
                "video_id": "fixture_target_keyword",
                "rank": 2,
                "match_type": "keyword",
                "matched_text": "placeholder_matched_fragment",
            }]
        body = json.dumps({
            "code": 0,
            "message": "success",
            "data": {"results": results},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_service():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_acceptance_counts_hit_rank_source_miss_and_unknown(fake_service):
    module = _load_module()
    cases = [
        {
            "id": "neutral_hit",
            "query": "placeholder_query_hit",
            "target_video_id": "fixture_target_keyword",
            "notes": "local note",
        },
        {
            "id": "neutral_miss",
            "query": "placeholder_query_miss",
            "target_video_id": "fixture_missing_target",
            "notes": "local note",
        },
        {
            "id": "neutral_semantic_only",
            "query": "placeholder_query_semantic",
            "target_video_id": "fixture_target_semantic",
            "notes": "local note",
        },
        {
            "id": "neutral_invalid_contract",
            "query": "placeholder_query_invalid",
            "target_video_id": "fixture_target_invalid",
            "notes": "local note",
        },
    ]

    report = module.run_acceptance(
        cases,
        base_url=fake_service,
        top_k=3,
        timeout=2.0,
    )
    by_id = {row["id"]: row for row in report["cases"]}

    assert by_id["neutral_hit"]["hit"] is True
    assert by_id["neutral_hit"]["target_rank"] == 2
    assert by_id["neutral_hit"]["match_type"] == "keyword"
    assert by_id["neutral_hit"]["keyword_recalled"] is True
    assert by_id["neutral_miss"]["hit"] is False
    assert by_id["neutral_semantic_only"]["hit"] is True
    assert by_id["neutral_semantic_only"]["keyword_recalled"] is False
    assert by_id["neutral_invalid_contract"]["status"] == "unknown"
    assert report["status"] == "unknown"
    assert report["exit_code"] == module.EXIT_UNKNOWN
    assert report["passed_count"] == 1
    assert report["failed_count"] == 2
    assert report["unknown_count"] == 1


def test_report_contains_no_query_target_or_matched_text(fake_service):
    module = _load_module()
    query = "placeholder_query_report_secret"
    target = "fixture_target_report_secret"
    matched = "placeholder_matched_secret"
    report = module.run_acceptance(
        [{"id": "neutral_report", "query": query, "target_video_id": target, "notes": ""}],
        base_url=fake_service,
        top_k=3,
        timeout=2.0,
    )
    serialized = json.dumps(report, ensure_ascii=False)
    assert query not in serialized
    assert target not in serialized
    assert matched not in serialized
    assert report["cases"][0]["query_sha256_prefix"]
    assert report["cases"][0]["target_sha256_prefix"]


def test_missing_cases_file_is_nonzero_and_not_silent():
    module = _load_module()
    missing = ROOT / "eval_local/test_missing_real_acceptance_cases.json"
    report = ROOT / "eval_local/test_missing_real_acceptance_report.json"
    missing.unlink(missing_ok=True)
    report.unlink(missing_ok=True)

    exit_code = module.main([
        "--cases", str(missing),
        "--report", str(report),
    ])

    assert exit_code == module.EXIT_SCRIPT_ERROR
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error"] == "cases_file_missing"
    report.unlink(missing_ok=True)
