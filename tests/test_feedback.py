"""意见收集（POST /api/feedback）行为测试。

只使用 tmp_path 落盘，不读取 NAS、素材原文件或真实向量库。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load_api_module():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    path = SCRIPTS / "05_api.py"
    spec = importlib.util.spec_from_file_location("api_feedback_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    api = _load_api_module()
    feedback_file = tmp_path / "feedback" / "feedback.jsonl"
    monkeypatch.setattr(api, "FEEDBACK_PATH", feedback_file)
    return api, TestClient(api.app), feedback_file


def _read_lines(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_feedback_appends_jsonl_with_timestamp(api_client):
    _api, client, feedback_file = api_client
    resp = client.post(
        "/api/feedback",
        json={
            "query": "示例查询词",
            "result_id": "abcd1234",
            "reason": "wrong_result",
            "detail": "第一条不是我想要的",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"code": 0, "message": "ok"}

    records = _read_lines(feedback_file)
    assert len(records) == 1
    rec = records[0]
    assert rec["query"] == "示例查询词"
    assert rec["result_id"] == "abcd1234"
    assert rec["reason"] == "wrong_result"
    assert rec["detail"] == "第一条不是我想要的"
    # 北京时间戳（显式时区，避免服务器 UTC 歧义）
    assert rec["ts"].endswith("+08:00")


def test_feedback_truncates_overlong_fields(api_client):
    _api, client, feedback_file = api_client
    resp = client.post(
        "/api/feedback",
        json={
            "query": "q" * 5000,
            "result_id": "r" * 5000,
            "reason": "missing_asset",
            "detail": "d" * 5000,
        },
    )
    assert resp.status_code == 200
    rec = _read_lines(feedback_file)[0]
    assert len(rec["query"]) == 200
    assert len(rec["result_id"]) == 200
    assert len(rec["detail"]) == 2000


def test_feedback_normalizes_reason_and_missing_fields(api_client):
    _api, client, feedback_file = api_client
    # 非法 reason 归为 other；大小写容错；缺字段全用默认值（不报错）
    client.post("/api/feedback", json={"query": "x", "reason": "hack"})
    client.post("/api/feedback", json={"query": "y", "reason": "WRONG_RESULT"})
    client.post("/api/feedback", json={})
    records = _read_lines(feedback_file)
    assert [r["reason"] for r in records] == ["other", "wrong_result", "other"]
    assert records[2]["query"] == ""
    assert records[2]["detail"] == ""


def test_feedback_appends_multiple_submissions(api_client):
    _api, client, feedback_file = api_client
    for i in range(3):
        resp = client.post("/api/feedback", json={"query": f"q{i}", "reason": "other"})
        assert resp.status_code == 200
    records = _read_lines(feedback_file)
    assert [r["query"] for r in records] == ["q0", "q1", "q2"]


def test_feedback_write_failure_returns_500(api_client, monkeypatch):
    api, client, _feedback_file = api_client

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api, "write_feedback", _boom)
    resp = client.post("/api/feedback", json={"query": "x", "reason": "other"})
    assert resp.status_code == 500
    assert "反馈落盘失败" in resp.json()["detail"]

# ---------- 中文 reason 兼容映射（前端契约用中文枚举） ----------

def test_normalize_maps_chinese_reason():
    """前端提交中文 reason（结果不准/缺少素材/功能建议）映射为英文枚举。"""
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "api_05_fb", repo / "scripts" / "05_api.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    body = m.FeedbackRequest(query="跑步", reason="结果不准",
                             result_id="k1", comment="画面不符")
    rec = m.normalize_feedback(body)
    assert rec["reason"] == "wrong_result"
    assert rec["detail"] == "画面不符"   # comment 别名回退到 detail

    body2 = m.FeedbackRequest(query="跑步", reason="缺少素材",
                              detail="找不到肯尼亚那批")
    assert m.normalize_feedback(body2)["reason"] == "missing_asset"
    assert m.normalize_feedback(body2)["detail"] == "找不到肯尼亚那批"

    body3 = m.FeedbackRequest(query="跑步", reason="功能建议")
    assert m.normalize_feedback(body3)["reason"] == "other"


def test_normalize_english_reason_still_works():
    """英文枚举与非法值行为不变（向后兼容）。"""
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "api_05_fb2", repo / "scripts" / "05_api.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.normalize_feedback(
        m.FeedbackRequest(reason="wrong_result"))["reason"] == "wrong_result"
    assert m.normalize_feedback(
        m.FeedbackRequest(reason="乱写的"))["reason"] == "other"
