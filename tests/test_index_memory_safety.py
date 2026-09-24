"""全量索引内存保护与断点状态的单元测试。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/06_index_all.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("index_memory_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_guard_raises_and_pending_state_is_resumable():
    module = _load_module()
    with pytest.raises(module.MemoryLimitReached) as exc:
        module.enforce_memory_limit(128.0, 64.0)
    assert exc.value.rss_mb == 128.0
    assert exc.value.limit_mb == 64.0

    video = {
        "video_key": "fixture_video",
        "mtime": 1.0,
        "size": 2.0,
    }
    rows = [{"frame_path": "/tmp/placeholder/frame.jpg"}]
    state = module.pending_upsert_state(video, rows)
    assert state["status"] == "done"
    assert state["pending_upsert"] is True
    assert state["frames"] == ["frame.jpg"]


def test_batches_never_exceed_configured_limit():
    module = _load_module()
    batches = list(module.iter_batches(list(range(10)), 3))
    assert [len(batch) for batch in batches] == [3, 3, 3, 1]


def test_memory_guard_allows_limit_boundary():
    module = _load_module()
    module.enforce_memory_limit(64.0, 64.0)
    module.enforce_memory_limit(128.0, 0)
