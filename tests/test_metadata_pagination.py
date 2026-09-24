"""元数据分页读取与关键词索引节流的内存测试。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "scripts/05_api.py"


def _load_api():
    spec = importlib.util.spec_from_file_location("pagination_api_test", str(API_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PagedCollection:
    def __init__(self, frames):
        self.frames = list(frames)
        self.get_calls = []

    def count(self):
        return len(self.frames)

    def get(self, include=None, limit=None, offset=0):
        assert include == ["metadatas"]
        self.get_calls.append((limit, offset))
        start = max(0, int(offset))
        end = len(self.frames) if limit is None else start + int(limit)
        return {"metadatas": [dict(item) for item in self.frames[start:end]]}


def _frame(video_id: str, filename: str):
    return {
        "video_id": video_id,
        "video_name": f"{filename}.mp4",
        "relpath": f"placeholder_collection/{filename}.mp4",
        "share": "placeholder_share",
        "time": 0.0,
    }


def test_paged_metadata_reads_all_rows_in_stable_order():
    api = _load_api()
    frames = [_frame(f"fixture_{i}", f"placeholder_{i}") for i in range(7)]
    col = _PagedCollection(frames)

    first = list(api.iter_collection_metadata(col, expected_count=7, batch_size=3))
    first_calls = list(col.get_calls)
    col.get_calls.clear()
    second = list(api.iter_collection_metadata(col, expected_count=7, batch_size=3))

    assert [item["video_id"] for item in first] == [item["video_id"] for item in second]
    assert len(first) == 7
    assert len({item["video_id"] for item in first}) == 7
    assert first_calls == [(3, 0), (3, 3), (1, 6)]
    assert col.get_calls == [(3, 0), (3, 3), (1, 6)]


def test_paged_and_single_read_build_equivalent_keyword_indexes():
    api = _load_api()
    frames = [
        _frame("fixture_a", "placeholder_alpha"),
        _frame("fixture_b", "placeholder_beta"),
        _frame("fixture_c", "placeholder_gamma"),
    ]
    col = _PagedCollection(frames)

    from_pages = api._build_keyword_index(
        api.iter_collection_metadata(col, expected_count=3, batch_size=2)
    )
    direct = api._build_keyword_index(frames)

    assert from_pages == direct


def test_keyword_index_rebuild_is_throttled_then_eventually_consistent(monkeypatch):
    api = _load_api()
    clock = [100.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(api, "KEYWORD_CHANNEL_ENABLED", True)
    monkeypatch.setattr(api, "KEYWORD_INDEX_REBUILD_MIN_INTERVAL", 5.0)
    monkeypatch.setattr(api, "METADATA_BATCH_SIZE", 1)
    monkeypatch.setattr(api, "_KEYWORD_INDEX", None)
    monkeypatch.setattr(api, "_KEYWORD_INDEX_COUNT", None)
    monkeypatch.setattr(api, "_KEYWORD_INDEX_LAST_REFRESH_AT", None)

    col = _PagedCollection([
        _frame("fixture_old", "placeholder_old"),
    ])
    first = api._ensure_keyword_index(col, 1)
    assert set(first) == {"fixture_old"}
    assert col.get_calls == [(1, 0)]

    col.frames.append(_frame("fixture_new", "placeholder_new"))
    second = api._ensure_keyword_index(col, 2)
    assert second is first
    assert set(second) == {"fixture_old"}
    assert col.get_calls == [(1, 0)]

    clock[0] += 5.0
    third = api._ensure_keyword_index(col, 2)
    assert set(third) == {"fixture_old", "fixture_new"}
    assert col.get_calls == [(1, 0), (1, 0), (1, 1)]
