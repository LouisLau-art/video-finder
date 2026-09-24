"""视频级检索结果的 HTTP 行为测试。

只使用内存中的假向量库和元数据，不读取 NAS、素材原文件或真实向量库。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

RESULT_FIELDS = {
    "rank",
    "score",
    "match_percentage",
    "video_id",
    "video_name",
    "time_seconds",
    "time_formatted",
    "duration",
    "resolution",
    "frame_image_url",
    "nas_path",
    "full_nas_uri",
    "synology_web_url",
}
HYBRID_FIELDS = RESULT_FIELDS | {"match_type", "matched_text"}


def _load_api_module():
    pytest.importorskip("fastapi")
    path = SCRIPTS / "05_api.py"
    spec = importlib.util.spec_from_file_location("api_video_level_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _frame(
    video_id: str,
    time_seconds: float,
    name: str,
    filename: str | None = None,
    relpath: str | None = None,
) -> dict[str, Any]:
    """构造不涉及真实文件的帧元数据。"""
    video_name = filename or f"{video_id}.mp4"
    relative_path = relpath or f"folder/{video_id}.mp4"
    return {
        "video_id": video_id,
        "video_name": video_name,
        "time": time_seconds,
        "frame_path": f"virtual/{video_id}/{name}.jpg",
        "video_path": f"test_share/{relative_path}",
        "share": "test_share",
        "relpath": relative_path,
    }


class _FakeCollection:
    def __init__(
        self,
        frames: list[dict[str, Any]],
        distances: list[float],
        orders: list[list[int]] | None = None,
    ):
        self.frames = frames
        self.distances = distances
        self.orders = orders or [list(range(len(frames)))]
        self.calls = 0
        self.requested: list[int] = []

    def query(self, *, query_embeddings, n_results, include):
        order = self.orders[min(self.calls, len(self.orders) - 1)]
        self.calls += 1
        self.requested.append(n_results)
        selected = order[:n_results]
        return {
            "ids": [[f"frame-{i}" for i in selected]],
            "metadatas": [[self.frames[i] for i in selected]],
            "distances": [[self.distances[i] for i in selected]],
        }

    def count(self):
        return len(self.frames)

    def get(self, include=None):
        assert include == ["metadatas"]
        return {"metadatas": [dict(meta) for meta in self.frames]}


def _client_for(monkeypatch, frames, distances, orders=None):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    api = _load_api_module()
    collection = _FakeCollection(frames, distances, orders)

    class _Encoder:
        name = "test-encoder"

        def encode_texts(self, texts):
            return [[0.0, 1.0, 0.0, 0.0] for _ in texts]

    encoder = _Encoder()

    def ensure_state():
        api._CHROMA_COUNT = collection.count()
        return encoder, collection

    monkeypatch.setattr(api, "ensure_state", ensure_state)
    monkeypatch.setattr(api, "_CHROMA_COUNT", len(frames))
    monkeypatch.setattr(api, "_ENCODER_NAME", "test-encoder")
    monkeypatch.setattr(api, "APP_TRANSLATE", False)
    monkeypatch.setattr(
        api,
        "load_search_module",
        lambda: SimpleNamespace(
            resolve_query_for_encoder=lambda query, *args, **kwargs: (query, None)
        ),
    )
    # 让路径字段走可控的测试配置，不读取或暴露真实站点配置。
    monkeypatch.setattr(api, "NAS_HOST", "")
    monkeypatch.setattr(api, "NAS_PREFIX", "")
    monkeypatch.setattr(api, "SYNOLOGY_WEB_BASE", "")
    monkeypatch.setattr(api, "NAS_PATH_MAP", {})
    return api, TestClient(api.app), collection


def _post_search(client, *, top_k: int, query: str = "stable query"):
    response = client.post(
        "/api/search",
        json={"query": query, "top_k": top_k},
    )
    assert response.status_code == 200
    return response.json()


def _result_signature(payload):
    return [
        (row["video_id"], row["time_seconds"], row["frame_image_url"])
        for row in payload["data"]["results"]
    ]


def test_search_returns_one_row_per_video_and_keeps_contract(monkeypatch):
    frames = [
        _frame("video_a", 1.0, "a_1"),
        _frame("video_a", 4.0, "a_4"),
        _frame("video_b", 2.0, "b_2"),
        _frame("video_c", 3.0, "c_3"),
        _frame("video_d", 5.0, "d_5"),
    ]
    distances = [0.2, 0.1, 0.3, 0.4, 0.5]
    _, client, collection = _client_for(monkeypatch, frames, distances)

    payload = _post_search(client, top_k=3)
    rows = payload["data"]["results"]

    assert payload["data"]["total"] == 3
    assert len(rows) == 3
    assert collection.calls == 1
    assert [row["video_id"] for row in rows] == ["video_a", "video_b", "video_c"]
    assert len({row["video_id"] for row in rows}) == 3
    assert [row["rank"] for row in rows] == [1, 2, 3]

    representative = rows[0]
    assert representative["time_seconds"] == 4.0
    assert representative["time_formatted"] == "00:04.0"
    assert representative["frame_image_url"] == "/api/frames/a_4.jpg"
    assert representative["score"] == pytest.approx(0.95)
    assert representative["match_percentage"] == "95%"
    assert representative["nas_path"] == "test_share/folder/video_a.mp4"

    for row in rows:
        assert RESULT_FIELDS.issubset(row)
        for field in (
            "video_id",
            "video_name",
            "time_formatted",
            "frame_image_url",
            "nas_path",
            "duration",
            "resolution",
        ):
            assert row[field]


def test_repeated_query_keeps_order_and_representative_frame(monkeypatch):
    frames = [
        _frame("video_a", 8.0, "a_8"),
        _frame("video_a", 2.0, "a_2"),
        _frame("video_b", 5.0, "b_5"),
        _frame("video_c", 1.0, "c_1"),
    ]
    distances = [0.1, 0.1, 0.2, 0.3]
    # 模拟同分帧在不同查询批次中的返回顺序变化。
    _, client, _collection = _client_for(
        monkeypatch,
        frames,
        distances,
        orders=[[0, 1, 2, 3], [1, 0, 3, 2]],
    )

    first = _post_search(client, top_k=3)
    second = _post_search(client, top_k=3)

    assert _result_signature(first) == _result_signature(second)
    assert [row["video_id"] for row in first["data"]["results"]] == [
        "video_a",
        "video_b",
        "video_c",
    ]
    assert first["data"]["results"][0]["time_seconds"] == 2.0
    assert first["data"]["results"][0]["frame_image_url"] == "/api/frames/a_2.jpg"


def test_video_rank_is_not_worse_than_best_frame_rank(monkeypatch):
    frames = [
        _frame("video_a", 1.0, "a_1"),
        _frame("video_a", 2.0, "a_2"),
        _frame("video_b", 3.0, "b_3"),
        _frame("video_c", 4.0, "c_4"),
        _frame("video_d", 5.0, "d_5"),
    ]
    distances = [0.1, 0.2, 0.3, 0.4, 0.5]
    _, client, collection = _client_for(monkeypatch, frames, distances)

    payload = _post_search(client, top_k=5)
    rows = payload["data"]["results"]
    positions = {row["video_id"]: index for index, row in enumerate(rows, 1)}
    best_frame_rank = {
        "video_a": 1,
        "video_b": 3,
        "video_c": 4,
        "video_d": 5,
    }

    assert [row["video_id"] for row in rows] == list(best_frame_rank)
    for video_id, old_rank in best_frame_rank.items():
        assert positions[video_id] <= old_rank


def test_common_case_uses_one_overfetch_query(monkeypatch):
    frames = [
        _frame(
            ("video_a", "video_b", "video_c")[i % 3],
            float(i),
            f"frame_{i}",
        )
        for i in range(250)
    ]
    distances = [i / 1000 for i in range(250)]
    _, client, collection = _client_for(monkeypatch, frames, distances)

    payload = _post_search(client, top_k=3)

    assert payload["data"]["total"] == 3
    assert collection.calls == 1
    assert collection.requested == [200]


def test_candidate_expansion_stops_at_max_attempts(monkeypatch):
    frames = [
        _frame("video_a", float(i), f"frame_{i}")
        for i in range(500)
    ]
    distances = [i / 1000 for i in range(500)]
    api, client, collection = _client_for(monkeypatch, frames, distances)
    monkeypatch.setattr(api, "VIDEO_LEVEL_OVERSAMPLE_FACTOR", 1)
    monkeypatch.setattr(api, "VIDEO_LEVEL_MIN_CANDIDATES", 10)
    monkeypatch.setattr(api, "VIDEO_LEVEL_MAX_EXPANSIONS", 2)
    monkeypatch.setattr(api, "VIDEO_LEVEL_EXPANSION_BUDGET_MS", 10_000.0)

    payload = _post_search(client, top_k=10)

    assert payload["data"]["total"] == 1
    assert collection.calls == 3
    assert collection.requested == [10, 20, 40]
    assert max(collection.requested) <= len(frames)


def test_candidate_expansion_respects_time_budget(monkeypatch):
    frames = [
        _frame("video_a", float(i), f"frame_{i}")
        for i in range(500)
    ]
    distances = [i / 1000 for i in range(500)]
    api, client, collection = _client_for(monkeypatch, frames, distances)
    monkeypatch.setattr(api, "VIDEO_LEVEL_OVERSAMPLE_FACTOR", 1)
    monkeypatch.setattr(api, "VIDEO_LEVEL_MIN_CANDIDATES", 10)
    monkeypatch.setattr(api, "VIDEO_LEVEL_MAX_EXPANSIONS", 4)
    monkeypatch.setattr(api, "VIDEO_LEVEL_EXPANSION_BUDGET_MS", 100.0)

    clock = [0.0]

    def tick() -> float:
        value = clock[0]
        clock[0] += 0.06
        return value

    monkeypatch.setattr(api, "time", SimpleNamespace(perf_counter=tick))

    payload = _post_search(client, top_k=10)

    assert payload["data"]["total"] == 1
    assert collection.calls == 2
    assert collection.requested == [10, 20]


def test_candidate_expansion_switch_disabled_keeps_single_query(monkeypatch):
    frames = [
        _frame("video_a", float(i), f"frame_{i}")
        for i in range(500)
    ]
    distances = [i / 1000 for i in range(500)]
    api, client, collection = _client_for(monkeypatch, frames, distances)
    monkeypatch.setattr(api, "VIDEO_LEVEL_CANDIDATE_EXPANSION_ENABLED", False)

    payload = _post_search(client, top_k=10)
    rows = payload["data"]["results"]

    assert payload["data"]["total"] == 1
    assert collection.calls == 1
    assert collection.requested == [10]
    assert RESULT_FIELDS.issubset(rows[0])


def test_top_k_returns_all_videos_when_library_has_fewer(monkeypatch):
    frames = [
        _frame("video_a", 1.0, "a_1"),
        _frame("video_a", 2.0, "a_2"),
        _frame("video_b", 3.0, "b_3"),
    ]
    _, client, _collection = _client_for(
        monkeypatch, frames, [0.1, 0.2, 0.3]
    )

    payload = _post_search(client, top_k=10)

    assert payload["data"]["total"] == 2
    assert [row["video_id"] for row in payload["data"]["results"]] == [
        "video_a",
        "video_b",
    ]


def test_filename_keyword_recall_adds_keyword_only_result(monkeypatch):
    frames = [
        _frame("video_semantic", 1.0, "semantic_1", filename="ordinary_scene.mp4"),
        _frame("video_other", 2.0, "other_1", filename="another_scene.mp4"),
        _frame("video_target", 3.0, "target_1", filename="special_event.mp4"),
    ]
    _, client, _collection = _client_for(
        monkeypatch, frames, [0.1, 0.2, 0.3]
    )

    payload = _post_search(client, top_k=2, query="special")
    rows = payload["data"]["results"]
    target = next(row for row in rows if row["video_id"] == "video_target")

    assert target["match_type"] == "keyword"
    assert target["matched_text"] == "special_event"
    assert target["score"] == 0.0
    assert HYBRID_FIELDS.issubset(target)
    assert all(HYBRID_FIELDS.issubset(row) for row in rows)


def test_description_query_keeps_pure_semantic_contract(monkeypatch):
    frames = [
        _frame("video_a", 1.0, "a_1", filename="scene_one.mp4"),
        _frame("video_b", 2.0, "b_1", filename="scene_two.mp4"),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1, 0.2])

    payload = _post_search(client, top_k=2, query="跑者")

    assert [row["match_type"] for row in payload["data"]["results"]] == [
        "semantic",
        "semantic",
    ]
    assert [row["matched_text"] for row in payload["data"]["results"]] == ["", ""]


def test_rrf_keeps_keyword_and_semantic_leaders_in_top_area(monkeypatch):
    frames = [
        _frame("video_semantic", 1.0, "semantic_1", filename="ordinary_scene.mp4"),
        _frame("video_target", 2.0, "target_1", filename="special_event.mp4"),
        _frame("video_shared", 3.0, "shared_1", filename="shared_scene.mp4"),
        _frame("video_filler", 4.0, "filler_1", filename="filler_scene.mp4"),
    ]
    _, client, _collection = _client_for(
        monkeypatch, frames, [0.1, 0.2, 0.3, 0.4]
    )

    payload = _post_search(client, top_k=3, query="special")
    rows = payload["data"]["results"]

    assert [row["video_id"] for row in rows[:2]] == [
        "video_target",
        "video_semantic",
    ]
    assert rows[0]["match_type"] == "both"
    assert rows[1]["match_type"] == "semantic"


def test_typed_filename_requires_exact_match(monkeypatch):
    frames = [
        _frame("video_typed", 1.0, "typed_1", filename="OUT_0601.mp4"),
        _frame("video_other_typed", 2.0, "typed_2", filename="OUT_06012.mp4"),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1, 0.2])

    partial = _post_search(client, top_k=2, query="0601")
    assert all(row["match_type"] == "semantic" for row in partial["data"]["results"])
    assert all(row["matched_text"] == "" for row in partial["data"]["results"])

    exact = _post_search(client, top_k=2, query="OUT_0601")
    exact_target = next(
        row for row in exact["data"]["results"] if row["video_id"] == "video_typed"
    )
    other = next(
        row for row in exact["data"]["results"]
        if row["video_id"] == "video_other_typed"
    )
    assert exact_target["match_type"] in {"keyword", "both"}
    assert exact_target["matched_text"] == "OUT_0601"
    assert other["match_type"] == "semantic"


def test_hash_directory_is_not_keyword_text(monkeypatch):
    frames = [
        _frame(
            "video_plain",
            1.0,
            "plain_1",
            filename="plain_scene.mp4",
            relpath="12345678901234567890123456789012/plain_scene.mp4",
        ),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1])

    payload = _post_search(
        client,
        top_k=1,
        query="12345678901234567890123456789012",
    )

    assert payload["data"]["results"][0]["match_type"] == "semantic"
    assert payload["data"]["results"][0]["matched_text"] == ""


def test_meaningful_parent_directory_recall_and_matched_text(monkeypatch):
    frames = [
        _frame("video_semantic", 1.0, "semantic_1", filename="ordinary_scene.mp4"),
        _frame("video_other", 2.0, "other_1", filename="another_scene.mp4"),
        _frame(
            "video_target",
            3.0,
            "target_1",
            filename="plain_scene.mp4",
            relpath="special_series/plain_scene.mp4",
        ),
    ]
    _, client, _collection = _client_for(
        monkeypatch, frames, [0.1, 0.2, 0.3]
    )

    payload = _post_search(client, top_k=2, query="special_series")
    target = next(
        row for row in payload["data"]["results"]
        if row["video_id"] == "video_target"
    )

    assert target["match_type"] == "keyword"
    assert target["matched_text"] == "special_series"


def test_pure_numeric_directory_is_ignored(monkeypatch):
    frames = [
        _frame(
            "video_plain",
            1.0,
            "plain_1",
            filename="plain_scene.mp4",
            relpath="2024/plain_scene.mp4",
        ),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1])

    payload = _post_search(client, top_k=1, query="2024")

    assert payload["data"]["results"][0]["match_type"] == "semantic"
    assert payload["data"]["results"][0]["matched_text"] == ""


def test_readable_year_directory_is_matched(monkeypatch):
    frames = [
        _frame(
            "video_target",
            1.0,
            "target_1",
            filename="plain_scene.mp4",
            relpath="2024年/2026春季/plain_scene.mp4",
        ),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1])

    payload = _post_search(client, top_k=1, query="2026春季")
    target = payload["data"]["results"][0]

    assert target["match_type"] in {"keyword", "both"}
    assert target["matched_text"] == "2026春季"


def test_full_path_intermediate_layer_does_not_match(monkeypatch):
    frames = [
        _frame(
            "video_plain",
            1.0,
            "plain_1",
            filename="plain_scene.mp4",
            relpath="meaningful_dir/plain_scene.mp4",
        ),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1])

    payload = _post_search(
        client,
        top_k=1,
        query="meaningful_dir/plain_scene",
    )

    assert payload["data"]["results"][0]["match_type"] == "semantic"
    assert payload["data"]["results"][0]["matched_text"] == ""


def test_keyword_index_refreshes_after_collection_growth(monkeypatch):
    frames = [
        _frame("video_old", 1.0, "old_1", filename="old_event.mp4"),
    ]
    _, client, collection = _client_for(
        monkeypatch, frames, [0.1], orders=[[0]]
    )

    first = _post_search(client, top_k=1, query="old_event")
    assert first["data"]["results"][0]["video_id"] == "video_old"

    collection.frames.append(
        _frame("video_new", 2.0, "new_1", filename="new_event.mp4")
    )
    collection.distances.append(0.9)

    second = _post_search(client, top_k=2, query="new_event")
    new_row = next(
        row for row in second["data"]["results"] if row["video_id"] == "video_new"
    )
    assert new_row["match_type"] == "keyword"
    assert new_row["matched_text"] == "new_event"


def test_keyword_switch_returns_legacy_fields_and_order(monkeypatch):
    frames = [
        _frame("video_first", 1.0, "first_1", filename="first_scene.mp4"),
        _frame("video_second", 2.0, "second_1", filename="second_scene.mp4"),
        _frame("video_target", 3.0, "target_1", filename="special_event.mp4"),
    ]
    api, client, _collection = _client_for(
        monkeypatch, frames, [0.1, 0.2, 0.3]
    )

    monkeypatch.setattr(api, "KEYWORD_CHANNEL_ENABLED", False)
    legacy = _post_search(client, top_k=2, query="special")
    assert all(set(row) == RESULT_FIELDS for row in legacy["data"]["results"])
    assert [row["video_id"] for row in legacy["data"]["results"]] == [
        "video_first",
        "video_second",
    ]

    monkeypatch.setattr(api, "KEYWORD_CHANNEL_ENABLED", True)
    enabled = _post_search(client, top_k=2, query="special")
    assert any(
        row["video_id"] == "video_target" and row["match_type"] == "keyword"
        for row in enabled["data"]["results"]
    )
    assert all(HYBRID_FIELDS.issubset(row) for row in enabled["data"]["results"])
