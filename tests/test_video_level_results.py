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

    def get(self, include=None, limit=None, offset=0):
        assert include == ["metadatas"]
        start = max(0, int(offset))
        end = len(self.frames) if limit is None else start + max(0, int(limit))
        return {"metadatas": [dict(meta) for meta in self.frames[start:end]]}


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

    monkeypatch.setattr(
        api,
        "time",
        SimpleNamespace(perf_counter=tick, monotonic=tick),
    )

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
    api, client, collection = _client_for(
        monkeypatch, frames, [0.1], orders=[[0]]
    )
    monkeypatch.setattr(api, "KEYWORD_INDEX_REBUILD_MIN_INTERVAL", 0.0)

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


def test_same_stem_different_video_key_stays_separate(monkeypatch):
    """同名文件（同 video_id、不同 video_key）必须各自成为一条结果。

    回归锁：检索层过去按非唯一的 video_id 聚合，导致同名不同目录的视频被
    合并、部分物理文件不可检索。身份必须用全局唯一 video_key。
    """
    frames = [
        {**_frame("scene", 1.0, "a_1", filename="scene.mp4"),
         "video_key": "aaaa1111", "relpath": "dir_a/scene.mp4"},
        {**_frame("scene", 2.0, "b_1", filename="scene.mp4"),
         "video_key": "bbbb2222", "relpath": "dir_b/scene.mp4"},
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1, 0.2])

    payload = _post_search(client, top_k=10)
    rows = payload["data"]["results"]

    assert payload["data"]["total"] == 2
    assert len(rows) == 2
    assert {row["nas_path"] for row in rows} == {
        "test_share/dir_a/scene.mp4",
        "test_share/dir_b/scene.mp4",
    }


def test_keyword_recall_matches_each_same_stem_file(monkeypatch):
    """关键词命中同名文件时，两条结果都要出现，且各自的 matched_text 正确。"""
    frames = [
        {**_frame("scene", 1.0, "a_1", filename="scene.mp4"),
         "video_key": "aaaa1111", "relpath": "big_event/scene.mp4"},
        {**_frame("scene", 2.0, "b_1", filename="scene.mp4"),
         "video_key": "bbbb2222", "relpath": "other/scene.mp4"},
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1, 0.2])

    payload = _post_search(client, top_k=10, query="big_event")
    rows = payload["data"]["results"]

    assert len(rows) == 2
    hit = next(row for row in rows if row["nas_path"] == "test_share/big_event/scene.mp4")
    assert hit["match_type"] in {"keyword", "both"}
    assert hit["matched_text"] == "big_event"


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


def test_keyword_channel_weight_boundaries(monkeypatch):
    """选择性门控的纯函数边界：0 命中、选择性、泛化、空/极小库。"""
    api = _load_api_module()
    boost = api.KEYWORD_BOOST_WEIGHT
    assert boost == pytest.approx(1.25)

    # 无命中不改变权重。
    assert api._keyword_channel_weight(0, 7899) == 1.0
    assert api._keyword_channel_weight(-3, 7899) == 1.0

    # 命中数少（专有名词）→ 提权；阈值包含端点。
    assert api._keyword_channel_weight(29, 7899) == pytest.approx(boost)
    assert api._keyword_channel_weight(40, 7899) == pytest.approx(boost)
    # 命中超出阈值（泛化词）→ 保持 1.0。
    assert api._keyword_channel_weight(41, 7899) == 1.0

    # 空库 / 极小库不能崩，且走绝对下限。
    assert api._keyword_channel_weight(5, 0) == pytest.approx(boost)
    assert api._keyword_channel_weight(5, 1) == pytest.approx(boost)

    # 语料变大时阈值按比例放宽。
    assert api._keyword_channel_weight(500, 100000) == pytest.approx(boost)
    assert api._keyword_channel_weight(501, 100000) == 1.0


def _legacy_rrf(rankings, k):
    """改动前的 RRF 参考实现，仅用于回归锁对比。"""
    scores = {}
    first_order = {}
    base = max(1, int(k))
    for ranking in rankings:
        seen = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            if item not in first_order:
                first_order[item] = len(first_order)
            scores[item] = scores.get(item, 0.0) + 1.0 / (base + rank)
    return sorted(scores, key=lambda item: (-scores[item], first_order[item]))


def test_rrf_without_weights_matches_legacy_order(monkeypatch):
    """回归锁：不传 weights 时，融合结果与改动前逐位一致。"""
    api = _load_api_module()
    cases = [
        [["a", "b", "c", "d"], ["b", "d", "e"], ["c", "a", "e", "f"]],
        [["x", "y"], ["y", "x"]],
        [["only"], []],
        [["dup", "dup", "z"], ["z", "dup"]],
    ]
    for rankings in cases:
        assert api.reciprocal_rank_fusion(*rankings, k=api.RRF_K) == _legacy_rrf(
            rankings, api.RRF_K
        )
        # 显式 weights=None 与全 1.0 权重都必须退化为旧行为。
        equal_weights = tuple(1.0 for _ in rankings)
        assert api.reciprocal_rank_fusion(
            *rankings, k=api.RRF_K, weights=equal_weights
        ) == _legacy_rrf(rankings, api.RRF_K)

    # 单参数扁平化兼容分支。
    flat = [["a", "b", "c"], ["b", "d"]]
    assert api.reciprocal_rank_fusion(flat, k=api.RRF_K) == _legacy_rrf(
        flat, api.RRF_K
    )
    assert api.reciprocal_rank_fusion(flat, k=api.RRF_K, weights=(1.0, 1.0)) == (
        _legacy_rrf(flat, api.RRF_K)
    )


def test_rrf_applies_per_channel_weights(monkeypatch):
    """权重确实作用于对应通道，且缺省通道按 1.0 处理。"""
    api = _load_api_module()
    # 关键词通道提权后，"b" 单通道贡献超过 "a"。
    boosted = api.reciprocal_rank_fusion(["a"], ["b"], k=api.RRF_K, weights=(1.0, 1.25))
    assert boosted[:2] == ["b", "a"]
    # 权重长度不足时后续通道按 1.0，结果回到逐位旧行为。
    partial = api.reciprocal_rank_fusion(
        ["a"], ["b"], k=api.RRF_K, weights=(1.0,)
    )
    assert partial == _legacy_rrf([["a"], ["b"]], api.RRF_K)


def _noise_frames():
    """语义通道占位素材：文件名不与关键词查询匹配。"""
    return [
        _frame("noise_a", 1.0, "na", filename="scene_a.mp4"),
        _frame("noise_b", 2.0, "nb", filename="scene_b.mp4"),
        _frame("noise_c", 3.0, "nc", filename="scene_c.mp4"),
        _frame("noise_d", 4.0, "nd", filename="scene_d.mp4"),
        _frame("noise_e", 5.0, "ne", filename="scene_e.mp4"),
    ]


def test_selective_keyword_query_promotes_keyword_only_target(monkeypatch):
    """选择性查询：语义 top_k 被噪音占满，关键词独有目标提权后挤进前 top_k。

    目标只被关键词命中且在关键词通道排第 3。把权重强制回 1.0 后目标跌出
    前 5，两个方向都断言，证明差异来自选择性提权本身。
    """
    frames = _noise_frames() + [
        _frame("kw_a", 6.0, "ka", filename="clip_a.mp4", relpath="solo_series/kw_a.mp4"),
        _frame("kw_b", 7.0, "kb", filename="clip_b.mp4", relpath="solo_series/kw_b.mp4"),
        _frame(
            "t_target",
            8.0,
            "tt",
            filename="clip_t.mp4",
            relpath="solo_series/t_target.mp4",
        ),
    ]
    distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    api, client, _collection = _client_for(monkeypatch, frames, distances)

    # 命中 3 个素材，占比很小 → 门控判定为提权。
    assert api._keyword_channel_weight(3, len(frames)) == pytest.approx(
        api.KEYWORD_BOOST_WEIGHT
    )

    boosted = _post_search(client, top_k=5, query="solo_series")
    boosted_ids = [row["video_id"] for row in boosted["data"]["results"]]
    target = next(row for row in boosted["data"]["results"] if row["video_id"] == "t_target")

    assert "t_target" in boosted_ids
    assert target["match_type"] == "keyword"
    assert target["matched_text"] == "solo_series"

    # 对照：同一份数据强制权重 1.0，目标从第 3 名降到前 5 之外。
    monkeypatch.setattr(api, "KEYWORD_BOOST_WEIGHT", 1.0)
    baseline = _post_search(client, top_k=5, query="solo_series")
    baseline_ids = [row["video_id"] for row in baseline["data"]["results"]]
    assert "t_target" not in baseline_ids
    assert boosted_ids.index("t_target") < 5


def test_generic_keyword_query_is_not_boosted(monkeypatch):
    """泛化查询（命中超阈值）：权重保持 1.0，前 top_k 不被关键词独有结果淹没。

    目标在关键词通道与选择性用例一样排第 3，唯一区别是命中数超过选择性
    阈值；默认走 1.0 时目标跌出前 5，强行当选择性处理时才进入前 5。
    """
    solo = [
        _frame("solo_001_a", 6.0, "s1", filename="clip_1.mp4", relpath="solo_series/solo_001_a.mp4"),
        _frame("solo_002_b", 7.0, "s2", filename="clip_2.mp4", relpath="solo_series/solo_002_b.mp4"),
        _frame("solo_003_target", 8.0, "s3", filename="clip_3.mp4", relpath="solo_series/solo_003_target.mp4"),
    ] + [
        _frame(
            f"solo_{i:03d}_x",
            9.0 + i,
            f"s{i}",
            filename=f"clip_{i}.mp4",
            relpath=f"solo_series/solo_{i:03d}_x.mp4",
        )
        for i in range(100, 147)
    ]
    frames = _noise_frames() + solo
    distances = [0.1, 0.2, 0.3, 0.4, 0.5] + [
        0.6 + 0.001 * i for i in range(len(solo))
    ]
    api, client, _collection = _client_for(monkeypatch, frames, distances)

    # 命中 50 个素材、超选择性下限 → 门控判定不提权。
    assert api._keyword_channel_weight(len(solo), len(frames)) == 1.0

    plain = _post_search(client, top_k=5, query="solo_series")
    plain_ids = [row["video_id"] for row in plain["data"]["results"]]
    assert "solo_003_target" not in plain_ids

    # 对照：把该查询强行纳入选择性区间（阈值抬到命中数之上）后目标进入前 5，
    # 证明上面的差异来自选择性门控而非数据构造。
    monkeypatch.setattr(api, "KEYWORD_SELECTIVE_MIN_HITS", len(frames) + 1)
    forced = _post_search(client, top_k=5, query="solo_series")
    forced_ids = [row["video_id"] for row in forced["data"]["results"]]
    assert "solo_003_target" in forced_ids
