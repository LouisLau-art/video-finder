"""台词（ASR）检索接入测试：台词通道、三路 RRF 融合与向后兼容。

只使用内存假集合（另有一个 tmp_path 下的空 chroma 库用于懒加载容错），
不读取 NAS、真实素材或真实向量库。装配模式沿用
test_video_level_results.py：importlib 动态加载 + monkeypatch ensure_state。
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
    "media_type",
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
    spec = importlib.util.spec_from_file_location("api_asr_retrieval_test", str(path))
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
    """构造不涉及真实文件的帧元数据（与 test_video_level_results 一致）。"""
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


def _asr_segment(
    video_key: str,
    video_id: str,
    start: float,
    text: str,
    segment_id: int = 0,
    relpath: str | None = None,
    end: float | None = None,
) -> dict[str, Any]:
    """构造 06c_index_asr.py 写入 transcripts 的元数据（字段对齐其契约）。"""
    relative_path = relpath or f"folder/{video_id}.mp4"
    minutes = int(start // 60)
    seconds = int(start - minutes * 60)
    return {
        "video_key": video_key,
        "video_id": video_id,
        "segment_id": segment_id,
        "share": "test_share",
        "relpath": relative_path,
        "time": start,
        "start": start,
        "end": start + 2.0 if end is None else end,
        "time_formatted": f"{minutes:02d}:{seconds:02d}",
        "text": text,
        "media_type": "video",
        "match_type": "transcript",
    }


class _FakeCollection:
    """内存帧集合（与现有测试同构）。"""

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


class _FakeTranscriptCollection:
    """内存台词集合：按距离升序返回前 n_results 条，模拟 chroma 行为。"""

    def __init__(self, segments: list[dict[str, Any]], distances: list[float]):
        self.segments = segments
        self.distances = distances
        self.calls = 0
        self.requested: list[int] = []

    def query(self, *, query_embeddings, n_results, include):
        self.calls += 1
        self.requested.append(n_results)
        order = sorted(
            range(len(self.segments)),
            key=lambda i: (self.distances[i], i),
        )
        selected = order[:n_results]
        return {
            "ids": [[f"seg-{i}" for i in selected]],
            "metadatas": [[dict(self.segments[i]) for i in selected]],
            "distances": [[self.distances[i] for i in selected]],
        }

    def count(self):
        return len(self.segments)


def _client_for(
    monkeypatch,
    frames,
    distances,
    transcripts=None,
    asr_distances=None,
    orders=None,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    api = _load_api_module()
    collection = _FakeCollection(frames, distances, orders)
    asr_collection = (
        None
        if transcripts is None
        else _FakeTranscriptCollection(
            transcripts,
            asr_distances
            if asr_distances is not None
            else [0.1 + 0.01 * i for i in range(len(transcripts))],
        )
    )

    class _Encoder:
        name = "test-encoder"

        def encode_texts(self, texts):
            return [[0.0, 1.0, 0.0, 0.0] for _ in texts]

    encoder = _Encoder()

    def ensure_state():
        api._CHROMA_COUNT = collection.count()
        return encoder, collection

    monkeypatch.setattr(api, "ensure_state", ensure_state)
    monkeypatch.setattr(api, "_ensure_transcripts_col", lambda: asr_collection)
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
    return api, TestClient(api.app), collection, asr_collection


def _post_search(client, *, top_k: int, query: str = "stable query"):
    response = client.post(
        "/api/search",
        json={"query": query, "top_k": top_k},
    )
    assert response.status_code == 200
    return response.json()


def _result_signature(payload):
    return [
        (row["video_id"], row["rank"], row["match_type"], row["matched_text"])
        for row in payload["data"]["results"]
    ]


# ---------- 1. 台词命中行契约 ----------

def test_transcript_hit_row_uses_transcript_time_and_text(monkeypatch):
    """台词独有素材：行标注 transcript，时间用台词起始秒，摘要用台词原句。"""
    frames = [
        _frame("video_noise_a", 1.0, "na", filename="scene_a.mp4"),
        _frame("video_noise_b", 2.0, "nb", filename="scene_b.mp4"),
    ]
    transcripts = [
        _asr_segment(
            "key_t",
            "video_t",
            start=8.5,
            text="这双鞋透气性极佳",
            relpath="deep/folder/video_t.mp4",
        )
    ]
    _, client, _collection, asr_collection = _client_for(
        monkeypatch, frames, [0.1, 0.2], transcripts=transcripts, asr_distances=[0.25]
    )

    payload = _post_search(client, top_k=3, query="透气性极佳")
    rows = payload["data"]["results"]

    # 台词通道按素材候选数（min(top_n, count)）查询一次。
    assert asr_collection.calls == 1
    assert asr_collection.requested == [1]
    # 语义两路 + 台词一路融合：噪音 1 名次贡献与台词 1 名次贡献同分，
    # 语义通道先进入融合因此排前。
    assert [row["video_id"] for row in rows] == ["video_noise_a", "video_t", "video_noise_b"]

    row = rows[1]
    assert row["rank"] == 2
    assert row["match_type"] == "transcript"
    assert row["matched_text"] == "这双鞋透气性极佳"
    assert row["time_seconds"] == 8.5
    assert row["time_formatted"] == "00:08.5"
    assert row["score"] == pytest.approx(0.875)
    assert row["media_type"] == "video"
    assert row["video_id"] == "video_t"
    assert row["video_name"] == "video_t.mp4"
    assert row["nas_path"] == "test_share/deep/folder/video_t.mp4"
    # 台词段没有帧图：封面留空，但契约字段必须存在。
    assert row["frame_image_url"] == ""
    assert set(row) == HYBRID_FIELDS

    # 语义行不受台词通道影响。
    assert all(
        other["match_type"] == "semantic" and other["matched_text"] == ""
        for other in rows
        if other["video_id"] != "video_t"
    )
    assert all(HYBRID_FIELDS.issubset(other) for other in rows)


# ---------- 2. 三路 RRF 融合 ----------

def test_three_way_fusion_order_and_no_duplicate_rows(monkeypatch):
    """语义 + 关键词 + 台词三路融合：顺序按 RRF 贡献，素材行不重复。

    video_kw 的语义距离最差（top_k 聚合被挤出语义集），保证它只经关键词
    通道命中；video_shared 三路全中；video_tr 只经台词通道命中。
    """
    frames = [
        {**_frame("video_sem", 4.0, "sem", filename="ordinary_scene.mp4"),
         "video_key": "key_sem"},
        {**_frame("video_kw", 5.0, "kw", filename="special_event.mp4"),
         "video_key": "key_kw"},
        {**_frame("video_shared", 6.0, "sh", filename="special_scene.mp4"),
         "video_key": "key_shared"},
        {**_frame("video_filler", 7.0, "fi", filename="filler_scene.mp4"),
         "video_key": "key_filler"},
        {**_frame("video_extra", 8.0, "ex", filename="plain_extra.mp4"),
         "video_key": "key_extra"},
    ]
    transcripts = [
        _asr_segment("key_shared", "video_shared", start=3.5, text="这句台词同时命中",
                     segment_id=0),
        _asr_segment("key_tr", "video_tr", start=12.0, text="仅台词命中"),
    ]
    _, client, _collection, asr_collection = _client_for(
        monkeypatch,
        frames,
        [0.1, 0.99, 0.5, 0.7, 0.8],
        transcripts=transcripts,
        asr_distances=[0.1, 0.2],
    )

    payload = _post_search(client, top_k=4, query="special")
    rows = payload["data"]["results"]

    assert asr_collection.requested == [2]
    # 三路贡献：shared = 3 路 / kw = 2 路（含提权） / sem = 1 路 / 仅台词 = 1 路。
    assert [row["video_id"] for row in rows] == [
        "video_shared",
        "video_kw",
        "video_sem",
        "video_tr",
    ]
    assert payload["data"]["total"] == 4
    assert len({row["video_id"] for row in rows}) == 4
    assert [row["rank"] for row in rows] == [1, 2, 3, 4]
    assert all(HYBRID_FIELDS.issubset(row) for row in rows)

    by_id = {row["video_id"]: row for row in rows}
    # 语义+关键词+台词三重命中：历史 "both"（语义+关键词）语义保持不变。
    shared = by_id["video_shared"]
    assert shared["match_type"] == "both"
    assert shared["matched_text"] == "special_scene"
    assert shared["time_seconds"] == 6.0  # 帧时刻，不是台词起始秒
    assert shared["frame_image_url"] == "/api/frames/sh.jpg"
    # 关键词独有：保持 0 分契约。
    assert by_id["video_kw"]["match_type"] == "keyword"
    assert by_id["video_kw"]["matched_text"] == "special_event"
    assert by_id["video_kw"]["score"] == 0.0
    # 语义独有：契约不变。
    assert by_id["video_sem"]["match_type"] == "semantic"
    assert by_id["video_sem"]["matched_text"] == ""
    # 台词独有：以台词段为代表。
    tr_only = by_id["video_tr"]
    assert tr_only["match_type"] == "transcript"
    assert tr_only["matched_text"] == "仅台词命中"
    assert tr_only["time_seconds"] == 12.0
    assert tr_only["time_formatted"] == "00:12.0"
    assert tr_only["score"] == pytest.approx(0.9)
    assert tr_only["frame_image_url"] == ""


def test_transcript_label_printed_priority_over_semantic(monkeypatch):
    """语义+台词双命中（无关键词）：台词命中优先，行以台词段为代表。"""
    frames = [
        _frame("video_frame", 1.0, "fr", filename="scene_a.mp4"),
        {**_frame("video_tgt", 2.0, "tg", filename="scene_tgt.mp4"),
         "video_key": "key_tgt"},
    ]
    transcripts = [
        _asr_segment("key_tgt", "video_tgt", start=33.0, text="口播的这句话"),
    ]
    _, client, _collection, _asr = _client_for(
        monkeypatch, frames, [0.1, 0.5], transcripts=transcripts, asr_distances=[0.2]
    )

    payload = _post_search(client, top_k=2, query="透气的跑步服")
    rows = payload["data"]["results"]

    assert [row["video_id"] for row in rows] == ["video_tgt", "video_frame"]
    target = rows[0]
    assert target["match_type"] == "transcript"
    assert target["matched_text"] == "口播的这句话"
    assert target["time_seconds"] == 33.0
    assert target["time_formatted"] == "00:33.0"
    assert target["score"] == pytest.approx(0.9)
    assert target["nas_path"] == "test_share/folder/video_tgt.mp4"
    assert target["frame_image_url"] == ""
    # 不在语义结果里的其它素材行保持原契约。
    assert rows[1]["match_type"] == "semantic"
    assert rows[1]["frame_image_url"] == "/api/frames/fr.jpg"


def test_transcript_and_keyword_hit_prefers_transcript_label(monkeypatch):
    """关键词+台词双命中（未进语义结果）：仍标注 transcript，摘要给台词原句。"""
    frames = [
        _frame("video_noise", 1.0, "na", filename="scene_a.mp4"),
        {**_frame("video_tgt", 2.0, "tg", filename="special_clip.mp4"),
         "video_key": "key_tgt"},
    ]
    transcripts = [
        _asr_segment("key_tgt", "video_tgt", start=20.0, text="这句台词"),
    ]
    # top_k=1：语义聚合只保留最佳素材，target 只经关键词 + 台词通道命中。
    _, client, _collection, _asr = _client_for(
        monkeypatch, frames, [0.1, 0.95], transcripts=transcripts, asr_distances=[0.15]
    )

    payload = _post_search(client, top_k=1, query="special")
    rows = payload["data"]["results"]

    assert len(rows) == 1
    row = rows[0]
    assert row["video_id"] == "video_tgt"
    assert row["match_type"] == "transcript"
    assert row["matched_text"] == "这句台词"
    assert row["time_seconds"] == 20.0
    assert row["score"] == pytest.approx(0.925)
    assert row["frame_image_url"] == ""


# ---------- 3. 向后兼容：台词集合不可用/空结果 ----------

def test_missing_transcripts_collection_preserves_two_way_behavior(monkeypatch):
    """transcripts 集合不存在（返回 None）时，行为与现有两路 RRF 完全一致。"""
    frames = [
        _frame("video_semantic", 1.0, "semantic_1", filename="ordinary_scene.mp4"),
        _frame("video_target", 2.0, "target_1", filename="special_event.mp4"),
        _frame("video_shared", 3.0, "shared_1", filename="shared_scene.mp4"),
        _frame("video_filler", 4.0, "filler_1", filename="filler_scene.mp4"),
    ]
    _, client, _collection, asr_collection = _client_for(
        monkeypatch, frames, [0.1, 0.2, 0.3, 0.4], transcripts=None
    )

    payload = _post_search(client, top_k=3, query="special")
    rows = payload["data"]["results"]

    assert asr_collection is None
    assert [row["video_id"] for row in rows[:2]] == ["video_target", "video_semantic"]
    assert rows[0]["match_type"] == "both"
    assert rows[0]["matched_text"] == "special_event"
    assert rows[1]["match_type"] == "semantic"
    assert all(row["match_type"] != "transcript" for row in rows)
    assert all(HYBRID_FIELDS.issubset(row) for row in rows)


def test_empty_transcripts_collection_keeps_existing_order(monkeypatch):
    """集合存在但无任何台词命中：结果顺序与两路融合逐位一致。"""
    frames = [
        _frame("video_semantic", 1.0, "semantic_1", filename="ordinary_scene.mp4"),
        _frame("video_target", 2.0, "target_1", filename="special_event.mp4"),
        _frame("video_shared", 3.0, "shared_1", filename="shared_scene.mp4"),
        _frame("video_filler", 4.0, "filler_1", filename="filler_scene.mp4"),
    ]
    distances = [0.1, 0.2, 0.3, 0.4]

    _, baseline_client, _c1, _a1 = _client_for(
        monkeypatch, frames, distances, transcripts=None
    )
    baseline = _post_search(baseline_client, top_k=3, query="special")

    _, empty_client, _c2, asr_empty = _client_for(
        monkeypatch, frames, distances, transcripts=[]
    )
    empty = _post_search(empty_client, top_k=3, query="special")

    # 集合为空：不发起查询（count==0 短路），结果与无集合时完全一致。
    assert asr_empty is not None
    assert asr_empty.calls == 0
    assert _result_signature(empty) == _result_signature(baseline)
    assert [row["match_type"] for row in empty["data"]["results"]] == [
        "both",
        "semantic",
        "semantic",
    ]


# ---------- 4. 纯函数与懒加载容错 ----------

def test_transcript_search_aggregates_best_segment_per_material(monkeypatch):
    """纯函数：同一素材取最佳（距离最小）台词段，identity 去重、按距离排序。"""
    api = _load_api_module()
    segments = [
        _asr_segment("key_a", "video_a", start=1.0, text="A 的第一段", segment_id=0),
        _asr_segment("key_a", "video_a", start=9.0, text="A 的第二段", segment_id=1),
        _asr_segment("key_b", "video_b", start=3.0, text="B 的台词", segment_id=0),
    ]
    col = _FakeTranscriptCollection(segments, [0.4, 0.1, 0.6])

    hits = api._transcript_search([0.0, 1.0], col, 50)

    assert [identity for identity, _score, _meta in hits] == ["key_a", "key_b"]
    assert hits[0][1] == pytest.approx(0.95)
    assert hits[0][2]["text"] == "A 的第二段"  # 最佳段作为素材代表
    assert hits[0][2]["start"] == 9.0
    assert hits[1][1] == pytest.approx(0.7)
    assert col.requested == [3]  # min(top_n, count)

    # top_n 收紧后候选变少，聚合结果随之收缩。
    narrowed = api._transcript_search([0.0, 1.0], col, 2)
    assert [identity for identity, _s, _m in narrowed] == ["key_a"]
    assert col.requested == [3, 2]

    #  None / 空集合都退化为空列表，绝不抛异常。
    assert api._transcript_search([0.0, 1.0], None, 50) == []
    assert api._transcript_search([0.0, 1.0], _FakeTranscriptCollection([], []), 50) == []


def test_ensure_transcripts_col_returns_none_without_db(monkeypatch):
    """未配置向量库路径（APP_DB 为空）时不触碰 chromadb，直接返回 None。"""
    api = _load_api_module()
    monkeypatch.setattr(api, "APP_DB", "")
    monkeypatch.setattr(api, "_TRANSCRIPTS_CLIENT", None)
    monkeypatch.setattr(api, "_TRANSCRIPTS_COL", None)

    assert api._ensure_transcripts_col() is None


def test_ensure_transcripts_col_tolerates_missing_collection(monkeypatch, tmp_path):
    """集合不存在（ASR 全量未跑）时返回 None；补齐后同路径下一次调用可用。"""
    chromadb = pytest.importorskip("chromadb")
    api = _load_api_module()
    monkeypatch.setattr(api, "APP_DB", str(tmp_path))
    monkeypatch.setattr(api, "_TRANSCRIPTS_CLIENT", None)
    monkeypatch.setattr(api, "_TRANSCRIPTS_COL", None)

    assert api._ensure_transcripts_col() is None

    client = chromadb.PersistentClient(path=str(tmp_path))
    client.get_or_create_collection(api.ASR_COLLECTION_NAME)
    col = api._ensure_transcripts_col()

    assert col is not None
    assert col.count() == 0
