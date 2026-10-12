"""检索层图片接入的行为测试（工单 02：聚合/关键词/RRF/结果行 + media_type 契约）。

只使用内存假向量库与临时目录文件，不读取 NAS、真实素材或真实向量库。
模式沿用 test_video_level_results.py：importlib 动态加载 + 假 collection。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# 视频行与图片行共享的基础契约（media_type 为工单 02 新增字段）。
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
# 图片行独有字段。
IMAGE_RESULT_FIELDS = RESULT_FIELDS | {"width", "height"}


def _load_api_module():
    pytest.importorskip("fastapi")
    path = SCRIPTS / "05_api.py"
    spec = importlib.util.spec_from_file_location(
        "api_image_retrieval_test", str(path)
    )
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
    """构造不涉及真实文件的视频帧元数据。"""
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


def _image_meta(
    image_key: str,
    stem: str,
    filename: str,
    relpath: str,
    width: int = 1920,
    height: int = 1080,
    image_path: str | None = None,
) -> dict[str, Any]:
    """构造图片条目元数据（字段集合对齐 06b_index_images.build_meta）。"""
    preview = image_path or f"virtual/{image_key}/{filename}"
    return {
        "image_key": image_key,
        "video_id": stem,
        "video_name": filename,
        "media_type": "image",
        "time": 0.0,
        "image_path": preview,
        "width": width,
        "height": height,
        "share": "test_share",
        "relpath": relpath,
        "frame_path": preview,
    }


class _FakeCollection:
    def __init__(self, frames: list[dict[str, Any]], distances: list[float]):
        self.frames = frames
        self.distances = distances
        self.calls = 0
        self.requested: list[int] = []

    def query(self, *, query_embeddings, n_results, include):
        self.calls += 1
        self.requested.append(n_results)
        selected = list(range(len(self.frames)))[:n_results]
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


def _client_for(monkeypatch, frames, distances):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    api = _load_api_module()
    collection = _FakeCollection(frames, distances)

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


def test_semantic_search_mixes_image_and_video_rows_with_contract(monkeypatch):
    """语义结果混排图片与视频：图片行契约齐全，视频行只追加 media_type。"""
    frames = [
        _image_meta(
            "img00001", "photo", "photo.jpg", "shoots/photo.jpg",
            width=4000, height=3000,
        ),
        _frame("video_a", 1.0, "a_1"),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1, 0.2])

    payload = _post_search(client, top_k=5)
    rows = payload["data"]["results"]

    assert payload["data"]["total"] == 2
    assert [row["media_type"] for row in rows] == ["image", "video"]

    image = rows[0]
    assert IMAGE_RESULT_FIELDS.issubset(image)
    assert image["video_id"] == "photo"
    assert image["video_name"] == "photo.jpg"  # 保留扩展名，而非 .mp4
    assert image["time_seconds"] == 0.0
    assert image["time_formatted"] == "--:--"
    assert image["duration"] == "--:--"
    assert image["width"] == 4000
    assert image["height"] == 3000
    assert image["resolution"] == "4000x3000"
    assert image["nas_path"] == "test_share/shoots/photo.jpg"
    assert image["frame_image_url"] == "/api/frames/photo.jpg"

    video = rows[1]
    assert RESULT_FIELDS.issubset(video)
    assert "width" not in video and "height" not in video
    assert video["video_name"] == "video_a.mp4"
    assert video["time_formatted"] == "00:01.0"
    assert video["resolution"] == "1080P"


def test_image_and_video_same_stem_stay_separate(monkeypatch):
    """同 stem 的图片与视频（不同 key）聚合后必须是两条结果。"""
    frames = [
        _image_meta("imgaaaa11", "scene", "scene.jpg", "dir_a/scene.jpg"),
        {**_frame("scene", 1.0, "f_1", filename="scene.mp4"),
         "video_key": "vidbbbb22", "relpath": "dir_b/scene.mp4"},
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1, 0.2])

    payload = _post_search(client, top_k=10)
    rows = payload["data"]["results"]

    assert payload["data"]["total"] == 2
    assert len(rows) == 2
    assert {row["media_type"] for row in rows} == {"image", "video"}
    assert {row["nas_path"] for row in rows} == {
        "test_share/dir_a/scene.jpg",
        "test_share/dir_b/scene.mp4",
    }


def test_image_row_nas_links_contract(monkeypatch):
    """图片行照常生成 NAS 路径、SMB URI 与群晖深链三件套。"""
    frames = [
        _image_meta("img00001", "poster", "poster.png", "brand_kv/poster.png"),
    ]
    api, client, _collection = _client_for(monkeypatch, frames, [0.1])
    monkeypatch.setattr(api, "NAS_HOST", "smb://nas.local")
    monkeypatch.setattr(api, "SYNOLOGY_WEB_BASE", "https://nas.example")

    row = _post_search(client, top_k=1)["data"]["results"][0]

    assert row["nas_path"] == "test_share/brand_kv/poster.png"
    assert row["full_nas_uri"] == "smb://nas.local/test_share/brand_kv/poster.png"
    assert row["synology_web_url"].startswith(
        "https://nas.example/index.cgi?launchApp=SYNO.SDS.App.FileStation3.Instance"
        "&launchParam=openfile%3D"
    )
    assert "%252Ftest_share%252Fbrand_kv%252F" in row["synology_web_url"]


def test_keyword_channel_hits_image_stem_and_parent_dir(monkeypatch):
    """关键词通道命中图片文件名 stem 与父目录片段。"""
    frames = [
        _frame("video_a", 1.0, "a_1", filename="ordinary_scene.mp4"),
        _frame("video_b", 2.0, "b_1", filename="another_scene.mp4"),
        _image_meta(
            "img00001", "white_sneaker", "white_sneaker.png",
            "product_shots/white_sneaker.png",
        ),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.1, 0.2, 0.9])

    by_stem = _post_search(client, top_k=2, query="white")
    stem_row = next(
        row for row in by_stem["data"]["results"]
        if row["video_id"] == "white_sneaker"
    )
    assert stem_row["match_type"] == "keyword"
    assert stem_row["matched_text"] == "white_sneaker"
    assert stem_row["media_type"] == "image"
    assert stem_row["nas_path"] == "test_share/product_shots/white_sneaker.png"

    by_parent = _post_search(client, top_k=2, query="product_shots")
    parent_row = next(
        row for row in by_parent["data"]["results"]
        if row["video_id"] == "white_sneaker"
    )
    assert parent_row["match_type"] == "keyword"
    assert parent_row["matched_text"] == "product_shots"


def test_typed_image_filename_requires_exact_match(monkeypatch):
    """类型化图片文件名只允许精确匹配（与视频同规则）。"""
    frames = [
        _image_meta("img00001", "OUT_0601", "OUT_0601.png",
                    "raw_shots/OUT_0601.png"),
        _frame("video_a", 1.0, "a_1", filename="ordinary_scene.mp4"),
    ]
    _, client, _collection = _client_for(monkeypatch, frames, [0.9, 0.1])

    partial = _post_search(client, top_k=2, query="0601")
    assert all(row["match_type"] == "semantic" for row in partial["data"]["results"])
    assert all(row["matched_text"] == "" for row in partial["data"]["results"])

    exact = _post_search(client, top_k=2, query="OUT_0601")
    target = next(
        row for row in exact["data"]["results"]
        if row["video_id"] == "OUT_0601"
    )
    assert target["match_type"] in {"keyword", "both"}
    assert target["matched_text"] == "OUT_0601"


def test_selective_boost_promotes_image_target(monkeypatch):
    """选择性提权对图片条目同样生效：数字/目录式关键词命中图片挤进前 top_k。"""
    noise = [
        _frame(f"noise_{i}", float(i), f"n{i}", filename=f"scene_{i}.mp4")
        for i in range(1, 6)
    ]
    images = [
        _image_meta("imgkw1", "clip_a", "clip_a.png", "solo_series/clip_a.png"),
        _image_meta("imgkw2", "clip_b", "clip_b.png", "solo_series/clip_b.png"),
        _image_meta("imgtarget", "clip_t", "clip_t.png", "solo_series/clip_t.png"),
    ]
    frames = noise + images
    distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    api, client, _collection = _client_for(monkeypatch, frames, distances)

    # 命中 3 个素材、占比很小 → 门控判定为提权。
    assert api._keyword_channel_weight(3, len(frames)) == pytest.approx(
        api.KEYWORD_BOOST_WEIGHT
    )

    boosted = _post_search(client, top_k=5, query="solo_series")
    rows = boosted["data"]["results"]
    target = next(row for row in rows if row["video_id"] == "clip_t")

    assert target["match_type"] == "keyword"
    assert target["media_type"] == "image"
    assert target["matched_text"] == "solo_series"
    assert IMAGE_RESULT_FIELDS.issubset(target)

    # 对照：权重强制 1.0 后目标跌出前 5，证明差异来自选择性提权本身。
    monkeypatch.setattr(api, "KEYWORD_BOOST_WEIGHT", 1.0)
    baseline = _post_search(client, top_k=5, query="solo_series")
    assert all(row["video_id"] != "clip_t" for row in baseline["data"]["results"])


def test_api_frames_serves_indexed_image_original(monkeypatch, tmp_path):
    """/api/frames 能通过元数据推导的目录取到图片原图（假文件树）。"""
    photos = tmp_path / "photos"
    photos.mkdir()
    original = photos / "poster.png"
    original.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png")

    frames_dir = tmp_path / "frames_out"
    frames_dir.mkdir()
    frame_file = frames_dir / "a_1.jpg"
    frame_file.write_bytes(b"\xff\xd8fake-jpeg")

    meta = _image_meta(
        "img00001", "poster", "poster.png", "brand_kv/poster.png",
        image_path=str(original),
    )
    api, client, _collection = _client_for(monkeypatch, [meta], [0.1])
    monkeypatch.setattr(api, "APP_FRAMES_DIRS", [str(frames_dir)])

    # 先走一次搜索：图片行的元数据目录经语义命中路径登记为兜底候选。
    search_payload = _post_search(client, top_k=1)
    assert search_payload["data"]["results"][0]["media_type"] == "image"

    image_resp = client.get("/api/frames/poster.png")
    assert image_resp.status_code == 200
    assert image_resp.content == b"\x89PNG\r\n\x1a\nnot-a-real-png"
    assert image_resp.headers["content-type"].startswith("image/png")

    # 帧图目录优先级行为不变。
    frame_resp = client.get("/api/frames/a_1.jpg")
    assert frame_resp.status_code == 200
    assert frame_resp.content == b"\xff\xd8fake-jpeg"

    assert client.get("/api/frames/missing.png").status_code == 404


def test_image_row_tolerates_missing_optional_fields(monkeypatch):
    """旧/异构图片元数据缺 width/height 时不崩，字段降级为 0/空串。"""
    meta = {
        "image_key": "img00001",
        "video_id": "legacy",
        "video_name": "legacy.png",
        "media_type": "image",
        "time": 0.0,
        "image_path": "virtual/img00001/legacy.png",
        "share": "test_share",
        "relpath": "brand_kv/legacy.png",
        "frame_path": "virtual/img00001/legacy.png",
    }
    _, client, _collection = _client_for(monkeypatch, [meta], [0.1])

    row = _post_search(client, top_k=1)["data"]["results"][0]

    assert IMAGE_RESULT_FIELDS.issubset(row)
    assert row["media_type"] == "image"
    assert row["width"] == 0
    assert row["height"] == 0
    assert row["resolution"] == ""
    assert row["nas_path"] == "test_share/brand_kv/legacy.png"
    assert row["time_formatted"] == "--:--"
