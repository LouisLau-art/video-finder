"""最小 pytest 冒烟测试：CPU-only、无网、不碰真实 chroma_db/ 和 frames/.

加载方式：scripts/ 带数字前缀不能直接 import，必须用 importlib.util.spec_from_file_location.
不下载 SigLIP/open_clip 模型，不跑 ffmpeg.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str, filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


m01 = load_module("mod_01_extract", "01_extract.py")
m02 = load_module("mod_02_embed_index", "02_embed_index.py")
m03 = load_module("mod_03_search", "03_search.py")


# ---------- 01_extract ----------

def test_timestamps_short_scene_single_midpoint():
    # <2s 取 1 帧中点：(0,1) -> [0.5]
    assert m01.timestamps_for_scene(0.0, 1.0, None) == [0.5]


def test_timestamps_medium_scene_two_frames():
    # 2-5s 取 2 帧：(0,3) -> 三等分点 [1.0, 2.0]
    assert m01.timestamps_for_scene(0.0, 3.0, None) == [1.0, 2.0]


def test_timestamps_long_scene_three_frames():
    # >5s 取 3 帧：(0,8) -> [2.0, 4.0, 6.0]
    assert m01.timestamps_for_scene(0.0, 8.0, None) == [2.0, 4.0, 6.0]


def test_fallback_timestamps_duration_5_fps_1():
    assert m01.fallback_timestamps(5, fps=1.0) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_fallback_timestamps_duration_none():
    assert m01.fallback_timestamps(None) == [0.0]


# ---------- 03_search.map_query ----------

def test_map_query_chinese_contains_english():
    mapped = m03.map_query("白色帐篷 草坪")
    toks = mapped.split()
    assert "white" in toks
    assert "tent" in toks
    assert "lawn" in toks


def test_map_query_no_whitetent_sticking():
    mapped = m03.map_query("白色帐篷")
    assert "whitetent" not in mapped
    assert "white" in mapped.split()
    assert "tent" in mapped.split()


def test_map_query_english_passthrough():
    assert m03.map_query("white tent lawn") == "white tent lawn"


# ---------- 02_embed_index.DummyEncoder ----------

def test_build_encoder_dummy_name():
    enc = m02.build_encoder("dummy")
    assert enc.name == "dummy"


def test_dummy_encode_texts_deterministic():
    enc = m02.DummyEncoder()
    a = enc.encode_texts(["hello world"])
    b = enc.encode_texts(["hello world"])
    np.testing.assert_allclose(a, b)


def test_dummy_encode_texts_different_texts_differ():
    enc = m02.DummyEncoder()
    vecs = enc.encode_texts(["hello", "world"])
    assert vecs.shape == (2, 512)
    assert not np.allclose(vecs[0], vecs[1])


def test_dummy_encode_texts_normalized():
    enc = m02.DummyEncoder()
    vecs = enc.encode_texts(["白色帐篷", "white tent lawn", "hello"])
    for v in vecs:
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


def test_dummy_encode_images_shape():
    enc = m02.DummyEncoder()
    imgs = [
        Image.new("RGB", (32, 32), color=(255, 0, 0)),
        Image.new("RGB", (32, 32), color=(0, 255, 0)),
    ]
    out = enc.encode_images(imgs)
    assert out.shape == (2, 512)
    for v in out:
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


# ---------- 可选：DummyEncoder + chromadb 最小链路 ----------

def test_dummy_chroma_upsert_query_minimal(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    enc = m02.DummyEncoder()
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_test"))
    col = client.get_or_create_collection(
        name="test_frames", metadata={"hnsw:space": "cosine"}
    )
    texts = ["white tent", "black car"]
    embs = enc.encode_texts(texts)
    col.upsert(
        ids=["doc0", "doc1"],
        embeddings=embs.tolist(),
        metadatas=[{"video_id": "v0", "time": 0.0}, {"video_id": "v1", "time": 1.0}],
        documents=texts,
    )
    assert col.count() == 2
    q = enc.encode_texts(["white tent"])
    res = col.query(query_embeddings=q.tolist(), n_results=1)
    assert res["ids"][0][0] == "doc0"
