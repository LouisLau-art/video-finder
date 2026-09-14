"""最小 pytest 冒烟测试：CPU-only、无网、不碰真实 chroma_db/ 和 frames/.

加载方式：scripts/ 带数字前缀不能直接 import，必须用 importlib.util.spec_from_file_location.
不下载 SigLIP/open_clip 模型，不跑 ffmpeg.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

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
m04 = load_module("mod_04_eval", "04_eval.py")


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


# ---------- 01_extract 黑场/纯色卡过滤 ----------

def _checkerboard(size: int, a: int, b: int) -> Image.Image:
    """生成 a/b 两色棋盘图，用于构造确定性的 mean/std."""
    img = Image.new("L", (size, size), color=a)
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = b if (x + y) % 2 == 0 else a
    return img


def test_blank_black_image_dropped():
    # 纯黑(片头尾黑场)均值远低于阈值 -> 丢弃
    assert m01.is_blank_frame(Image.new("RGB", (64, 64), (0, 0, 0))) is True


def test_blank_solid_color_dropped():
    # 纯色图(任意颜色) std≈0 -> 丢弃
    for color in [(255, 255, 255), (0, 128, 255), (200, 30, 30)]:
        assert m01.is_blank_frame(Image.new("RGB", (64, 64), color)) is True


def test_normal_colorful_frame_kept():
    img = Image.new("RGB", (96, 96), (40, 90, 160))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 50, 60], fill=(230, 60, 40))
    d.ellipse([50, 40, 90, 85], fill=(250, 230, 60))
    d.line([0, 0, 95, 95], fill=(255, 255, 255), width=3)
    assert m01.is_blank_frame(img) is False


def test_dark_but_textured_night_frame_kept():
    # 暗光/夜景正常帧: 背景均值低但高于 mean 阈值，且有纹理/高光 -> 不误杀
    img = Image.new("RGB", (64, 64), (10, 12, 18))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 40, 40], fill=(90, 100, 130))
    d.line([0, 63, 63, 0], fill=(200, 200, 220), width=2)
    assert m01.is_blank_frame(img) is False


def test_blank_threshold_mean_boundary():
    # 棋盘 0/16: mean 恰为 8.0、std 8.0；严格小于阈值才算黑场
    img = _checkerboard(64, 0, 16)
    assert m01.frame_gray_stats(img)[0] == pytest.approx(8.0, abs=0.01)
    assert m01.is_blank_frame(img, mean_threshold=8.0) is False
    assert m01.is_blank_frame(img, mean_threshold=8.1) is True


def test_blank_threshold_std_boundary():
    # 棋盘 100/112: std≈6.0；严格小于阈值才算纯色
    img = _checkerboard(64, 100, 112)
    _mean, std = m01.frame_gray_stats(img)
    assert std == pytest.approx(6.0, abs=0.05)
    assert m01.is_blank_frame(img, std_threshold=6.0) is False
    assert m01.is_blank_frame(img, std_threshold=6.1) is True


def test_blank_low_std_pattern_dropped():
    # 低对比双色图案(类似纯色卡/压缩纯色块) std<阈值 -> 丢弃
    img = _checkerboard(64, 120, 128)
    _mean, std = m01.frame_gray_stats(img)
    assert std < m01.BLANK_STD_THRESHOLD
    assert m01.is_blank_frame(img) is True


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


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def test_map_query_huabao_no_cjk_residue():
    # 回归：花苞不得被切碎成 flower苞，必须完整映射且无 CJK 残留
    mapped = m03.map_query("花苞")
    assert not _has_cjk(mapped)
    assert mapped.split() == ["flower", "bud"]


def test_map_query_hua_still_flower():
    assert m03.map_query("花") == "flower"


def test_map_query_long_word_priority():
    # 长词优先：汽车(表内长词)不得被车(短词)切碎成"汽 car"；颜色+物组合同步验证
    assert m03.map_query("汽车") == "car"
    assert not _has_cjk(m03.map_query("汽车"))
    assert m03.map_query("白色花苞").split() == ["white", "flower", "bud"]


# ---------- 03_search 中文翻译层: 只测门控与回退, 不下载 300MB 模型 ----------

class _StubTranslator:
    """测试桩: 记录调用并按配置返回/抛错, 绝不触发 transformers."""

    def __init__(self, out: str = "red vest", error: Exception | None = None):
        self.out = out
        self.error = error
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.out


def test_has_cjk_detection():
    assert m03.has_cjk("红色背心") is True
    assert m03.has_cjk("红色 vest") is True
    assert m03.has_cjk("red vest") is False
    assert m03.has_cjk("123 !@#") is False


def test_resolve_query_no_cjk_zero_overhead(monkeypatch):
    # 纯英文: 零翻译开销，绝不构造翻译器(懒加载)
    called: list[str] = []
    monkeypatch.setattr(m03, "ZhEnTranslator", lambda: called.append("load"))
    q, tr = m03.resolve_query("white tent lawn")
    assert (q, tr) == ("white tent lawn", None)
    assert called == []


def test_resolve_query_no_cjk_skips_injected_translator():
    bomb = _StubTranslator(error=RuntimeError("不应被调用"))
    q, tr = m03.resolve_query("white tent", translator=bomb)
    assert (q, tr) == ("white tent", None)
    assert bomb.calls == []


def test_resolve_query_cjk_uses_translator():
    stub = _StubTranslator(out="red vest")
    q, tr = m03.resolve_query("红色背心", translator=stub)
    assert q == "red vest"
    assert tr == "red vest"
    assert stub.calls == ["红色背心"]


def test_resolve_query_cjk_translator_none_falls_back_zh2en():
    # translator 缺失(如离线): 回退现有 ZH2EN 映射
    q, tr = m03.resolve_query("白色帐篷 草坪", translator=None)
    toks = q.split()
    assert tr is None
    assert "white" in toks and "tent" in toks and "lawn" in toks


def test_resolve_query_translate_failure_falls_back_zh2en():
    bomb = _StubTranslator(error=RuntimeError("offline"))
    q, tr = m03.resolve_query("白色帐篷", translator=bomb)
    assert (q, tr) == ("white tent", None)
    assert bomb.calls == ["白色帐篷"]


def test_resolve_query_empty_translation_falls_back():
    stub = _StubTranslator(out="")
    q, tr = m03.resolve_query("白色帐篷", translator=stub)
    assert (q, tr) == ("white tent", None)
    assert stub.calls == ["白色帐篷"]


def test_get_translator_load_failure_returns_none(monkeypatch):
    # 加载失败(无网/缺依赖)必须返回 None 并缓存失败, 不抛异常
    monkeypatch.setattr(m03, "_TRANSLATOR_TRIED", False)
    monkeypatch.setattr(m03, "_TRANSLATOR", None)

    def _boom():
        raise RuntimeError("no net")

    monkeypatch.setattr(m03, "ZhEnTranslator", _boom)
    assert m03.get_translator() is None
    assert m03._TRANSLATOR_TRIED is True
    # 二次调用直接吃缓存, 不再重试加载
    assert m03.get_translator() is None


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


# ---------- 02_embed_index.Siglip2Encoder: 文本按官方 max_length 编码 ----------

def test_siglip2_encode_texts_uses_max_length_padding():
    # SigLIP2 官方要求文本 padding="max_length"(训练时用法): 所有文本补到 64 token
    # 再取最后一个 token 的 hidden state; 定长 padding 同时保证批量与单条编码一致,
    # 不受 batch 组成影响(此前 padding=True 会被 pad token 污染)。
    import torch

    calls: list[dict] = []

    class _FakeProcessor:
        def __call__(self, text, return_tensors=None, **kwargs):
            calls.append({"text": list(text), "return_tensors": return_tensors, **kwargs})
            return {"input_ids": torch.ones(len(text), 64, dtype=torch.long)}

    class _FakeModel:
        def get_text_features(self, input_ids, **kwargs):
            return torch.ones(input_ids.shape[0], 4)

    enc = object.__new__(m02.Siglip2Encoder)
    enc.processor = _FakeProcessor()
    enc.model = _FakeModel()
    texts = ["red vest", "a photo of a jacket with stripes", "3d animation"]
    out = enc.encode_texts(texts)
    assert len(calls) == 1  # 一次批量调用(定长 padding 不需要逐条)
    assert calls[0]["text"] == texts
    assert calls[0]["padding"] == "max_length"
    assert calls[0]["max_length"] == 64
    assert calls[0]["truncation"] is True
    assert out.shape == (3, 4)
    for v in out:
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


# ---------- 03_search encoder 门控: cnclip 原样, siglip2/openclip 走翻译层 ----------

def test_resolve_query_for_encoder_cnclip_raw_no_translator():
    # cnclip 中文原生: 原样返回, 即使注入了翻译桩也绝不调用
    bomb = _StubTranslator(error=RuntimeError("不应被调用"))
    q, tr = m03.resolve_query_for_encoder("红色背心", "cnclip", translator=bomb)
    assert (q, tr) == ("红色背心", None)
    assert bomb.calls == []


def test_resolve_query_for_encoder_cnclip_never_lazy_loads(monkeypatch):
    # cnclip 默认路径也不得触碰懒加载翻译器
    called: list[str] = []
    monkeypatch.setattr(m03, "ZhEnTranslator", lambda: called.append("load"))
    q, tr = m03.resolve_query_for_encoder("红色背心", "cnclip")
    assert (q, tr) == ("红色背心", None)
    assert called == []


def test_resolve_query_for_encoder_cnclip_mixed_text_unchanged():
    q, tr = m03.resolve_query_for_encoder("白色帐篷 white tent", "cnclip")
    assert q == "白色帐篷 white tent"
    assert tr is None


def test_resolve_query_for_encoder_siglip2_default_no_translate():
    # 默认中文直查: 只过 ZH2EN 映射, 不触碰翻译器(评测: 短中文查询直查更优)
    bomb = _StubTranslator(error=RuntimeError("不应被调用"))
    q, tr = m03.resolve_query_for_encoder("红色背心", "siglip2", translator=bomb)
    assert (q, tr) == ("红色背心", None)
    assert bomb.calls == []


def test_resolve_query_for_encoder_siglip2_translate_flag():
    stub = _StubTranslator(out="red vest")
    q, tr = m03.resolve_query_for_encoder("红色背心", "siglip2", translator=stub, translate=True)
    assert (q, tr) == ("red vest", "red vest")
    assert stub.calls == ["红色背心"]


def test_resolve_query_for_encoder_openclip_translate_flag():
    stub = _StubTranslator(out="white tent")
    q, tr = m03.resolve_query_for_encoder("白色帐篷", "openclip", translator=stub, translate=True)
    assert (q, tr) == ("white tent", "white tent")
    assert stub.calls == ["白色帐篷"]


def test_resolve_query_for_encoder_siglip2_translator_none_zh2en():
    # --translate 且翻译器不可用: 回退 ZH2EN 兜底行为
    q, tr = m03.resolve_query_for_encoder("白色帐篷", "siglip2", translator=None, translate=True)
    assert tr is None
    assert "tent" in q.split()


# ---------- 02_embed_index.CnClipEncoder: 编码参数与输出归一化 (桩, 不下载模型) ----------

class _FakePoolingOutput:
    """模拟 transformers 5.x get_*_features 返回的 BaseModelOutputWithPooling."""

    def __init__(self, t):
        self.pooler_output = t


def test_cnclip_encode_texts_padding_truncation_and_norm():
    import torch

    calls: list[dict] = []

    class _FakeProcessor:
        def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
            calls.append({"text": list(text), "return_tensors": return_tensors, **kwargs})
            return {"input_ids": torch.ones(len(text), 8, dtype=torch.long)}

    class _FakeModel:
        def get_text_features(self, input_ids, **kwargs):
            return _FakePoolingOutput(torch.ones(input_ids.shape[0], 4))

    enc = object.__new__(m02.CnClipEncoder)
    enc.processor = _FakeProcessor()
    enc.model = _FakeModel()
    out = enc.encode_texts(["红色背心", "运动上装"])
    assert calls == [{
        "text": ["红色背心", "运动上装"], "return_tensors": "pt",
        "padding": True, "truncation": True,
    }]
    assert out.shape == (2, 4)
    for v in out:
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


def test_cnclip_encode_images_norm_from_pooling_output():
    import torch

    seen: list[int] = []

    class _FakeProcessor:
        def __call__(self, images=None, return_tensors=None, **kwargs):
            seen.append(len(images))
            return {"pixel_values": torch.ones(len(images), 3, 4, 4)}

    class _FakeModel:
        def get_image_features(self, pixel_values, **kwargs):
            return _FakePoolingOutput(torch.ones(pixel_values.shape[0], 4))

    enc = object.__new__(m02.CnClipEncoder)
    enc.processor = _FakeProcessor()
    enc.model = _FakeModel()
    imgs = [Image.new("RGB", (8, 8), (255, 0, 0)),
            Image.new("RGB", (8, 8), (0, 255, 0))]
    out = enc.encode_images(imgs)
    assert seen == [2]
    assert out.shape == (2, 4)
    for v in out:
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


# ---------- 04_eval 指标纯函数 (不碰模型/chroma) ----------

def test_ranked_video_ids_dedupes_preserving_order():
    metas = [{"video_id": "a"}, {"video_id": "a"}, {"video_id": "b"},
             {}, {"video_id": "c"}, {"video_id": "a"}]
    assert m04.ranked_video_ids(metas) == ["a", "b", "c"]


def test_hit_at_k_boundaries():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert m04.hit_at_k(ranked, ["e"], 5) is True
    assert m04.hit_at_k(ranked, ["e"], 4) is False
    assert m04.hit_at_k(ranked, ["f"], 10) is True   # k 超出排名长度
    assert m04.hit_at_k(ranked, ["z"], 10) is False
    assert m04.hit_at_k(ranked, ["a"], 0) is False
    assert m04.hit_at_k(ranked, [], 3) is False


def test_reciprocal_rank_first_hit_and_miss():
    assert m04.reciprocal_rank(["a"], ["a"]) == 1.0
    assert m04.reciprocal_rank(["x", "b", "a"], ["a", "b"]) == pytest.approx(0.5)
    assert m04.reciprocal_rank(["x", "y"], ["a"]) == 0.0
    assert m04.reciprocal_rank([], ["a"]) == 0.0


def test_evaluate_ranking_hits_and_rr():
    out = m04.evaluate_ranking(["x", "a"], ["a"], ks=(1, 2, 10))
    assert out["hits"] == {1: False, 2: True, 10: True}
    assert out["rr"] == pytest.approx(0.5)


class _EvalTranslatorStub:
    def translate(self, text: str) -> str:
        return "RED VEST"


def test_preprocess_translate_mode_passes_translator_and_flag():
    """translate 模式必须显式带 translate=True + 翻译器, 否则会静默退化成直查."""
    class _M3:
        def __init__(self):
            self.calls: dict | None = None
        def get_translator(self):
            return _EvalTranslatorStub()
        def resolve_query_for_encoder(self, raw, model, translator=None, translate=False):
            self.calls = {"translator": translator, "translate": translate}
            return "red vest", "RED VEST"

    stub_m3 = _M3()
    q, tr = m04.preprocess_for_mode(stub_m3, "红色背心", "siglip2", "translate")
    assert (q, tr) == ("red vest", "RED VEST")
    assert stub_m3.calls is not None
    assert isinstance(stub_m3.calls["translator"], _EvalTranslatorStub)
    assert stub_m3.calls["translate"] is True


def test_preprocess_translate_mode_raises_when_model_unavailable():
    """翻译器不可用时大声失败, 禁止静默回退造数."""
    class _M3:
        def get_translator(self):
            return None

    with pytest.raises(RuntimeError):
        m04.preprocess_for_mode(_M3(), "红色背心", "siglip2", "translate")


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
