#!/usr/bin/env python3
"""03_search.py — 文本搜视频帧 (CLI).

用法:
    python scripts/03_search.py "white tent lawn" --topk 20
    python scripts/03_search.py "白色帐篷 草坪 黑人" --topk 10
    python scripts/03_search.py --help

查询预处理: 含 CJK 汉字([\u4e00-\u9fff])时优先用 Helsinki-NLP/opus-mt-zh-en
(懒加载, CPU)翻译成英文再编码；模型加载/翻译任何失败则回退现有 ZH2EN 映射，
保证无网也能用。纯英文查询零翻译开销。
encoder 与 02 保持一致(读 chroma_db/encoder.json，无文件则按 siglip2->openclip->dummy 顺序自动).
输出按相似度排序: 分数 + video_id + 秒 + 帧路径(截图).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Protocol, cast

# 中文翻译模型: Helsinki-NLP/opus-mt-zh-en (~300MB), 懒加载, device=-1(CPU)
TRANSLATE_MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 惰性单例状态: False=未尝试加载, True=已尝试
_TRANSLATOR_TRIED = False
_TRANSLATOR = None  # ZhEnTranslator | None
_UNSET = object()  # 区分“未指定(懒加载真实模型)”与“显式 None(不可用, 用于测试/离线)”

# 简单中英映射: 中文词 -> 英文，查询前做字符串替换
ZH2EN: dict[str, str] = {
    "帐篷": "tent",
    "草坪": "lawn",
    "草地": "lawn",
    "草原": "grassland",
    "白色": "white",
    "黑色": "black",
    "黑人": "black man",
    "白人": "white man",
    "男人": "man",
    "男子": "man",
    "女人": "woman",
    "女子": "woman",
    "小孩": "child",
    "孩子": "child",
    "狗": "dog",
    "猫": "cat",
    "车": "car",
    "汽车": "car",
    "房子": "house",
    "树": "tree",
    "花": "flower",
    "花苞": "flower bud",
    "水": "water",
    "天空": "sky",
    "夜晚": "night",
    "白天": "daytime",
    "室内": "indoor",
    "室外": "outdoor",
}


def map_query(q: str) -> str:
    out = f" {q} "
    # 长词优先：避免短词(如"车"/"花")先命中，把长词("汽车"/"花苞")切碎成中英混合垃圾词
    for zh, en in sorted(ZH2EN.items(), key=lambda kv: len(kv[0]), reverse=True):
        if zh in out:
            # 前后补空格，避免“白色帐篷”粘成 whitetent
            out = out.replace(zh, f" {en} ")
    # 挤掉多余空格
    return " ".join(out.split())


# ---------- 中文翻译层 (Helsinki-NLP/opus-mt-zh-en) ----------

def has_cjk(s: str) -> bool:
    """是否含 CJK 汉字([\\u4e00-\\u9fff])，只有含汉字才需要翻译."""
    return CJK_RE.search(s) is not None


class TranslatorLike(Protocol):
    """翻译器接口(便于测试注入桩)."""

    def translate(self, text: str) -> str: ...


class ZhEnTranslator:
    """Helsinki-NLP/opus-mt-zh-en 中->英翻译器(CPU).

    transformers 5.x 已移除 translation pipeline 任务，这里直接用
    AutoTokenizer + AutoModelForSeq2SeqLM 生成。
    """

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(TRANSLATE_MODEL_ID)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL_ID)
        self._model.eval()
        print(f"[translator] 已加载 {TRANSLATE_MODEL_ID} (CPU)")

    def translate(self, text: str) -> str:
        import torch
        enc = self._tok(text, return_tensors="pt")
        with torch.no_grad():
            # 查询很短, 64 token 上限足够; 用 max_length 避免与生成默认值冲突告警
            out = self._model.generate(**enc, max_length=64)
        return self._tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def get_translator() -> TranslatorLike | None:
    """懒加载翻译器单例; 加载失败返回 None 并缓存失败(无网不断链)."""
    global _TRANSLATOR_TRIED, _TRANSLATOR
    if not _TRANSLATOR_TRIED:
        _TRANSLATOR_TRIED = True
        try:
            _TRANSLATOR = ZhEnTranslator()
        except Exception as e:  # noqa: BLE001 — 原型: 无网/缺依赖时回退 ZH2EN
            print(
                f"[warn] 翻译模型加载失败({TRANSLATE_MODEL_ID})，回退 ZH2EN 映射: {e}",
                file=sys.stderr,
            )
            _TRANSLATOR = None
    return cast("TranslatorLike | None", _TRANSLATOR)


def resolve_query(
    raw: str, translator: TranslatorLike | None | object = _UNSET
) -> tuple[str, str | None]:
    """查询预处理: 含 CJK 时优先翻译→英文, 失败/不可用回退 ZH2EN 映射.

    translator 缺省 = 懒加载真实模型；显式传对象(测试桩)或 None(禁用翻译)便于测试。
    返回 (最终查询串, 翻译结果或 None)；纯英文查询零翻译开销直接返回。
    """
    if not has_cjk(raw):
        return map_query(raw), None  # 无汉字: 不碰翻译模型
    tr: TranslatorLike | None = (
        get_translator() if translator is _UNSET else cast("TranslatorLike | None", translator)
    )
    if tr is not None:
        try:
            en = tr.translate(raw)
            if en:
                return map_query(en), en
        except Exception as e:  # noqa: BLE001 — 翻译失败不断链
            print(f"[warn] 翻译失败，回退 ZH2EN 映射: {e}", file=sys.stderr)
    return map_query(raw), None


def load_encoder(db: str, prefer: str | None):
    """与 02 共用一套 encoder 类：复用 02 的实现，避免两边不一致."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "embed_index", str(Path(__file__).parent / "02_embed_index.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if prefer:
        return mod.build_encoder(prefer)
    meta = Path(db) / "encoder.json"
    if meta.exists():
        try:
            name = json.loads(meta.read_text(encoding="utf-8")).get("encoder")
            if name in ("siglip2", "openclip", "dummy"):
                print(f"[encoder] 按 encoder.json 用 {name}")
                return mod.build_encoder(name)
        except Exception as e:
            print(f"[warn] encoder.json 读取失败，自动选择: {e}")
    return mod.build_encoder(None)


def main() -> int:
    ap = argparse.ArgumentParser(description="文本搜视频帧: 输出排序+分数+帧路径")
    ap.add_argument("query", help='查询词，如 "white tent lawn black man"（支持简单中文，如"白色帐篷 草坪"）')
    ap.add_argument("--topk", type=int, default=20, help="返回前 K 个结果")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="frames")
    ap.add_argument("--model", default=None, choices=["siglip2", "openclip", "dummy"],
                    help="强制指定 encoder，默认读 encoder.json")
    args = ap.parse_args()

    import chromadb

    raw_q = args.query
    q, translated = resolve_query(raw_q)
    if translated:
        print(f"[query] 翻译(zh->en): {raw_q!r} -> {translated!r}")
        if q != translated:
            print(f"[query] 映射: {translated!r} -> {q!r}")
    elif q != raw_q:
        print(f"[query] 映射: {raw_q!r} -> {q!r}")
    else:
        print(f"[query] {q!r}")

    encoder = load_encoder(args.db, args.model)
    q_emb = encoder.encode_texts([q])

    client = chromadb.PersistentClient(path=args.db)
    try:
        col = client.get_collection(args.collection)
    except Exception:
        print(f"collection 不存在: {args.collection}，先跑 02_embed_index.py")
        return 1

    n = col.count()
    if n == 0:
        print("索引为空，先跑 02_embed_index.py")
        return 1
    k = max(1, min(int(args.topk), n))
    res = col.query(query_embeddings=q_emb.tolist(), n_results=k,
                    include=["metadatas", "documents", "distances"])

    metas = res["metadatas"][0]
    dists = res["distances"][0]
    docs = res["documents"][0] if res.get("documents") else [""] * len(metas)

    print(f"[result] top{k} (共 {n} 帧, encoder={encoder.name}):")
    print(f"{'rank':<5}{'score':<9}{'dist':<9}{'video_id':<22}{'time':<9}frame_path")
    for i, (m, d, doc) in enumerate(zip(metas, dists, docs), 1):
        # chroma cosine distance ∈ [0,2]，转相似度分数
        score = 1.0 - float(d) / 2.0
        vid = str(m.get("video_id", "?"))
        t = float(m.get("time", 0.0))
        fp = str(m.get("frame_path", ""))
        exists = "" if Path(fp).exists() else "  [文件缺失?]"
        print(f"{i:<5}{score:<9.4f}{float(d):<9.4f}{vid:<22}{t:<9.1f}{fp}{exists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
