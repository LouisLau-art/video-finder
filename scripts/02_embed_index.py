#!/usr/bin/env python3
"""02_embed_index.py — 把 manifest 里的帧 embedding 后写入本地 Chroma.

优先: transformers SigLIP2-Base CPU (google/siglip2-base-patch16-224)
回退1: open_clip ViT-B/32 (laion2b_s34b_b79k)
回退2: 离线 dummy 哈希向量(仅保证链路跑通, 检索质量无意义, 会大声警告)

batch 小、CPU-only. payload 含 video_id / time / frame_path.

用法:
    python scripts/02_embed_index.py
    python scripts/02_embed_index.py --manifest frames/manifest.jsonl --db chroma_db --batch-size 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIGLIP2_ID = "google/siglip2-base-patch16-224"
CNCLIP_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
OPENCLIP_MODEL = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"


# ---------- encoders ----------

def normalize_features(x):
    """get_*_features 返回值 → L2 归一化 tensor.

    transformers 5.x 的 get_image_features/get_text_features 可能返回
    BaseModelOutputWithPooling(CLIP 系: pooler_output 为投影后特征)。
    """
    import torch
    if not isinstance(x, torch.Tensor):
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(x, attr, None)
            if v is not None:
                x = v
                break
        else:
            x = x[0]
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)


class Siglip2Encoder:
    name = "siglip2"
    model_id = SIGLIP2_ID

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()
        self.dim = int(self.model.config.vision_config.hidden_size)
        print(f"[encoder] SigLIP2 loaded: {self.model_id}, dim={self.dim}")

    @staticmethod
    def _norm(x):
        return normalize_features(x)

    def encode_images(self, imgs: list[Image.Image]) -> np.ndarray:
        import torch
        inputs = self.processor(images=imgs, return_tensors="pt")
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = self._norm(feats)
        return feats.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        # SigLIP2 官方要求文本 padding="max_length"(64) —— 训练时用法: 所有文本补到
        # 定长后取最后一个 token(含 padding)的 hidden state 作表示; 不加 attention_mask
        # (官方文档示例即如此, 实测加不加结果不同)。定长 padding 同时保证批量与单条
        # 编码结果完全一致, 不受 batch 组成影响。
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        with torch.no_grad():
            feats = self._norm(self.model.get_text_features(**inputs))
        return feats.cpu().numpy().astype(np.float32)


class CnClipEncoder:
    """中文原生 CLIP (OFA-Sys/chinese-clip-vit-base-patch16).

    文本直接编码中文, 不需要翻译层; 与 SigLIP2/open_clip 的英文语义空间不通用,
    查询时必须配对使用同 encoder 建的索引。
    """

    name = "cnclip"
    model_id = CNCLIP_ID

    def __init__(self):
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        self.torch = torch
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_id)
        self.model = ChineseCLIPModel.from_pretrained(self.model_id)
        self.model.eval()
        self.dim = int(getattr(self.model.config, "projection_dim", 512) or 512)
        print(f"[encoder] CN-CLIP loaded: {self.model_id}, dim={self.dim}")

    def encode_images(self, imgs: list[Image.Image]) -> np.ndarray:
        import torch
        inputs = self.processor(images=imgs, return_tensors="pt")
        with torch.no_grad():
            feats = normalize_features(self.model.get_image_features(**inputs))
        return feats.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        # 中文原生: 文本直接吃中文(fine-tune 时即中文语料), padding/truncation 常规处理
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            feats = normalize_features(self.model.get_text_features(**inputs))
        return feats.cpu().numpy().astype(np.float32)


class OpenClipEncoder:
    name = "openclip"
    model_id = f"{OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}"

    def __init__(self):
        import torch
        import open_clip
        self.torch = torch
        self.open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
        self.dim = 512
        print(f"[encoder] open_clip loaded: {self.model_id}")

    def encode_images(self, imgs: list[Image.Image]) -> np.ndarray:
        import torch
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in imgs])
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return feats.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        tokens = self.tokenizer(texts)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return feats.cpu().numpy().astype(np.float32)


class DummyEncoder:
    """离线兜底: 图像按颜色直方图+哈希, 文本按哈希, 归一化512维.

    仅用于无网验证链路, 检索质量无意义!
    """

    name = "dummy"
    model_id = "dummy-hash-512"
    dim = 512

    def __init__(self):
        print("[encoder][WARN] 用 dummy 哈希向量(无网兜底), 检索分数无语义意义!")

    def _hash_vec(self, seed: bytes) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        h = seed
        i = 0
        while i < self.dim:
            h = hashlib.sha256(h).digest()
            for b in h:
                if i >= self.dim:
                    break
                v[i] = (b / 255.0) - 0.5
                i += 1
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

    def encode_images(self, imgs: list[Image.Image]) -> np.ndarray:
        vecs = []
        for im in imgs:
            small = im.convert("RGB").resize((16, 16))
            seed = np.asarray(small, dtype=np.uint8).tobytes()
            vecs.append(self._hash_vec(seed))
        return np.stack(vecs).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._hash_vec(t.encode("utf-8")) for t in texts]).astype(np.float32)


def build_encoder(prefer: str | None = None):
    """prefer: siglip2 | cnclip | openclip | dummy | None(按顺序自动).

    cnclip 只在显式指定时使用(中文原生, 与现有英文索引不通用), 不进自动回退链。
    """
    order = [prefer] if prefer else ["siglip2", "openclip", "dummy"]
    last_err: Exception | None = None
    for name in order:
        try:
            if name == "siglip2":
                return Siglip2Encoder()
            if name == "cnclip":
                return CnClipEncoder()
            if name == "openclip":
                return OpenClipEncoder()
            if name == "dummy":
                return DummyEncoder()
        except Exception as e:  # noqa: BLE001 — 原型: 下载失败就换下一个
            print(f"[encoder] {name} 加载失败，回退下一个: {e}")
            last_err = e
    raise RuntimeError(f"所有 encoder 都失败: {last_err}")


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="帧 embedding + 写入本地 Chroma (CPU-only)")
    ap.add_argument("--manifest", default="frames/manifest.jsonl")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="frames")
    ap.add_argument("--batch-size", type=int, default=4, help="CPU 小 batch，默认 4")
    ap.add_argument("--model", default=None, choices=["siglip2", "cnclip", "openclip", "dummy"],
                    help="强制指定 encoder，默认自动: siglip2 -> openclip -> dummy")
    ap.add_argument("--rebuild", action="store_true", help="重建 collection（删掉旧数据）")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists() or manifest.stat().st_size == 0:
        print(f"manifest 为空或不存在: {manifest}，先跑 01_extract.py")
        return 1
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("manifest 里 0 条记录，先跑 01_extract.py")
        return 1

    import chromadb

    encoder = build_encoder(args.model)
    client = chromadb.PersistentClient(path=args.db)
    if args.rebuild:
        try:
            client.delete_collection(args.collection)
            print(f"[chroma] 已删除旧 collection: {args.collection}")
        except Exception:
            pass
    col = client.get_or_create_collection(
        name=args.collection, metadata={"hnsw:space": "cosine"}
    )

    bs = max(int(args.batch_size), 1)
    n = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        imgs: list[Image.Image] = []
        keep: list[dict] = []
        for r in chunk:
            p = Path(r["frame_path"])
            if not p.exists():
                print(f"[warn] 帧文件不存在跳过: {p}")
                continue
            try:
                imgs.append(Image.open(p).convert("RGB"))
                keep.append(r)
            except Exception as e:
                print(f"[warn] 读图失败跳过 {p}: {e}")
        if not keep:
            continue
        embs = encoder.encode_images(imgs)
        ids = [f"{r['video_id']}@{r['time']:.1f}" for r in keep]
        metas = [
            {"video_id": r["video_id"], "time": float(r["time"]),
             "frame_path": str(r["frame_path"])}
            for r in keep
        ]
        docs = [f"{r['video_id']} @ {r['time']:.1f}s" for r in keep]
        col.upsert(ids=ids, embeddings=embs.tolist(), metadatas=metas, documents=docs)
        n += len(keep)
        print(f"[index] {n}/{len(rows)} ...")

    Path(args.db, "encoder.json").write_text(
        json.dumps({"encoder": encoder.name, "model_id": encoder.model_id,
                    "dim": int(encoder.dim)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {n} 帧 -> chroma {args.db} collection={args.collection} "
          f"(encoder={encoder.name} {encoder.model_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
