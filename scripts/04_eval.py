#!/usr/bin/env python3
"""04_eval.py — 视频级检索评测 (离线/通用).

读 queries JSON（如 eval_local/queries.json，字段: frames_dir/manifest/topk/
queries[{zh,targets}]），对每个配置:
  1) subprocess 调 02_embed_index.py 建索引（与生产一致）
  2) 逐查询: 按配置预处理（siglip2 翻译 / siglip2 不翻译走 ZH2EN / cnclip 原样）
     -> 编码 -> chromadb 直接 query
  3) 帧级结果按相似度顺序去重成视频级排名
  4) 算视频级 hit@1/5/10 + MRR，打印表格并写 JSON 报告

配置（--configs，默认全跑）:
  siglip2+tr      siglip2 + opus-mt 翻译层（现有默认路径）
  siglip2+zh2en   siglip2 + 强制 translator=None -> ZH2EN 映射兜底
  cnclip+raw      CN-CLIP 中文原生 + 查询原样

用法:
    uv run python scripts/04_eval.py --queries eval_local/queries.json
    uv run python scripts/04_eval.py --configs cnclip+raw --skip-index
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# 标准指标截断点
KS: tuple[int, ...] = (1, 5, 10)

CONFIG_SPECS: dict[str, dict[str, str]] = {
    "siglip2+tr": {"model": "siglip2", "query_mode": "translate"},
    "siglip2+zh2en": {"model": "siglip2", "query_mode": "zh2en"},
    "cnclip+raw": {"model": "cnclip", "query_mode": "raw"},
}
DEFAULT_CONFIGS = list(CONFIG_SPECS)


# ---------- 纯函数: 排名聚合与指标 (便于离线测试) ----------

def ranked_video_ids(metadatas: list[dict]) -> list[str]:
    """帧级命中按相似度顺序(输入序)去重成视频级排名, 保持首次出现顺序."""
    seen: set[str] = set()
    out: list[str] = []
    for m in metadatas:
        vid = str(m.get("video_id", ""))
        if vid and vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


def hit_at_k(ranked: list[str], targets: list[str], k: int) -> bool:
    """top-k 视频里是否出现任一 target."""
    if k <= 0:
        return False
    tset = set(targets)
    return any(v in tset for v in ranked[:k])


def reciprocal_rank(ranked: list[str], targets: list[str]) -> float:
    """首个命中的视频排名的倒数; 未命中为 0.0."""
    tset = set(targets)
    for i, v in enumerate(ranked, 1):
        if v in tset:
            return 1.0 / i
    return 0.0


def evaluate_ranking(ranked: list[str], targets: list[str],
                     ks: tuple[int, ...] = KS) -> dict:
    """单查询指标: hit@k 与 RR."""
    return {
        "hits": {int(k): bool(hit_at_k(ranked, targets, k)) for k in ks},
        "rr": float(reciprocal_rank(ranked, targets)),
    }


# ---------- 运行时辅助 (会碰模型/chroma, 测试不调用) ----------

def load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_search_module():
    return load_module(Path(__file__).resolve().parent / "03_search.py", "search_cli")


def build_index(manifest: Path, db: Path, model: str, batch_size: int) -> float:
    """subprocess 调 02_embed_index.py 建索引（与生产一致）, 返回耗时秒."""
    script = Path(__file__).resolve().parent / "02_embed_index.py"
    cmd = [
        sys.executable, str(script),
        "--manifest", str(manifest),
        "--db", str(db),
        "--model", model,
        "--batch-size", str(batch_size),
        "--rebuild",
    ]
    print(f"[index] 建索引 model={model} -> {db} ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-20:])
        raise RuntimeError(f"建索引失败 model={model} (exit={proc.returncode}):\n{tail}")
    last = [l for l in proc.stdout.splitlines() if l.strip()]
    print(f"[index] model={model} 完成 {dt:.1f}s :: {last[-1] if last else ''}", flush=True)
    return dt


def preprocess_for_mode(m3, raw: str, model: str, mode: str) -> tuple[str, str | None]:
    """按配置预处理查询 (复用 03 的门控逻辑)."""
    if mode == "translate":
        return m3.resolve_query_for_encoder(raw, model)
    if mode == "zh2en":
        return m3.resolve_query_for_encoder(raw, model, translator=None)
    if mode == "raw":
        if model != "cnclip":
            raise ValueError(f"raw 查询模式仅适用于 cnclip, 得到 {model}")
        return raw, None
    raise ValueError(f"未知 query_mode: {mode}")

def run_config(spec: dict, queries: list[dict], db: Path, topk: int, m3) -> dict:
    """跑单个配置: 加载 encoder -> 逐查询 -> 指标. 返回报告片段."""
    import chromadb

    model = spec["model"]
    mode = spec["query_mode"]
    encoder = m3.load_encoder(str(db), model)

    client = chromadb.PersistentClient(path=str(db))
    col = client.get_collection("frames")
    n = col.count()
    if n == 0:
        raise RuntimeError(f"索引为空: {db}")
    # 查询条数取 topk 与最大 K 的较大者, 保证 hit@10 有数
    k = min(max(int(topk), max(KS)), n)

    per_query: list[dict] = []
    t0 = time.perf_counter()
    for q in queries:
        raw = q["zh"]
        targets = list(q["targets"])
        tq0 = time.perf_counter()
        resolved, translated = preprocess_for_mode(m3, raw, model, mode)
        q_emb = encoder.encode_texts([resolved])
        res = col.query(query_embeddings=q_emb.tolist(), n_results=k,
                        include=["metadatas", "distances"])
        metas = list(res["metadatas"][0])
        ranked = ranked_video_ids(metas)
        metrics = evaluate_ranking(ranked, targets)
        per_query.append({
            "zh": raw,
            "resolved": resolved,
            "translated": translated,
            "targets": targets,
            "top1_video": ranked[0] if ranked else None,
            "ranked_videos": ranked,
            "hits": metrics["hits"],
            "rr": metrics["rr"],
            "latency_s": time.perf_counter() - tq0,
        })
    total_s = time.perf_counter() - t0

    agg = {f"hit@{k}": sum(bool(p["hits"][k]) for p in per_query) / len(per_query)
           for k in KS}
    agg["mrr"] = sum(p["rr"] for p in per_query) / len(per_query)
    return {
        "model": model,
        "query_mode": mode,
        "encoder_name": encoder.name,
        "encoder_dim": int(encoder.dim),
        "topk": k,
        "num_queries": len(per_query),
        "metrics": agg,
        "timing": {"query_total_s": total_s,
                   "query_avg_s": total_s / len(per_query) if per_query else 0.0},
        "per_query": per_query,
    }


def print_summary(report: dict) -> None:
    cfgs = report["configs"]
    print()
    print("=" * 72)
    print(f"{'config':<16}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'MRR':>8}{'avg_ms':>9}")
    print("-" * 72)
    for name, c in cfgs.items():
        m = c["metrics"]
        print(f"{name:<16}{m['hit@1']:>8.3f}{m['hit@5']:>8.3f}"
              f"{m['hit@10']:>8.3f}{m['mrr']:>8.3f}"
              f"{c['timing']['query_avg_s'] * 1000:>9.0f}")
    print("=" * 72)

    print()
    print("每查询 top1 视频:")
    header = f"{'query':<14}" + "".join(f"{n:<30}" for n in cfgs)
    print(header)
    print("-" * len(header))
    nq = len(next(iter(cfgs.values()))["per_query"]) if cfgs else 0
    for i in range(nq):
        q = next(iter(cfgs.values()))["per_query"][i]["zh"]
        row = f"{q:<14}"
        for c in cfgs.values():
            p = c["per_query"][i]
            mark = "✓" if p["hits"][1] else "✗"
            row += f"{mark} {str(p['top1_video']):<28}"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="视频级检索评测: 多配置对比 (hit@k/MRR)")
    ap.add_argument("--queries", default="eval_local/queries.json",
                    help="评测查询 JSON 路径")
    ap.add_argument("--db-root", default="/tmp/vf_eval_chroma",
                    help="每个模型建索引的根目录 (默认 /tmp/vf_eval_chroma)")
    ap.add_argument("--out", default="eval_local/report.json", help="JSON 报告输出路径")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help=f"逗号分隔, 可选: {', '.join(CONFIG_SPECS)}")
    ap.add_argument("--topk", type=int, default=None,
                    help="查询条数; 缺省用 queries 文件里的 topk (默认 10)")
    ap.add_argument("--batch-size", type=int, default=4, help="建索引 batch (透传 02)")
    ap.add_argument("--skip-index", action="store_true",
                    help="跳过建索引, 复用 db-root 下已有索引")
    args = ap.parse_args()

    spec_path = Path(args.queries)
    if not spec_path.exists():
        print(f"queries 文件不存在: {spec_path}")
        return 1
    qspec = json.loads(spec_path.read_text(encoding="utf-8"))
    queries = qspec.get("queries") or []
    if not queries:
        print(f"queries 为空: {spec_path}")
        return 1
    manifest = Path(qspec.get("manifest", ""))
    if not args.skip_index and (not manifest.exists() or manifest.stat().st_size == 0):
        print(f"manifest 不存在或为空: {manifest}")
        return 1
    topk = int(args.topk) if args.topk is not None else int(qspec.get("topk", 10))
    db_root = Path(args.db_root)
    db_root.mkdir(parents=True, exist_ok=True)

    cfg_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in cfg_names if c not in CONFIG_SPECS]
    if unknown:
        print(f"未知配置: {unknown}; 可选: {list(CONFIG_SPECS)}")
        return 1

    m3 = load_search_module()
    print(f"[eval] queries={spec_path} 共 {len(queries)} 条, topk={topk}, "
          f"manifest={manifest}, db_root={db_root}")
    print(f"[eval] 配置: {', '.join(cfg_names)}")

    report: dict = {
        "queries_file": str(spec_path),
        "manifest": str(manifest),
        "frames_dir": qspec.get("frames_dir"),
        "topk": topk,
        "ks": list(KS),
        "configs": {},
        "index_build_s": {},
    }
    built: dict[str, float] = {}
    for name in cfg_names:
        spec = CONFIG_SPECS[name]
        model = spec["model"]
        db = db_root / model
        if args.skip_index:
            if not (db / "encoder.json").exists():
                print(f"[index] 跳过建索引但索引不存在: {db}")
                return 1
            print(f"[index] 复用已有索引: {db}")
        elif model not in built:
            # 同一模型的多个配置(如 siglip2+tr / siglip2+zh2en)只建一次索引
            built[model] = build_index(manifest, db, model, args.batch_size)
            report["index_build_s"][model] = built[model]
        report["configs"][name] = run_config(spec, queries, db, topk, m3)
        print(f"[eval] {name}: {report['configs'][name]['metrics']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"[done] 报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
