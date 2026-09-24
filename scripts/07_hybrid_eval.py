#!/usr/bin/env python3
"""工单 04：视频级语义通道、关键词通道与融合评测。

专有名称样本由向量库元数据自验证生成，默认写入 gitignored 的 eval_local/；
报告只保存聚合指标，不保存真实素材名。`--failure-shapes-only` 运行开启关键词通道的失败形状门禁，`--failure-shapes-no-keyword` 运行应当失败的负对照。

退出码：0=回归通过，1=脚本错误，2=回归不通过，3=无法判定。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "scripts/05_api.py"
DEFAULT_EVAL_DIR = ROOT / "eval_local"
HASH_NAME_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)

EXIT_OK = 0
EXIT_SCRIPT_ERROR = 1
EXIT_REGRESSION_FAILED = 2
EXIT_REGRESSION_UNKNOWN = 3


class _ArgumentParser(argparse.ArgumentParser):
    """把命令行参数错误归入脚本错误退出码。"""

    def error(self, message: str):  # pragma: no cover - argparse 专用分支
        self.print_usage(sys.stderr)
        self.exit(EXIT_SCRIPT_ERROR, f"{self.prog}: error: {message}\n")


def _load_api_module(name: str):
    """动态加载带数字前缀的 API 脚本。"""
    spec = importlib.util.spec_from_file_location(name, str(API_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 05_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_gitignored(path: Path) -> None:
    """拒绝把含真实名称的评测输出写到可提交路径。"""
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(path)],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"评测输出路径未被 gitignore 拒绝: {path}")


def _write_json(path: Path, payload: Any) -> None:
    _require_gitignored(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _query_digest(query: str) -> str:
    return hashlib.sha256(str(query).encode("utf-8")).hexdigest()[:12]


def _configure_api(api: Any, db: str, collection: str) -> None:
    api.APP_DB = str(db)
    api.APP_COLLECTION = str(collection)
    api.APP_FRAMES_DIRS = api.resolve_frames_dirs(None)
    api.APP_MODEL = None
    api.APP_TRANSLATE = False


def _encode_query(api: Any, encoder: Any, m3: Any, query: str):
    resolved, _translated = m3.resolve_query_for_encoder(
        query,
        api._ENCODER_NAME,
        translate=api.APP_TRANSLATE,
    )
    return encoder.encode_texts([resolved])[0]


def _ranked_ids(payload: dict[str, Any]) -> list[str]:
    return [str(row["video_id"]) for row in payload["data"]["results"]]


def _run_semantic(api: Any, vec: Any, query: str, top_k: int) -> list[str]:
    payload = api.search_by_embedding(vec, top_k, query, keyword_query=None)
    return _ranked_ids(payload)


def _run_fused(api: Any, vec: Any, query: str, top_k: int) -> list[str]:
    old_enabled = api.KEYWORD_CHANNEL_ENABLED
    api.KEYWORD_CHANNEL_ENABLED = True
    try:
        payload = api.search_by_embedding(
            vec, top_k, query, keyword_query=query
        )
    finally:
        api.KEYWORD_CHANNEL_ENABLED = old_enabled
    return _ranked_ids(payload)


def _metric_summary(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    top_k: int,
) -> dict[str, Any]:
    """按视频级目标计算语义通道、关键词通道与融合结果的命中率和目标独占数。"""
    semantic_hits = 0
    keyword_hits = 0
    fused_hits = 0
    keyword_only = 0
    semantic_only = 0
    both = 0
    for case in cases:
        targets = {str(value) for value in case.get("targets", [])}
        if not targets:
            continue
        semantic = set(rankings[case["id"]]["semantic"][:top_k])
        keyword = set(rankings[case["id"]]["keyword"][:top_k])
        fused = set(rankings[case["id"]]["fused"][:top_k])
        sh = bool(targets & semantic)
        kh = bool(targets & keyword)
        fh = bool(targets & fused)
        semantic_hits += int(sh)
        keyword_hits += int(kh)
        fused_hits += int(fh)
        keyword_only += int(kh and not sh)
        semantic_only += int(sh and not kh)
        both += int(sh and kh)
    total = len(cases)
    rate = lambda count: round(count / total, 6) if total else 0.0
    return {
        "sample_count": total,
        "semantic_hits": semantic_hits,
        "keyword_hits": keyword_hits,
        "fused_hits": fused_hits,
        "semantic_hit_rate": rate(semantic_hits),
        "keyword_hit_rate": rate(keyword_hits),
        "fused_hit_rate": rate(fused_hits),
        "keyword_only_target_hits": keyword_only,
        "semantic_only_target_hits": semantic_only,
        "both_target_hits": both,
    }


def _generic_injection_summary(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    top_k: int,
) -> dict[str, Any]:
    """统计通用词关键词 Top20 中、语义 Top20 没有的注入素材。"""
    keyword_occurrences = 0
    injected_occurrences = 0
    injected_unique: set[str] = set()
    query_count = 0
    for case in cases:
        if not case.get("generic"):
            continue
        query_count += 1
        semantic = set(rankings[case["id"]]["semantic"][:top_k])
        keyword = rankings[case["id"]]["keyword"][:top_k]
        keyword_set = set(keyword)
        injected = keyword_set - semantic
        keyword_occurrences += len(keyword_set)
        injected_occurrences += len(injected)
        injected_unique.update(injected)
    ratio = (
        round(injected_occurrences / keyword_occurrences, 6)
        if keyword_occurrences
        else 0.0
    )
    return {
        "query_count": query_count,
        "keyword_topk_material_occurrences": keyword_occurrences,
        "injected_material_occurrences": injected_occurrences,
        "injected_unique_materials": len(injected_unique),
        "injection_ratio": ratio,
        "interpretation": "关键词 Top20 注入候选比例，不等价于真实误命中率",
    }


def _build_proprietary_cases(
    index: dict[str, dict[str, Any]],
    filename_limit: int,
    parent_limit: int,
    generic_limit: int,
) -> list[dict[str, Any]]:
    """从可匹配文本构造自验证样本，不把真实文本写入版本库。"""
    filename_cases: list[dict[str, Any]] = []
    for video_id in sorted(index):
        entry = index[video_id]
        filename = str(entry.get("filename") or "")
        if not filename:
            continue
        # 文件名样本不能同时出现在该素材的有意义父目录中。
        if any(filename in str(parent) for parent in entry.get("parents", ())):
            continue
        filename_cases.append({
            "id": f"filename-{len(filename_cases):04d}",
            "query": filename,
            "targets": [video_id],
            "source": "filename",
            "generic": False,
            "construction": "extracted_filename_text",
        })
        if len(filename_cases) >= filename_limit:
            break

    parent_frequency = Counter(
        str(parent)
        for entry in index.values()
        for parent in entry.get("parents", ())
    )
    parent_candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for video_id in sorted(index):
        entry = index[video_id]
        filename = str(entry.get("filename") or "")
        for parent in entry.get("parents", ()):
            parent = str(parent)
            key = (video_id, parent)
            if not parent or parent in filename or key in seen:
                continue
            seen.add(key)
            parent_candidates.append({
                "id": "",
                "query": parent,
                "targets": [video_id],
                "source": "parent",
                "generic": parent_frequency[parent] >= 5,
                "construction": "extracted_parent_text",
                "frequency": parent_frequency[parent],
            })

    generic = sorted(
        (item for item in parent_candidates if item["generic"]),
        key=lambda item: (-item["frequency"], item["query"], item["targets"][0]),
    )[:generic_limit]
    selected_parent = generic + [
        item for item in sorted(
            (item for item in parent_candidates if not item["generic"]),
            key=lambda item: (item["query"], item["targets"][0]),
        )
        if (item["targets"][0], item["query"]) not in {
            (selected["targets"][0], selected["query"]) for selected in generic
        }
    ]
    selected_parent = selected_parent[:parent_limit]
    for number, item in enumerate(selected_parent):
        item["id"] = f"parent-{number:04d}"
        item.pop("frequency", None)

    cases = filename_cases + selected_parent
    return cases


def _coverage_ledger(index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """统计自然文本覆盖率与包含类型化精确匹配的能力上限。"""
    counts = Counter()
    for entry in index.values():
        filename = str(entry.get("filename") or "")
        typed = bool(entry.get("typed"))
        hash_name = bool(HASH_NAME_RE.fullmatch(filename))
        readable_filename = (not typed) and any(ch.isalpha() for ch in filename)
        has_parent = bool(entry.get("parents"))
        natural = readable_filename or has_parent
        typed_exact = typed and not hash_name
        exact_capable = natural or typed_exact
        counts["total"] += 1
        counts["readable_filename"] += int(readable_filename)
        counts["typed_filename"] += int(typed)
        counts["typed_exact_filename"] += int(typed_exact)
        counts["hash_filename"] += int(hash_name)
        counts["meaningful_parent"] += int(has_parent)
        counts["natural_text_coverable"] += int(natural)
        counts["exact_capable_including_typed"] += int(exact_capable)
    total = counts["total"]
    ratio = lambda key: round(counts[key] / total, 6) if total else 0.0
    return {
        "counts": dict(sorted(counts.items())),
        "readable_filename_rate": ratio("readable_filename"),
        "typed_filename_rate": ratio("typed_filename"),
        "typed_exact_filename_rate": ratio("typed_exact_filename"),
        "hash_filename_rate": ratio("hash_filename"),
        "meaningful_parent_rate": ratio("meaningful_parent"),
        "natural_text_coverage_rate": ratio("natural_text_coverable"),
        "exact_capability_upper_bound_rate": ratio("exact_capable_including_typed"),
        "note": "类型化文件名只支持完整精确匹配；哈希名且无有意义父目录时无自然关键词覆盖，也不计入能力上限",
    }


def _load_metadata(api: Any, col: Any) -> list[dict[str, Any]]:
    payload = col.get(include=["metadatas"])
    return [dict(item) for item in (payload.get("metadatas") or []) if isinstance(item, dict)]


def _metadata_digest(metas: list[dict[str, Any]]) -> str:
    """对规范化后的元数据做全量 SHA-256 摘要，不保存原始名称。"""
    canonical_rows = sorted(
        json.dumps(
            meta,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        for meta in metas
    )
    digest = hashlib.sha256()
    for row in canonical_rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _index_signature(
    db: str,
    collection: str,
    count: int,
    encoder_name: str,
    metas: list[dict[str, Any]],
) -> str:
    """把索引身份与元数据内容摘要组合成可比较签名。"""
    return "|".join(
        (
            f"db={Path(db).resolve()}",
            f"collection={collection}",
            f"count={int(count)}",
            f"encoder={encoder_name}",
            f"metadata_sha256={_metadata_digest(metas)}",
        )
    )


def _signature_method() -> dict[str, Any]:
    return {
        "algorithm": "SHA-256",
        "scope": "all_collection_metadata_canonicalized_and_sorted",
        "includes_embeddings": False,
        "rationale": "检测元数据内容替换，不把真实名称写入产物",
        "risk": "元数据完全相同但向量被替换时无法检测；SHA-256 碰撞风险可忽略",
    }


def run_proprietary(
    api: Any,
    db: str,
    collection: str,
    top_k: int,
    filename_limit: int,
    parent_limit: int,
    generic_limit: int,
    cases_path: Path,
) -> dict[str, Any]:
    _configure_api(api, db, collection)
    api.KEYWORD_CHANNEL_ENABLED = True
    encoder, col = api.ensure_state()
    m3 = api.load_search_module()
    metas = _load_metadata(api, col)
    index = api._build_keyword_index(metas)
    cases = _build_proprietary_cases(index, filename_limit, parent_limit, generic_limit)
    _write_json(cases_path, {
        "version": 1,
        "source": "self_validated_from_vector_metadata",
        "contains_real_names": True,
        "gitignored": True,
        "optimistic_upper_bound": True,
        "cases": cases,
    })

    rankings: dict[str, list[str]] = {}
    for case in cases:
        vec = _encode_query(api, encoder, m3, case["query"])
        keyword_index = api._ensure_keyword_index(col, int(col.count()))
        keyword_ids = [
            video_id for video_id, _matched in api.keyword_search(
                case["query"], keyword_index
            )
        ]
        rankings[case["id"]] = {
            "semantic": _run_semantic(api, vec, case["query"], top_k),
            "keyword": keyword_ids,
            "fused": _run_fused(api, vec, case["query"], top_k),
        }

    summary = _metric_summary(cases, rankings, top_k)
    summary["by_source"] = {
        source: _metric_summary(
            [case for case in cases if case["source"] == source], rankings, top_k
        )
        for source in sorted({case["source"] for case in cases})
    }
    summary["source_counts"] = dict(sorted(Counter(c["source"] for c in cases).items()))
    summary["generic_query_count"] = sum(bool(c.get("generic")) for c in cases)
    summary["generic_injection"] = _generic_injection_summary(cases, rankings, top_k)
    summary["coverage_ledger"] = _coverage_ledger(index)
    summary["index_signature"] = _index_signature(
        db, collection, len(metas), api._ENCODER_NAME, metas
    )
    summary["index_signature_method"] = _signature_method()
    summary["query_provenance"] = {
        "kind": "self_validated_extracted_text",
        "optimistic_upper_bound": True,
        "note": "查询词直接从可匹配文本抽取，不是独立用户输入；命中率只能作为乐观上界",
    }
    return summary


def _ranking_metrics(ranked: list[str], targets: list[str], ks: tuple[int, ...]) -> dict[str, Any]:
    target_set = set(targets)
    hits = {k: False for k in ks}
    reciprocal_rank = 0.0
    for rank, video_id in enumerate(ranked, 1):
        if video_id in target_set:
            reciprocal_rank = 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] = True
            break
    return {"hits": hits, "mrr": reciprocal_rank}


def run_semantic_regression(
    api: Any,
    db: str,
    collection: str,
    eval_dir: Path,
    baseline_snapshot: Path | None,
    write_baseline: bool,
) -> dict[str, Any]:
    _configure_api(api, db, collection)
    encoder, col = api.ensure_state()
    m3 = api.load_search_module()
    metas = _load_metadata(api, col)
    signature = _index_signature(
        db, collection, len(metas), api._ENCODER_NAME, metas
    )
    output: dict[str, Any] = {
        "signature": signature,
        "index_signature_method": _signature_method(),
        "query_sets": {},
    }
    snapshot = None
    if baseline_snapshot is not None and baseline_snapshot.exists():
        try:
            snapshot = json.loads(baseline_snapshot.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            snapshot = None
    snapshot_comparable = bool(
        snapshot and snapshot.get("eval_signature") == signature
    )
    for filename in ("queries.json", "queries_long.json"):
        path = eval_dir / filename
        if not path.exists():
            continue
        spec = json.loads(path.read_text(encoding="utf-8"))
        cases = [
            {
                "id": str(index),
                "query": str(item.get("zh") or ""),
                "targets": [str(value) for value in item.get("targets", [])],
            }
            for index, item in enumerate(spec.get("queries", []))
        ]
        top_k = int(spec.get("topk", 10))
        metrics = {"hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr": 0.0}
        for case in cases:
            if not case["query"]:
                continue
            vec = _encode_query(api, encoder, m3, case["query"])
            ranked = _run_semantic(api, vec, case["query"], top_k)
            one = _ranking_metrics(ranked, case["targets"], (1, 5, 10))
            for key in ("hit@1", "hit@5", "hit@10"):
                metrics[key] += int(one["hits"][int(key.split("@")[1])])
            metrics["mrr"] += one["mrr"]
        total = len(cases)
        if total:
            for key in ("hit@1", "hit@5", "hit@10"):
                metrics[key] = round(metrics[key] / total, 6)
            metrics["mrr"] = round(metrics["mrr"] / total, 6)
        current = {
            "sample_count": total,
            "metrics": metrics,
            "queries_file": str(path),
        }
        legacy_path = eval_dir / ("report.json" if filename == "queries.json" else "report_long.json")
        legacy_reference = None
        comparable = False
        if legacy_path.exists():
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            legacy_cfg = legacy.get("configs", {}).get("cnclip+raw", {})
            legacy_metrics = legacy_cfg.get("metrics")
            if legacy_metrics:
                legacy_reference = {
                    "metrics": legacy_metrics,
                    "has_eval_signature": bool(legacy.get("eval_signature")),
                }
                comparable = legacy.get("eval_signature") == signature
        current["legacy_reference"] = legacy_reference
        current["legacy_baseline_comparable"] = comparable
        current["baseline_snapshot_reference"] = (
            snapshot.get("query_sets", {}).get(filename)
            if snapshot_comparable else None
        )
        current["baseline_comparable"] = snapshot_comparable
        current["baseline_note"] = (
            "同签名快照可比较"
            if snapshot_comparable
            else "没有同索引签名的快照；旧报告差值不能作为严格回归结论"
        )
        if legacy_reference:
            current["delta_vs_legacy_reference"] = {
                key: round(metrics[key] - legacy_reference["metrics"].get(key, metrics[key]), 6)
                for key in ("hit@1", "hit@5", "hit@10", "mrr")
            }
        snapshot_reference = current["baseline_snapshot_reference"]
        if snapshot_reference:
            current["delta_vs_baseline_snapshot"] = {
                key: round(metrics[key] - snapshot_reference.get(key, metrics[key]), 6)
                for key in ("hit@1", "hit@5", "hit@10", "mrr")
            }
            current["regression_pass"] = bool(
                metrics["hit@10"] >= snapshot_reference.get("hit@10", 0)
                and metrics["mrr"] >= snapshot_reference.get("mrr", 0)
            )
        else:
            current["regression_pass"] = None
        output["query_sets"][filename] = current

    if baseline_snapshot is not None and write_baseline:
        _write_json(baseline_snapshot, {
            "version": 1,
            "eval_signature": signature,
            "query_sets": {
                name: value.get("metrics")
                for name, value in output["query_sets"].items()
            },
        })
    return output


class _FixtureCollection:
    """为失败形状回归提供确定性的内存向量库。"""

    def __init__(self, fixture: dict[str, Any]):
        self.frames = [dict(frame) for frame in fixture.get("frames", [])]
        self.order = [str(value) for value in fixture.get("semantic_order", [])]
        self.by_id = {str(frame["video_id"]): frame for frame in self.frames}

    def count(self) -> int:
        return len(self.frames)

    def get(self, include=None):
        if include != ["metadatas"]:
            raise AssertionError("失败形状夹具只允许读取元数据")
        return {"metadatas": [dict(frame) for frame in self.frames]}

    def query(self, *, query_embeddings, n_results, include):
        if include != ["metadatas", "distances"]:
            raise AssertionError("失败形状夹具只允许读取元数据和距离")
        selected = self.order[: max(0, int(n_results))]
        metas = [dict(self.by_id[video_id]) for video_id in selected]
        distances = [index / 100.0 for index, _ in enumerate(selected)]
        return {
            "ids": [[f"fixture-frame-{index}" for index in range(len(selected))]],
            "metadatas": [metas],
            "distances": [distances],
        }


class _FixtureEncoder:
    name = "fixture-encoder"

    def encode_texts(self, texts):
        return [[0.0, 1.0] for _ in texts]


def _load_failure_shape_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("cases"), list) or not isinstance(payload.get("fixtures"), list):
        raise ValueError("失败形状夹具缺少 cases/fixtures")
    return payload


def run_failure_shape_gate(
    api: Any,
    fixture_path: Path,
    keyword_enabled: bool,
) -> dict[str, Any]:
    """运行脱敏失败形状；关闭关键词通道时故意返回回归失败。"""
    payload = _load_failure_shape_fixture(fixture_path)
    fixtures = {str(item["id"]): item for item in payload["fixtures"]}
    saved = {
        "encoder": api._ENCODER,
        "encoder_name": api._ENCODER_NAME,
        "collection": api._CHROMA_COL,
        "count": api._CHROMA_COUNT,
        "keyword_index": api._KEYWORD_INDEX,
        "keyword_index_count": api._KEYWORD_INDEX_COUNT,
        "keyword_enabled": api.KEYWORD_CHANNEL_ENABLED,
    }
    rows: list[dict[str, Any]] = []
    try:
        for case in payload["cases"]:
            fixture = fixtures.get(str(case["fixture_id"]))
            if fixture is None:
                raise ValueError(f"找不到失败形状夹具: {case.get('fixture_id')}")
            collection = _FixtureCollection(fixture)
            api._ENCODER = _FixtureEncoder()
            api._ENCODER_NAME = api._ENCODER.name
            api._CHROMA_COL = collection
            api._CHROMA_COUNT = collection.count()
            api._KEYWORD_INDEX = None
            api._KEYWORD_INDEX_COUNT = None
            api.KEYWORD_CHANNEL_ENABLED = keyword_enabled
            top_k = int(fixture.get("top_k", 3))
            vector = api._ENCODER.encode_texts([str(case["query"])])[0]
            result = api.search_by_embedding(
                vector,
                top_k,
                str(case["query"]),
                keyword_query=str(case["query"]),
            )
            result_ids = [str(row["video_id"]) for row in result["data"]["results"]]
            actual_hit = str(case["expected_target_video_id"]) in result_ids
            expected_hit = bool(keyword_enabled)
            rows.append({
                "id": str(case["id"]),
                "failure_shape": str(case["failure_shape"]),
                "expected_hit": expected_hit,
                "actual_hit": actual_hit,
                "passed": actual_hit == expected_hit,
            })
    finally:
        api._ENCODER = saved["encoder"]
        api._ENCODER_NAME = saved["encoder_name"]
        api._CHROMA_COL = saved["collection"]
        api._CHROMA_COUNT = saved["count"]
        api._KEYWORD_INDEX = saved["keyword_index"]
        api._KEYWORD_INDEX_COUNT = saved["keyword_index_count"]
        api.KEYWORD_CHANNEL_ENABLED = saved["keyword_enabled"]

    all_passed = all(row["passed"] for row in rows)
    all_missed = all(not row["actual_hit"] for row in rows)
    if keyword_enabled:
        status = "passed" if all_passed else "failed"
        code = EXIT_OK if all_passed else EXIT_REGRESSION_FAILED
    elif all_missed:
        # 负对照达到预期，但目标召回门禁应当失败。
        status = "negative_control_passed_gate_failed"
        code = EXIT_REGRESSION_FAILED
    else:
        status = "invalid_negative_control"
        code = EXIT_SCRIPT_ERROR
    return {
        "case_count": len(rows),
        "keyword_enabled": keyword_enabled,
        "all_expected_hits": all(row["actual_hit"] for row in rows),
        "all_expected_misses": all_missed,
        "status": status,
        "exit_code": code,
        "cases": rows,
    }


def _regression_gate(query_sets: dict[str, Any]) -> tuple[str, int]:
    """把各评测集的回归状态汇总为可区分的退出码。"""
    states = [item.get("regression_pass") for item in query_sets.values()]
    if not states:
        return "not_run", EXIT_REGRESSION_UNKNOWN
    if any(state is False for state in states):
        return "failed", EXIT_REGRESSION_FAILED
    if any(state is None for state in states):
        return "indeterminate", EXIT_REGRESSION_UNKNOWN
    return "passed", EXIT_OK


def _combine_gate(
    query_sets: dict[str, Any],
    failure_shape_report: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """合并语义回归门禁与失败形状护栏，脚本错误优先返回 1。"""
    semantic_state, semantic_code = _regression_gate(query_sets)
    shape_code = int((failure_shape_report or {}).get("exit_code", EXIT_OK))
    if shape_code == EXIT_SCRIPT_ERROR:
        return "script_error", EXIT_SCRIPT_ERROR
    if shape_code == EXIT_REGRESSION_FAILED:
        return "failed", EXIT_REGRESSION_FAILED
    return semantic_state, semantic_code


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="视频级语义通道、关键词通道与融合评测")
    parser.add_argument("--db", default="data/local-runtime/index_full/cnclip")
    parser.add_argument("--collection", default="frames")
    parser.add_argument("--semantic-db", default="data/local-runtime/eval_chroma/cnclip")
    parser.add_argument("--eval-dir", default="eval_local")
    parser.add_argument("--cases-out", default="eval_local/proprietary_queries.json")
    parser.add_argument("--report-out", default="eval_local/hybrid_report.json")
    parser.add_argument(
        "--failure-shape-fixture",
        default="eval_fixtures/real_failure_shapes.json",
    )
    parser.add_argument(
        "--failure-report-out",
        default="eval_local/failure_shape_report.json",
    )
    parser.add_argument("--failure-shapes-only", action="store_true")
    parser.add_argument("--failure-shapes-no-keyword", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--filename-cases", type=int, default=30)
    parser.add_argument("--parent-cases", type=int, default=30)
    parser.add_argument("--generic-cases", type=int, default=10)
    parser.add_argument("--baseline-snapshot", default="eval_local/semantic_baseline.json")
    parser.add_argument(
        "--gate-report",
        help="只读取已有评测报告并按回归状态返回退出码",
    )
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--skip-proprietary", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gate_report:
        gate_report = json.loads(Path(args.gate_report).read_text(encoding="utf-8"))
        query_sets = gate_report.get("semantic_regression", {}).get("query_sets", {})
        shape_report = gate_report.get("failure_shapes")
        state, code = _combine_gate(query_sets, shape_report)
        print(json.dumps({
            "gate_report": str(args.gate_report),
            "regression_status": state,
            "exit_code": code,
        }, ensure_ascii=False))
        return code

    if args.failure_shapes_only or args.failure_shapes_no_keyword:
        if args.failure_shapes_only and args.failure_shapes_no_keyword:
            raise ValueError("失败形状开关模式不能同时开启")
        api = _load_api_module("hybrid_eval_api_failure_shapes")
        shape_report = run_failure_shape_gate(
            api,
            Path(args.failure_shape_fixture),
            keyword_enabled=not args.failure_shapes_no_keyword,
        )
        _write_json(Path(args.failure_report_out), shape_report)
        print(json.dumps({
            "failure_shape_report": str(args.failure_report_out),
            "status": shape_report["status"],
            "case_count": shape_report["case_count"],
            "exit_code": shape_report["exit_code"],
        }, ensure_ascii=False))
        return int(shape_report["exit_code"])

    eval_dir = Path(args.eval_dir)
    report: dict[str, Any] = {
        "version": 1,
        "tool": "07_hybrid_eval.py",
        "contains_real_names": False,
    }
    shape_api = _load_api_module("hybrid_eval_api_failure_shapes")
    report["failure_shapes"] = run_failure_shape_gate(
        shape_api,
        Path(args.failure_shape_fixture),
        keyword_enabled=True,
    )
    if not args.skip_proprietary:
        api = _load_api_module("hybrid_eval_api_proprietary")
        report["proprietary"] = run_proprietary(
            api,
            args.db,
            args.collection,
            args.top_k,
            args.filename_cases,
            args.parent_cases,
            args.generic_cases,
            Path(args.cases_out),
        )
    if not args.skip_semantic:
        api = _load_api_module("hybrid_eval_api_semantic")
        report["semantic_regression"] = run_semantic_regression(
            api,
            args.semantic_db,
            args.collection,
            eval_dir,
            Path(args.baseline_snapshot) if args.baseline_snapshot else None,
            args.write_baseline,
        )
    state, code = _combine_gate(
        report.get("semantic_regression", {}).get("query_sets", {}),
        report.get("failure_shapes"),
    )
    report["regression_status"] = state
    report["regression_exit_code"] = code
    if "failure_shapes" in report:
        report["failure_shape_status"] = report["failure_shapes"]["status"]
        report["failure_shape_exit_code"] = report["failure_shapes"]["exit_code"]
    _write_json(Path(args.report_out), report)
    print(json.dumps({
        "report": str(args.report_out),
        "proprietary_sample_count": report.get("proprietary", {}).get("sample_count"),
        "semantic_query_sets": sorted(report.get("semantic_regression", {}).get("query_sets", {})),
        "failure_shape_status": report.get("failure_shape_status", "skipped"),
        "regression_status": state,
        "exit_code": code,
    }, ensure_ascii=False))
    return code


def cli_main() -> int:
    """将未处理异常统一映射为脚本错误退出码。"""
    try:
        return main()
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"评测脚本错误 ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return EXIT_SCRIPT_ERROR


if __name__ == "__main__":
    raise SystemExit(cli_main())
