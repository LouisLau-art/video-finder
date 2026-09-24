"""工单 04 评测统计逻辑的内存单元测试。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/07_hybrid_eval.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("hybrid_eval_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_summary_counts_exclusive_video_hits():
    module = _load_eval_module()
    cases = [
        {"id": "keyword_case", "targets": ["v_keyword"]},
        {"id": "semantic_case", "targets": ["v_semantic"]},
        {"id": "both_case", "targets": ["v_shared"]},
    ]
    rankings = {
        "keyword_case": {
            "semantic": ["v_other"],
            "keyword": ["v_keyword"],
            "fused": ["v_keyword"],
        },
        "semantic_case": {
            "semantic": ["v_semantic"],
            "keyword": ["v_other"],
            "fused": ["v_semantic"],
        },
        "both_case": {
            "semantic": ["v_shared"],
            "keyword": ["v_shared"],
            "fused": ["v_shared"],
        },
    }

    summary = module._metric_summary(cases, rankings, top_k=20)

    assert summary["semantic_hit_rate"] == pytest.approx(2 / 3)
    assert summary["keyword_hit_rate"] == pytest.approx(2 / 3)
    assert summary["fused_hit_rate"] == pytest.approx(1)
    assert summary["keyword_only_target_hits"] == 1
    assert summary["semantic_only_target_hits"] == 1
    assert summary["both_target_hits"] == 1


def test_generic_injection_counts_keyword_only_candidates():
    module = _load_eval_module()
    cases = [{"id": "generic", "generic": True, "targets": ["v_target"]}]
    rankings = {
        "generic": {
            "semantic": ["v_target", "v_semantic"],
            "keyword": ["v_target", "v_injected"],
            "fused": ["v_target", "v_injected"],
        }
    }

    summary = module._generic_injection_summary(cases, rankings, top_k=2)

    assert summary["keyword_topk_material_occurrences"] == 2
    assert summary["injected_material_occurrences"] == 1
    assert summary["injected_unique_materials"] == 1
    assert summary["injection_ratio"] == pytest.approx(0.5)


def test_coverage_ledger_separates_natural_typed_and_hash_names():
    module = _load_eval_module()
    index = {
        "v_readable": {
            "filename": "alpha_name",
            "parents": (),
            "typed": False,
        },
        "v_typed": {
            "filename": "OUT_0601",
            "parents": (),
            "typed": True,
        },
        "v_hash": {
            "filename": "0123456789abcdef0123456789abcdef",
            "parents": (),
            "typed": True,
        },
        "v_parent": {
            "filename": "raw_file",
            "parents": ("春季系列",),
            "typed": False,
        },
    }

    ledger = module._coverage_ledger(index)

    assert ledger["counts"]["total"] == 4
    assert ledger["counts"]["natural_text_coverable"] == 2
    assert ledger["counts"]["exact_capable_including_typed"] == 3
    assert ledger["natural_text_coverage_rate"] == pytest.approx(0.5)
    assert ledger["exact_capability_upper_bound_rate"] == pytest.approx(0.75)


def test_proprietary_case_builder_covers_filename_parent_and_generic_sources():
    module = _load_eval_module()
    index = {
        "v_file": {
            "filename": "event_name",
            "parents": ("ordinary",),
            "typed": False,
        },
        "v_parent": {
            "filename": "raw_file",
            "parents": ("产品", "spring"),
            "typed": False,
        },
        **{f"v_generic_{i}": {
            "filename": f"raw_{i}",
            "parents": ("产品",),
            "typed": False,
        } for i in range(6)},
    }

    cases = module._build_proprietary_cases(index, 5, 5, 2)
    sources = {case["source"] for case in cases}

    assert "filename" in sources
    assert "parent" in sources
    assert any(case["generic"] for case in cases)
    assert all(case["query"] and case["targets"] for case in cases)
