"""图片删除下架的检索层过滤测试（ticket-04-part2）。

测公开行为：state_images.json 中处于 deleted_pending（缓冲期内下架）的图片行，
不出现在搜索结果里；done 行不受影响；state 文件缺失时行为与旧版一致。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def _load_api_module():
    spec = importlib.util.spec_from_file_location(
        "api_05_gc", REPO / "scripts" / "05_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- 1. 下架集合加载（纯函数） ----------

def test_load_deleted_images_reads_pending_entries(tmp_path):
    """deleted_pending 的 image_key 进下架集合；done/failed 不进。"""
    m = _load_api_module()
    state = tmp_path / "state_images.json"
    state.write_text(json.dumps({
        "aaaa1111": {"status": "deleted_pending", "deleted_at": 1.0},
        "bbbb2222": {"status": "done"},
        "cccc3333": {"status": "failed"},
        "dddd4444": {"status": "deleted_pending", "deleted_at": 2.0},
    }), encoding="utf-8")
    deleted = m.load_deleted_image_keys(state)
    assert deleted == {"aaaa1111", "dddd4444"}


def test_load_deleted_images_missing_file_returns_empty(tmp_path):
    """state 文件不存在（从未跑过 GC）：返回空集合，不影响检索。"""
    m = _load_api_module()
    assert m.load_deleted_image_keys(tmp_path / "nope.json") == set()


def test_load_deleted_images_corrupt_file_returns_empty(tmp_path):
    """state 损坏：容错返回空集合（不让检索挂掉）。"""
    m = _load_api_module()
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    assert m.load_deleted_image_keys(f) == set()


# ---------- 2. 行级过滤（纯函数） ----------

def _row(image_key: str) -> dict:
    return {"rank": 1, "video_id": "x", "media_type": "image",
            "image_key": image_key}


def test_filter_drops_pending_image_rows():
    """下架图片行被剔除；视频行与非下架图片行保留，rank 重排连续。"""
    m = _load_api_module()
    rows = [
        _row("aaaa1111"),                       # 下架 -> 剔除
        {"rank": 2, "video_id": "v", "media_type": "video", "video_key": "v1"},
        _row("bbbb2222"),                       # 正常图片 -> 保留
        _row("cccc3333"),                       # 下架 -> 剔除
    ]
    kept = m.filter_deleted_rows(rows, {"aaaa1111", "cccc3333"})
    assert len(kept) == 2
    # 保留下来的行 rank 重排为 1、2（不留空洞）
    assert [r["rank"] for r in kept] == [1, 2]
    assert kept[1]["image_key"] == "bbbb2222"


def test_filter_noop_when_empty_deleted_set():
    """下架集合为空（正常态）：行原样返回（逐位一致，不复制对象）。"""
    m = _load_api_module()
    rows = [_row("aaaa1111"), {"rank": 2, "media_type": "video"}]
    kept = m.filter_deleted_rows(rows, set())
    assert kept == rows


def test_filter_video_rows_never_dropped():
    """视频帧没有 image_key，即使集合非空也不受影响（GC 只碰图片）。"""
    m = _load_api_module()
    rows = [{"rank": 1, "media_type": "video", "video_key": "v1"}]
    kept = m.filter_deleted_rows(rows, {"anything"})
    assert kept == rows
