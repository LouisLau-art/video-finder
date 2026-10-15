"""06d_gc_images.py 图片删除下架与 30 天缓冲回收的单元测试。

只使用内存假库 / 假 state，不读取 NAS、真实向量库。
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "06d_gc_images.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gc_images_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- 1. 下架判定（纯函数） ----------

def test_mark_deleted_sets_first_seen_missing_timestamp():
    """首次发现缺失：记 deleted_at 时间戳，status 改 deleted_pending。"""
    m = _load_module()
    now = time.time()
    entry = {"status": "done", "mtime": 100.0, "size": 42}
    out = m.mark_deleted(entry, now)
    assert out["status"] == "deleted_pending"
    assert out["deleted_at"] == now
    # 原索引信息保留，供 30 天后对账
    assert out["mtime"] == 100.0 and out["size"] == 42


def test_mark_deleted_keeps_original_timestamp_on_rescan():
    """二次扫描仍缺失：deleted_at 保持首次时间，不被刷新（30 天从首次缺失起算）。"""
    m = _load_module()
    first = time.time()
    entry = m.mark_deleted({"status": "done"}, first)
    out = m.mark_deleted(entry, first + 86400 * 10)
    assert out["deleted_at"] == first
    assert out["status"] == "deleted_pending"


def test_mark_deleted_restores_when_file_reappears():
    """文件重新出现（误删恢复）：清掉 deleted_at 与 pending 状态。"""
    m = _load_module()
    entry = m.mark_deleted({"status": "done"}, time.time())
    out = m.mark_restored(entry)
    assert out.get("status") == "done"
    assert "deleted_at" not in out
    assert "deleted_pending" not in json.dumps(out, ensure_ascii=False)


def test_mark_restored_noop_when_never_deleted():
    """未下架条目调用 restore 不改动（幂等）。"""
    m = _load_module()
    entry = {"status": "done", "mtime": 1.0}
    out = m.mark_restored(entry)
    assert out == {"status": "done", "mtime": 1.0}


# ---------- 2. 30 天真删判定（纯函数） ----------

def test_should_purge_false_before_30_days():
    """缺失未满 30 天：不真删（缓冲期内）。"""
    m = _load_module()
    now = time.time()
    entry = {"status": "deleted_pending", "deleted_at": now - 86400 * 29}
    assert m.should_purge(entry, now) is False


def test_should_purge_true_after_30_days():
    """缺失满 30 天：真删。"""
    m = _load_module()
    now = time.time()
    entry = {"status": "deleted_pending", "deleted_at": now - 86400 * 31}
    assert m.should_purge(entry, now) is True


def test_should_purge_false_for_done_entries():
    """done 条目（文件还在）永不真删。"""
    m = _load_module()
    assert m.should_purge({"status": "done"}, time.time()) is False


# ---------- 3. 与 state 的对账主流程 ----------

class _FakeCollection:
    """记录 delete 调用的假 Chroma 集合。"""

    def __init__(self):
        self.deleted_ids: list[str] = []

    def delete(self, ids=None, where=None):
        if ids:
            self.deleted_ids.extend(ids)


def _write_state(path: Path, entries: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def test_reconcile_marks_missing_and_purges_after_buffer(tmp_path):
    """端到端：缺失条目先标记，满 30 天才进 Chroma delete。"""
    m = _load_module()
    now = time.time()
    state_path = tmp_path / "state_images.json"
    _write_state(state_path, {
        "k1": {"status": "done", "mtime": 1.0, "size": 10,
               "frame_path": "/gone/a.jpg"},                      # 刚缺失 -> 标记
        "k2": {"status": "deleted_pending", "mtime": 2.0, "size": 20,
               "deleted_at": now - 86400 * 31,
               "frame_path": "/gone/b.jpg"},                      # 满 30 天 -> 真删
        "k3": {"status": "done", "mtime": 3.0, "size": 30,
               "frame_path": str(tmp_path / "still_here.jpg")},   # 还在 -> 不动
    })
    (tmp_path / "still_here.jpg").write_bytes(b"x")

    col = _FakeCollection()
    stats = m.reconcile_deletions(
        state_path=state_path,
        collection=col,
        existing_paths={str(tmp_path / "still_here.jpg")},
        now=now,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    # k1：刚标记，未真删
    assert state["k1"]["status"] == "deleted_pending"
    assert "k1" not in "".join(col.deleted_ids)
    # k2：满 30 天，Chroma 删除 + state 移除
    assert any("k2" in i for i in col.deleted_ids)
    assert "k2" not in state
    # k3：不受影响
    assert state["k3"]["status"] == "done"
    # 统计：新标记 1、真删 1、存活 1
    assert stats == {"marked": 1, "purged": 1, "alive": 1}


def test_reconcile_restores_reappeared_file(tmp_path):
    """误删后文件回来：条目恢复 done，不进 Chroma delete。"""
    m = _load_module()
    now = time.time()
    state_path = tmp_path / "state_images.json"
    alive = tmp_path / "back.jpg"
    alive.write_bytes(b"x")
    _write_state(state_path, {
        "k1": {"status": "deleted_pending", "deleted_at": now - 86400 * 5,
               "frame_path": str(alive), "mtime": 1.0, "size": 5},
    })

    col = _FakeCollection()
    m.reconcile_deletions(
        state_path=state_path, collection=col,
        existing_paths={str(alive)}, now=now)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["k1"]["status"] == "done"
    assert "deleted_at" not in state["k1"]
    assert col.deleted_ids == []
