"""06_index_all 的「跳过管理」能力单测：标记跳过 + 仅重跑被跳过的。

背景：NAS 里的超长 4K 录屏远超磁盘余量，需人工判定暂不处理（标记 skipped）。
但跳过不应是永久放弃——补索引路径修好后要能「只重跑被跳过的」，
而不触发 8000+ 已索引素材的整批重抽。

只使用内存假数据与临时目录，不读取 NAS、真实向量库或真实模型。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/06_index_all.py"
GB = 1024 ** 3


def _load_module():
    spec = importlib.util.spec_from_file_location("index_skip_mgmt_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _video(key: str, size_gb: float, relpath: str = "d/v.mp4") -> dict:
    return {"video_key": key, "size": int(size_gb * GB), "mtime": 1.0,
            "relpath": relpath, "share": "s", "stem": Path(relpath).stem,
            "video_path": f"/nas/{relpath}"}


# ---------- select_for_mark_skip ----------

def test_select_all_when_no_threshold():
    """min_gb<=0 -> 全部命中（不限体积）。"""
    m = _load_module()
    vids = [_video("a", 0.1), _video("b", 15.0)]
    assert [v["video_key"] for v in m.select_for_mark_skip(vids, 0.0)] == ["a", "b"]


def test_select_only_above_threshold():
    """min_gb>0 -> 只命中体积 >= 阈值的（含恰好相等）。"""
    m = _load_module()
    vids = [_video("small", 4.7), _video("edge", 12.0), _video("big", 17.0)]
    picked = m.select_for_mark_skip(vids, 12.0)
    assert [v["video_key"] for v in picked] == ["edge", "big"]


def test_select_excludes_below_threshold():
    """低于阈值的排除，避免误伤普通素材。"""
    m = _load_module()
    vids = [_video("a", 11.9), _video("b", 12.1)]
    picked = m.select_for_mark_skip(vids, 12.0)
    assert [v["video_key"] for v in picked] == ["b"]


# ---------- build_skip_entry ----------

def test_build_skip_entry_shape():
    """skip 条目保留体积/mtime、frames 为空、带 reason 与终态 status。"""
    m = _load_module()
    entry = m.build_skip_entry(_video("k", 15.0, relpath="d/x.mp4"),
                               "超长录屏，暂不处理")
    assert entry["status"] == "skipped"
    assert entry["frames"] == []
    assert entry["skip_reason"] == "超长录屏，暂不处理"
    assert entry["size"] == int(15.0 * GB)
    assert entry["mtime"] == 1.0


# ---------- mark_skip 主流程（通过 main 的 CLI 集成） ----------

def _run_main(module, argv, monkeypatch, videos):
    """跑 module.main()，把 discover/load_state/save_state 换成假实现。"""
    monkeypatch.setattr(module, "discover_videos", lambda *a, **k: list(videos))
    monkeypatch.setattr(module, "check_roots_readable", lambda roots: [])
    saved = {}
    monkeypatch.setattr(module, "load_state", lambda db: dict(saved) if saved else {})
    monkeypatch.setattr(module, "save_state",
                        lambda db, st: saved.update(st))
    monkeypatch.setattr("sys.argv", ["06_index_all.py"] + argv)
    rc = module.main()
    return rc, saved


def test_mark_skip_marks_only_above_threshold(tmp_path, monkeypatch):
    """--mark-skip 只标记超阈值视频，且跳过已 done 的不降级。"""
    m = _load_module()
    videos = [_video("small", 5.0, "a/v.mp4"), _video("big", 17.0, "b/v.mp4"),
              _video("donebig", 20.0, "c/v.mp4")]
    # main() 会用 video_key_of 重算 key；测试须用同一算法构造预置 state 的 key
    k_big = m.video_key_of("s", "b/v.mp4")
    k_done = m.video_key_of("s", "c/v.mp4")
    monkeypatch.setattr(m, "discover_videos", lambda *a, **k: list(videos))
    monkeypatch.setattr(m, "check_roots_readable", lambda roots: [])
    state = {k_done: {"status": "done", "frames": ["f.jpg"]}}
    monkeypatch.setattr(m, "load_state", lambda db: dict(state))
    saved = {}
    monkeypatch.setattr(m, "save_state", lambda db, st: saved.update(st))
    monkeypatch.setattr("sys.argv", [
        "06_index_all.py", "--mark-skip", "--mark-skip-min-gb", "12",
        "--skip-reason", "超长录屏", "--db", str(tmp_path),
        "--frames-dir", str(tmp_path / "frames")])

    rc = m.main()

    assert rc == 0
    assert saved[k_big]["status"] == "skipped"
    assert saved[k_big]["skip_reason"] == "超长录屏"
    assert m.video_key_of("s", "a/v.mp4") not in saved   # 低于阈值不标记
    assert saved[k_done]["status"] == "done"             # 已 done 不降级


def test_mark_skip_requires_reason(tmp_path, monkeypatch):
    """--mark-skip 缺 --skip-reason -> 拒绝执行（防止无声跳过）。"""
    m = _load_module()
    monkeypatch.setattr(m, "discover_videos", lambda *a, **k: [_video("a", 20.0)])
    monkeypatch.setattr(m, "check_roots_readable", lambda roots: [])
    monkeypatch.setattr(m, "load_state", lambda db: {})
    monkeypatch.setattr("sys.argv", [
        "06_index_all.py", "--mark-skip", "--db", str(tmp_path),
        "--frames-dir", str(tmp_path / "frames")])
    assert m.main() == 2


def test_only_skipped_decision_reprocesses_only_skipped():
    """--only-skipped 语义：只有 skipped 重跑，done/failed/新素材一律跳过。"""
    m = _load_module()
    skipped = {"status": "skipped"}
    done = {"status": "done", "frames": ["f.jpg"]}
    failed = {"status": "failed"}

    assert m.skip_state_decision(skipped, False, True) == "reprocess"
    assert m.skip_state_decision(done, False, True) == "skip"
    assert m.skip_state_decision(failed, False, True) == "skip"
    assert m.skip_state_decision(None, False, True) == "skip"  # 新素材也不动


def test_default_decision_skips_skipped_only():
    """默认（无 only-skipped/force）：skipped 跳过；done/failed 交常规逻辑。"""
    m = _load_module()
    assert m.skip_state_decision({"status": "skipped"}, False, False) == "skip"
    assert m.skip_state_decision({"status": "done"}, False, False) is None
    assert m.skip_state_decision({"status": "failed"}, False, False) is None
    assert m.skip_state_decision(None, False, False) is None


def test_force_decision_always_normal():
    """--force：一律交常规逻辑（全部重跑）。"""
    m = _load_module()
    assert m.skip_state_decision({"status": "skipped"}, True, False) is None
    assert m.skip_state_decision({"status": "done"}, True, False) is None


def test_mark_skip_then_only_skipped_roundtrip(tmp_path, monkeypatch):
    """往返：标记跳过 -> 只有它被 --only-skipped 重新纳入。"""
    m = _load_module()
    entry = m.build_skip_entry(_video("big", 15.0), "超长录屏")
    assert m.skip_state_decision(entry, False, False) == "skip"
    assert m.skip_state_decision(entry, False, True) == "reprocess"

