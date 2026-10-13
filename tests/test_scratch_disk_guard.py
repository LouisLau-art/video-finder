"""06_index_all 本地 scratch 拷贝的磁盘空间保护单测。

真实事故（2026-10-13）：NAS 新增 5 个 14–17GB 的 4K 长录屏，3 个 worker
并行把它们整片拷进本地 scratch，堆积 39GB+，宿主机可用空间一度只剩 27GB
（97%→98%，距写满约 10 分钟），靠人工停服务 + 清 scratch 才避免撑爆。

修复判据：拷贝前用 shutil.disk_usage().free 预检，只有
free - file_size >= reserve 时才允许拷贝；否则跳过本地拷贝、回退 NAS 直读。
free 已包含其他 worker 正在写入的部分拷贝所占空间，因此并发天然被计入。
"""
from __future__ import annotations

import collections
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

GB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/06_index_all.py"

Usage = collections.namedtuple("usage", "total used free")


def _load_module():
    spec = importlib.util.spec_from_file_location("index_scratch_guard_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- 纯函数判据：should_copy_to_scratch ----------


def test_allows_copy_when_free_is_plenty():
    module = _load_module()
    # 25GB 可用 vs 14GB 文件 + 5GB 预留：拷完还剩 11GB，安全
    assert module.should_copy_to_scratch(25 * GB, 14 * GB, 5 * GB) is True


def test_refuses_copy_when_free_just_short():
    module = _load_module()
    # 比 size + reserve 差 1 字节：拒绝
    assert module.should_copy_to_scratch(19 * GB - 1, 14 * GB, 5 * GB) is False


def test_boundary_exact_reserve_is_allowed():
    module = _load_module()
    # free - size == reserve（拷完恰好剩 5GB）：取等放行。
    # 理由：reserve 是拷贝完成后必须保住的「下限」而非严格余量，
    # 正好触底仍满足该约束；取严格大于会把恰好够格的场景误判为拒绝。
    assert module.should_copy_to_scratch(19 * GB, 14 * GB, 5 * GB) is True


def test_zero_reserve_degenerate_case():
    module = _load_module()
    # reserve=0：只要装得下就允许（free == size 也算装得下）
    assert module.should_copy_to_scratch(17 * GB, 17 * GB, 0) is True
    # 哪怕只差 1 字节也装不下
    assert module.should_copy_to_scratch(17 * GB - 1, 17 * GB, 0) is False


def test_accident_shape_17gb_file_refused():
    module = _load_module()
    # 事故形状：17GB 4K 长录屏 + 5GB 预留，磁盘只剩 10GB 可用 -> 拒绝本地拷贝
    assert module.should_copy_to_scratch(10 * GB, 17 * GB, 5 * GB) is False
    # 同样 17GB 文件，若有 25GB 可用则放行（25 - 17 = 8 >= 5，拷完剩 8GB）；
    # 24GB 以下即拒绝（19 - 17 = 2 < 5，拷完只剩 2GB，低于事故时那种临界水位）。
    assert module.should_copy_to_scratch(25 * GB, 17 * GB, 5 * GB) is True
    assert module.should_copy_to_scratch(19 * GB, 17 * GB, 5 * GB) is False


# ---------- _extract_job 接线：空间不足回退直读 / 空间充足走本地拷贝 ----------


def _job(tmp_path: Path, video: Path, scratch: Path, reserve_bytes: int) -> dict:
    return {
        "video_path": str(video), "share": "NAS", "relpath": "a/b.mp4",
        "video_key": "deadbeef", "stem": "b",
        "frames_dir": str(tmp_path / "frames"),
        "local_scratch": str(scratch),
        "fps": 1.0, "max_per_scene": 3,
        "scratch_min_free_bytes": reserve_bytes,
    }


def _stub_extract(seen: dict):
    def process_video(path, staging, fps, max_per_scene):
        seen["input"] = Path(path)
        f = Path(staging) / "f.jpg"
        f.write_bytes(b"x")
        return [{"time": 0.0, "frame_path": str(f)}]

    return SimpleNamespace(process_video=process_video)


def test_extract_job_falls_back_to_direct_read_when_scratch_full(
        tmp_path, monkeypatch, capsys):
    module = _load_module()
    video = tmp_path / "nas" / "big.mp4"
    video.parent.mkdir()
    video.write_bytes(b"v" * 8)
    scratch = tmp_path / "scratch"
    seen: dict = {}
    monkeypatch.setattr(module, "_load_extract_module", lambda: _stub_extract(seen))
    # 可用空间 1GB，远小于 5GB 预留 -> 预检拒绝
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: Usage(0, 0, 1 * GB))

    copy_calls: list = []

    def _spy_copyfile(src, dst):  # 预检拒绝时不应被调用
        copy_calls.append((src, dst))

    monkeypatch.setattr(module.shutil, "copyfile", _spy_copyfile)

    res = module._extract_job(_job(tmp_path, video, scratch, 5 * GB))

    assert res["ok"] is True and res["frames"] == 1
    # process_video 收到的是 NAS 原路径，不是本地拷贝
    assert seen["input"] == video
    assert copy_calls == []
    err = capsys.readouterr().err
    assert "空间不足" in err and "直读" in err
    # 走的是预检分支，而不是「拷贝异常回退」分支
    assert "本地拷贝失败" not in err
    assert not list(scratch.glob("*.mp4"))  # 没有留下任何拷贝


def test_extract_job_copies_locally_when_space_allowed(tmp_path, monkeypatch):
    module = _load_module()
    video = tmp_path / "nas" / "small.mp4"
    video.parent.mkdir()
    video.write_bytes(b"v" * 8)
    scratch = tmp_path / "scratch"
    seen: dict = {}
    real_copyfile = shutil.copyfile

    def _spy_copyfile(src, dst):
        seen["copy"] = (Path(src), Path(dst))
        real_copyfile(src, dst)

    monkeypatch.setattr(module, "_load_extract_module", lambda: _stub_extract(seen))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: Usage(0, 0, 20 * GB))
    monkeypatch.setattr(module.shutil, "copyfile", _spy_copyfile)

    res = module._extract_job(_job(tmp_path, video, scratch, 5 * GB))

    assert res["ok"] is True and res["frames"] == 1
    copied_src, copied_dst = seen["copy"]
    assert copied_src == video
    assert copied_dst.name == "deadbeef.mp4" and copied_dst.parent == scratch
    assert seen["input"] == copied_dst  # 解码用的是本地拷贝
    assert not copied_dst.exists()  # 用后即删，不占空间