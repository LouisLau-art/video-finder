"""01_extract 和 06_index_all 的「抽帧限宽」能力单测。

背景：4K 超长视频（如 50 分钟 3840x2160）全片抽帧若不做缩放，
单帧约 1~1.5MB，一部视频可产生数千张 4K 巨帧（数 GB 甚至数十 GB），
会直接撑爆工位机磁盘。
加入限宽 1280（保持宽高比等比缩放）后，单帧从 1MB 骤降至 ~100KB（体积缩减 90%），
且小于限宽的低分辨率视频保持原样（不放大）。
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load_module(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m01 = _load_module("01_extract.py", "mod_01_extract_test")


def _create_test_video(path: Path, width: int = 1920, height: int = 1080) -> None:
    """使用 ffmpeg testsrc 生成 1 秒单色测试视频（yuv420p）。"""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=10",
        "-pix_fmt", "yuv420p", "-t", "1", str(path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_extract_frame_scales_down_large_frame(tmp_path: Path):
    """大分辨率（1920x1080）抽取时限宽 640，等比缩放为 640x360。"""
    v = tmp_path / "large.mp4"
    _create_test_video(v, 1920, 1080)

    out = tmp_path / "frame_640.jpg"
    ok = m01.extract_frame(v, 0.5, out, max_width=640)
    assert ok is True
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (640, 360)


def test_extract_frame_preserves_smaller_resolution(tmp_path: Path):
    """小分辨率（640x480）抽帧限宽 1280 时保持 640x480，不放大。"""
    v = tmp_path / "small.mp4"
    _create_test_video(v, 640, 480)

    out = tmp_path / "frame_small.jpg"
    ok = m01.extract_frame(v, 0.5, out, max_width=1280)
    assert ok is True
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (640, 480)


def test_extract_frame_unlimited_when_zero(tmp_path: Path):
    """max_width=0 表示不限宽，保持原画质尺寸。"""
    v = tmp_path / "orig.mp4"
    _create_test_video(v, 1920, 1080)

    out = tmp_path / "frame_orig.jpg"
    ok = m01.extract_frame(v, 0.5, out, max_width=0)
    assert ok is True
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (1920, 1080)
