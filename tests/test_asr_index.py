"""tests/test_asr_index.py — 06c_index_asr.py 语音台词转写与索引管线测试。

只使用模拟/内存假数据，不连接真实 NAS、真实音视频硬件或真实模型下载。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "06c_index_asr.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("index_asr_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asr_mod = _load_module()


# ---------- 1. 文本清洗 ----------

def test_clean_sensevoice_tags():
    raw1 = "<|zh|><|NEUTRAL|><|BGM|><|withitn|>最近穿的是这件5公里跑步速干T"
    assert asr_mod.clean_sensevoice_tags(raw1) == "最近穿的是这件5公里跑步速干T"

    raw2 = "<|HAPPY|><|Speech|>加油，坚持到底！ <|BGM|>"
    assert asr_mod.clean_sensevoice_tags(raw2) == "加油，坚持到底！"

    raw_empty = "<|zh|><|NEUTRAL|>"
    assert asr_mod.clean_sensevoice_tags(raw_empty) == ""
    assert asr_mod.clean_sensevoice_tags("") == ""


# ---------- 2. 时间戳格式化 ----------

def test_format_timestamp():
    assert asr_mod.format_timestamp(0.0) == "00:00"
    assert asr_mod.format_timestamp(8.3) == "00:08"
    assert asr_mod.format_timestamp(65.4) == "01:05"
    assert asr_mod.format_timestamp(3665.0) == "01:01:05"
    assert asr_mod.format_timestamp(-5.0) == "00:00"


# ---------- 3. 身份键计算 ----------

def test_compute_video_key():
    k1 = asr_mod.compute_video_key("示例共享", "0视频/产品/t.mp4")
    k2 = asr_mod.compute_video_key("示例共享", "0视频\\产品\\t.mp4")
    assert len(k1) == 8
    assert k1 == k2  # 斜杠归一化


# ---------- 4. Chroma 文档构建 ----------

def test_build_chroma_transcript_entry():
    seg = {
        "segment_id": 3,
        "start": 8.35,
        "end": 14.2,
        "time_formatted": "00:08",
        "text": "这双鞋透气性极佳",
    }
    entry = asr_mod.build_chroma_transcript_entry(
        video_key="a1b2c3d4",
        video_id="running_shoes",
        share="示例共享",
        relpath="0视频/running_shoes.mp4",
        segment=seg,
    )
    assert entry["id"] == "a1b2c3d4_asr_0003"
    assert entry["document"] == "这双鞋透气性极佳"
    meta = entry["metadata"]
    assert meta["video_key"] == "a1b2c3d4"
    assert meta["video_id"] == "running_shoes"
    assert meta["segment_id"] == 3
    assert meta["start"] == 8.35
    assert meta["time"] == 8.35
    assert meta["time_formatted"] == "00:08"
    assert meta["media_type"] == "video"
    assert meta["match_type"] == "transcript"


# ---------- 5. 检查音轨 ----------

@patch("subprocess.run")
def test_check_audio_stream(mock_run):
    mock_run.return_value = MagicMock(return_value=0, stdout="audio\n")
    assert asr_mod.check_audio_stream(Path("/fake/video.mp4")) is True

    mock_run.return_value = MagicMock(return_value=0, stdout="\n")
    assert asr_mod.check_audio_stream(Path("/fake/video.mp4")) is False


# ---------- 6. 抽取音频 WAV ----------

@patch("subprocess.run")
def test_extract_audio_wav(mock_run, tmp_path):
    out_wav = tmp_path / "test.wav"

    def side_effect(*args, **kwargs):
        out_wav.write_bytes(b"RIFF" + b"\x00" * 100)
        return MagicMock(returncode=0)

    mock_run.side_effect = side_effect
    assert asr_mod.extract_audio_wav(Path("/fake/in.mp4"), out_wav) is True
    assert out_wav.exists()


# ---------- 7. Transcriber 切段转写逻辑 ----------

def test_transcriber_transcribe_mock(tmp_path):
    transcriber = asr_mod.AsrTranscriber(language="zh")
    transcriber._loaded = True

    # 模拟 VAD
    mock_vad = MagicMock()
    mock_vad.generate.return_value = [{"value": [[500, 3000], [4000, 7000]]}]
    transcriber.vad_model = mock_vad

    # 模拟 ASR
    mock_asr = MagicMock()
    mock_asr.generate.side_effect = [
        [{"text": "<|zh|><|NEUTRAL|>第一句话"}],
        [{"text": "<|zh|><|HAPPY|>第二句话"}],
    ]
    transcriber.asr_model = mock_asr

    # 创建一个假 wav
    wav_file = tmp_path / "dummy.wav"
    import numpy as np
    import soundfile as sf
    sr = 16000
    fake_audio = np.zeros(sr * 8, dtype=np.float32)
    sf.write(str(wav_file), fake_audio, sr)

    res = transcriber.transcribe(wav_file)
    assert len(res) == 2
    assert res[0]["segment_id"] == 0
    assert res[0]["text"] == "第一句话"
    assert res[0]["start"] == 0.5
    assert res[0]["end"] == 3.0
    assert res[0]["time_formatted"] == "00:00"

    assert res[1]["segment_id"] == 1
    assert res[1]["text"] == "第二句话"
    assert res[1]["start"] == 4.0
    assert res[1]["end"] == 7.0
    assert res[1]["time_formatted"] == "00:04"


# ---------- 8. 发现范围与剪枝（与 06/06b 语义一致）----------

def test_discover_nas_videos_prunes_and_filters(tmp_path):
    """NAS 分支：剪枝 FCP/隐藏/依赖目录，后缀与 include/exclude 语义与 06 一致。"""
    root = tmp_path / "mnt"
    keep = root / "0视频" / "产品"
    keep.mkdir(parents=True)
    (keep / "keep.mp4").write_bytes(b"x")
    (keep / "upper.MP4").write_bytes(b"x")  # 后缀大小写不敏感
    (keep / "skip.mkv").write_bytes(b"x")   # 非 06 后缀，不转写
    (keep / ".hidden.mp4").write_bytes(b"x")
    fcp = root / "0视频" / "bundle.fcpbundle"
    fcp.mkdir()
    (fcp / "inner.mp4").write_bytes(b"x")
    deps = root / "0视频" / "node_modules"
    deps.mkdir()
    (deps / "dep.mp4").write_bytes(b"x")
    hidden_dir = root / "0视频" / ".cache"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.mp4").write_bytes(b"x")
    exclude_me = root / "0视频" / "旧素材"
    exclude_me.mkdir()
    (exclude_me / "old.mp4").write_bytes(b"x")

    found = asr_mod.discover_nas_videos([str(root)])
    rels = sorted(rel for _share, rel, _p in found)
    assert rels == [
        "0视频/产品/keep.mp4",
        "0视频/产品/upper.MP4",
        "0视频/旧素材/old.mp4",  # 未传 exclude 时正常发现
    ]
    assert all(share == "mnt" for share, _rel, _p in found)  # 未配置映射时回退 basename

    filtered = asr_mod.discover_nas_videos(
        [str(root)], include_dir=["产品"], exclude_dir=["旧素材"])
    frels = sorted(rel for _s, rel, _p in filtered)
    assert frels == ["0视频/产品/keep.mp4", "0视频/产品/upper.MP4"]

    none_found = asr_mod.discover_nas_videos([str(root)], include_dir=["不存在目录"])
    assert none_found == []


def test_discover_local_videos(tmp_path):
    """本地分支：返回 share='local'，跳过隐藏与非视频文件。"""
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mov").write_bytes(b"x")
    (tmp_path / "sub" / "c.txt").write_text("x")
    (tmp_path / ".d.mp4").write_bytes(b"x")

    found = asr_mod.discover_local_videos(tmp_path)
    assert [(share, rel) for share, rel, _p in found] == [
        ("local", "a.mp4"),
        ("local", "sub/b.mov"),
    ]
