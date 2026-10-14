#!/usr/bin/env python3
"""06c_index_asr.py — 视频音频台词转写与多模态检索索引管线。

利用阿里 SenseVoice-Small + FSMN-VAD 离线模型，从视频中提取音频并转写为带起止时间戳的台词文本；
使用现有 CN-CLIP 文本塔将台词向量化并写入 ChromaDB collection（默认 'transcripts'），
使检索系统同时具备"画面语义 + 口播台词语义 + 台词关键词精确定位到秒"的能力。

流程：
1. 发现素材并检查断点状态（state_asr.json）
2. ffprobe 探测音轨（无音轨素材标记 has_audio=False 跳过）
3. ffmpeg 提取 16kHz 单声道 WAV 至暂存区
4. FSMN-VAD 检测语音片段，SenseVoice-Small 转写台词文本
5. CN-CLIP 文本塔对台词片段计算 512 维向量
6. 写入 Chroma 'transcripts' collection 并保存 state_asr.json

用法：
    .venv/bin/python scripts/06c_index_asr.py --dry-run
    .venv/bin/python scripts/06c_index_asr.py --limit 10 --videos-dir videos_sample
    .venv/bin/python scripts/06c_index_asr.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

DEFAULT_DB = "data/local-runtime/index_full/cnclip"
DEFAULT_STATE = "data/local-runtime/index_full/state_asr.json"
DEFAULT_COLLECTION = "transcripts"
DEFAULT_SCRATCH_DIR = "data/local-runtime/scratch_asr"

# 与 06 管线保持同一后缀集合：ASR 只应转写「会被 06 抽帧索引」的素材，
# 否则会产出没有帧图、也搜不到画面的孤儿台词（身份对不上）。
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
CLEAN_TAG_RE = re.compile(r"<\|[^|>]+(?:\|[^|>]+)*\|>")

# 与 06 相同的代码依赖目录剪枝（CIFS 上枚举极慢）。
PRUNE_DIRNAMES = {
    "site-packages", "node_modules", "__pycache__", "venv",
    ".tox", ".eggs", "__pypackages__", "pip-wheel-metadata",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "@eadir", "#recycle", "#snapshot",
}

# 连续失败熔断阈值
MAX_CONSECUTIVE_FAILURES = 20


def _load_share_map() -> dict:
    """shares 列表 -> {mount: name}；失败返回空 dict（与 06/06b 同款）。"""
    try:
        spec = importlib.util.spec_from_file_location(
            "site_config", str(SCRIPTS / "site_config.py"))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        out: dict = {}
        for s in mod.load().get("shares", []) or []:
            if isinstance(s, dict) and s.get("mount") and s.get("name"):
                out[str(s["mount"]).rstrip("/")] = str(s["name"])
        return out
    except Exception:
        return {}


SHARE_MAP = _load_share_map()


def _default_roots() -> list[str]:
    """--roots 默认值：site.json 的 shares 挂载点列表；无配置时退回本地默认。"""
    if SHARE_MAP:
        return sorted(SHARE_MAP)
    return ["/mnt/nas"]


def share_of(root: str) -> str:
    """挂载点 -> 共享名；未知根目录回退用其 basename。"""
    return SHARE_MAP.get(root.rstrip("/"), Path(root).name)


def clean_sensevoice_tags(text: str) -> str:
    """清除 SenseVoice-Small 输出的情感、语言、BGM 等特殊标记。"""
    if not text:
        return ""
    cleaned = CLEAN_TAG_RE.sub("", text)
    # 清理多余连续空白
    return re.sub(r"\s+", " ", cleaned).strip()


def format_timestamp(seconds: float) -> str:
    """将秒数格式化为 MM:SS 或 HH:MM:SS。"""
    if seconds < 0:
        return "00:00"
    total_sec = int(round(seconds))
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def compute_video_key(share: str, relpath: str) -> str:
    """计算与 06 / 05 同构的 8 位 md5 身份键。"""
    key_src = f"{share}/{relpath}".replace("\\", "/")
    return hashlib.md5(key_src.encode("utf-8")).hexdigest()[:8]


def check_audio_stream(video_path: Path) -> bool:
    """使用 ffprobe 检查视频文件是否包含音频流。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return "audio" in res.stdout.strip().lower()
    except Exception:
        return False


def extract_audio_wav(video_path: Path, output_wav: Path) -> bool:
    """使用 ffmpeg 将视频音频抽取为 16kHz 16-bit 单声道 WAV 文件。"""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, check=False)
        return res.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0
    except Exception:
        return False


class AsrTranscriber:
    """封装 FSMN-VAD 与 SenseVoice-Small 离线模型的语音识别器。"""

    def __init__(self, language: str = "zh"):
        self.language = language
        self._loaded = False
        self.vad_model = None
        self.asr_model = None

    def load_models(self):
        if self._loaded:
            return
        # 确保本地已下载缓存可用
        from funasr import AutoModel

        print("[asr] 正在加载 FSMN-VAD 语音活动检测模型...")
        self.vad_model = AutoModel(
            model="fsmn-vad",
            trust_remote_code=True,
            disable_update=True,
        )
        print("[asr] 正在加载 SenseVoice-Small 离线语音识别模型...")
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            disable_update=True,
        )
        self._loaded = True
        print("[asr] ASR 模型加载完毕。")

    def transcribe(self, wav_path: Path) -> list[dict[str, Any]]:
        """从音频文件中检测语音分段并转写文字，返回带时间戳的片段列表。"""
        self.load_models()
        import soundfile as sf

        audio_data, sample_rate = sf.read(str(wav_path))
        if audio_data is None or len(audio_data) == 0:
            return []

        # 1. 语音切片检测 (VAD)
        vad_res = self.vad_model.generate(input=str(wav_path))
        if not vad_res or "value" not in vad_res[0]:
            return []

        segments_ms = vad_res[0]["value"]
        results: list[dict[str, Any]] = []

        # 2. 逐段切波形并转写
        for idx, (start_ms, end_ms) in enumerate(segments_ms):
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            chunk = audio_data[start_sample:end_sample]
            if len(chunk) < int(0.2 * sample_rate):  # 忽略小于 0.2 秒的杂音
                continue

            asr_res = self.asr_model.generate(
                input=chunk,
                fs=sample_rate,
                language=self.language,
                use_itn=True,
            )
            raw_text = asr_res[0]["text"] if asr_res else ""
            clean_text = clean_sensevoice_tags(raw_text)
            if not clean_text:
                continue

            start_sec = round(start_ms / 1000.0, 2)
            end_sec = round(end_ms / 1000.0, 2)
            results.append({
                "segment_id": idx,
                "start": start_sec,
                "end": end_sec,
                "time_formatted": format_timestamp(start_sec),
                "text": clean_text,
            })

        return results


def discover_local_videos(vdir: Path) -> list[tuple[str, str, Path]]:
    """本地目录发现（--videos-dir 分支）：返回 [(share='local', rel, path)]。"""
    out: list[tuple[str, str, Path]] = []
    for p in sorted(vdir.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."):
            out.append(("local", p.relative_to(vdir).as_posix(), p))
    return out


def discover_nas_videos(
    roots: list[str],
    include_dir: list[str] | None = None,
    exclude_dir: list[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """NAS 挂载点发现（与 06/06b 语义一致）：返回 [(share, relpath, path)]。

    跳过：.fcpbundle 子树、隐藏目录/隐藏路径段、代码依赖与群晖系统目录
    （PRUNE_DIRNAMES）、非 mp4/mov/m4v 后缀。
    include/exclude 为 relpath 的子串包含匹配（可重复，与 06 一致）。
    """
    include = list(include_dir or [])
    exclude = list(exclude_dir or [])
    out: list[tuple[str, str, Path]] = []
    for root in roots:
        r = Path(root)
        if not r.is_dir():
            print(f"[warn] 根目录不存在跳过: {root}", file=sys.stderr)
            continue
        share = share_of(root)
        for dirpath, dirnames, filenames in os.walk(
                root, onerror=lambda e: print(
                    f"[warn] 遍历失败 {e.filename}: {e.strerror}", file=sys.stderr)):
            dirnames[:] = [
                d for d in dirnames
                if not d.endswith(".fcpbundle") and not d.startswith(".")
                and d.lower() not in PRUNE_DIRNAMES
            ]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                if p.suffix.lower() not in VIDEO_EXTS:
                    continue
                try:
                    rel = p.relative_to(r).as_posix()
                except ValueError:
                    continue
                if any(seg.startswith(".") for seg in rel.split("/")):
                    continue
                if include and not any(s in rel for s in include):
                    continue
                if exclude and any(s in rel for s in exclude):
                    continue
                out.append((share, rel, p))
    return out


def build_chroma_transcript_entry(
    video_key: str,
    video_id: str,
    share: str,
    relpath: str,
    segment: dict[str, Any],
) -> dict[str, Any]:
    """构建写入 Chroma transcripts collection 的文档条目规范。"""
    seg_id = segment["segment_id"]
    doc_id = f"{video_key}_asr_{seg_id:04d}"
    text = segment["text"]
    start = float(segment["start"])
    end = float(segment["end"])
    time_fmt = segment["time_formatted"]

    metadata = {
        "video_key": video_key,
        "video_id": video_id,
        "segment_id": seg_id,
        "share": share,
        "relpath": relpath,
        "time": start,
        "start": start,
        "end": end,
        "time_formatted": time_fmt,
        "text": text,
        "media_type": "video",
        "match_type": "transcript",
    }
    return {
        "id": doc_id,
        "document": text,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="06c: 视频音频台词 ASR 转写与索引管线")
    parser.add_argument("--db", default=DEFAULT_DB, help="Chroma 数据库路径")
    parser.add_argument("--state", default=DEFAULT_STATE, help="ASR 断点状态文件路径")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection 集合名")
    parser.add_argument("--scratch-dir", default=DEFAULT_SCRATCH_DIR, help="抽取音频暂存目录")
    parser.add_argument("--videos-dir", help="指定单独处理的本地视频目录（如 videos_sample）")
    parser.add_argument("--roots", nargs="*", help="NAS 挂载点（默认取 site.json 的 shares）")
    parser.add_argument("--include-dir", action="append", default=[],
                        help="仅处理 relpath 含该子串的视频（可重复，语义同 06）")
    parser.add_argument("--exclude-dir", action="append", default=[],
                        help="跳过 relpath 含该子串的视频（可重复，语义同 06）")
    parser.add_argument("--limit", type=int, default=0, help="处理视频数量限制（0 为不限）")
    parser.add_argument("--dry-run", action="store_true", help="干跑，仅探测统计不写盘")
    parser.add_argument("--language", default="zh", help="ASR 目标语种（默认 zh）")
    args = parser.parse_args()

    state_file = Path(args.state)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_data: dict[str, Any] = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        except Exception as e:
            print(f"[warn] 读取 state 失败，将新建: {e}")

    # 收集待处理视频
    if args.videos_dir:
        video_list = discover_local_videos(Path(args.videos_dir))
    else:
        roots = list(args.roots) if args.roots else _default_roots()
        video_list = discover_nas_videos(roots, args.include_dir, args.exclude_dir)

    print(f"[plan] 扫描到视频总数: {len(video_list)}")
    todo_list = []
    for share, rel, vpath in video_list:
        vkey = compute_video_key(share, rel)
        if vkey in state_data and state_data[vkey].get("status") in ("done", "no_audio", "skipped"):
            continue
        todo_list.append((vkey, share, rel, vpath))

    print(f"[plan] 待处理视频数: {len(todo_list)}")
    if args.limit > 0:
        todo_list = todo_list[:args.limit]
        print(f"[plan] 应用 --limit {args.limit}，实际处理: {len(todo_list)}")

    if args.dry_run:
        print("[dry-run] 干跑完成，退出。")
        return

    if not todo_list:
        print("[info] 无待处理素材，全部已是最新。")
        return

    # 初始化转写器与向量编码器
    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    import chromadb
    import importlib.util
    spec = importlib.util.spec_from_file_location("embed_index", str(SCRIPTS / "02_embed_index.py"))
    assert spec and spec.loader
    embed_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embed_mod)

    encoder = embed_mod.CnClipEncoder()
    chroma_client = chromadb.PersistentClient(path=args.db)
    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    transcriber = AsrTranscriber(language=args.language)

    consecutive_fails = 0
    t_start = time.time()

    for idx, (vkey, share, rel, vpath) in enumerate(todo_list, start=1):
        stem = vpath.stem
        wav_path = scratch_dir / f"{vkey}.wav"
        print(f"[{idx}/{len(todo_list)}] 处理视频: {stem} (key={vkey})")

        try:
            # 1. 检查音轨
            has_audio = check_audio_stream(vpath)
            if not has_audio:
                state_data[vkey] = {
                    "status": "no_audio",
                    "video_id": stem,
                    "share": share,
                    "relpath": rel,
                    "segments_count": 0,
                    "updated_at": int(time.time()),
                }
                print("  ↳ 无音轨，标记跳过。")
                consecutive_fails = 0
                continue

            # 2. 提取 WAV
            ok = extract_audio_wav(vpath, wav_path)
            if not ok:
                raise RuntimeError("ffmpeg 音频抽取失败")

            # 3. 转写
            segments = transcriber.transcribe(wav_path)
            if not segments:
                state_data[vkey] = {
                    "status": "done",
                    "video_id": stem,
                    "share": share,
                    "relpath": rel,
                    "segments_count": 0,
                    "updated_at": int(time.time()),
                }
                print("  ↳ 未检测到有效语音台词。")
                consecutive_fails = 0
                continue

            # 4. 向量化与写入 Chroma
            texts = [seg["text"] for seg in segments]
            embeddings = encoder.encode_texts(texts)

            docs = []
            metas = []
            ids = []
            for seg in segments:
                entry = build_chroma_transcript_entry(vkey, stem, share, rel, seg)
                ids.append(entry["id"])
                docs.append(entry["document"])
                metas.append(entry["metadata"])

            collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=docs,
                metadatas=metas,
            )

            state_data[vkey] = {
                "status": "done",
                "video_id": stem,
                "share": share,
                "relpath": rel,
                "segments_count": len(segments),
                "updated_at": int(time.time()),
            }
            print(f"  ↳ 转写并索引完成: {len(segments)} 条台词片段。")
            consecutive_fails = 0

        except Exception as e:
            consecutive_fails += 1
            print(f"  ↳ 失败: {e} (连续失败次数: {consecutive_fails})")
            state_data[vkey] = {
                "status": "failed",
                "error": str(e),
                "updated_at": int(time.time()),
            }
            if consecutive_fails >= MAX_CONSECUTIVE_FAILURES:
                print(f"[FUSE] 连续 {MAX_CONSECUTIVE_FAILURES} 次失败，触发熔断停机。")
                break
        finally:
            if wav_path.exists():
                wav_path.unlink(missing_ok=True)

        # 周期刷盘状态
        if idx % 10 == 0 or idx == len(todo_list):
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)

    # 结束全量落盘
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"[done] ASR 索引管线执行完毕，耗时 {elapsed:.1f} 秒。")


if __name__ == "__main__":
    main()
