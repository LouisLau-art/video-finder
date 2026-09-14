#!/usr/bin/env python3
"""01_extract.py — ffmpeg 按 1fps 抽帧，优先 PySceneDetect 场景切分，每场景留 1-3 中帧，失败回退纯 1fps.

每帧命名: {video_id}_{ss}.jpg  (ss 保留 1 位小数, 如 demo_2.5.jpg)
输出: frames/*.jpg + manifest.jsonl (每行含 video_id, time, frame_path)

用法:
    python scripts/01_extract.py
    python scripts/01_extract.py --input-dir videos_sample --frames-dir frames --manifest frames/manifest.jsonl
    python scripts/01_extract.py --fps 1 --max-per-scene 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageStat

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}

# 黑场/纯色卡过滤阈值（保守取值，宁可漏滤也不误杀暗光正常帧）:
# - MEAN: 纯黑底(含 JPEG 噪声)灰度均值通常 1~4，品牌黑底 logo 卡实测 1.5~6.5；
#   正常夜景/暗光画面因有高光与纹理，实测最低均值约 17。取 8 留出安全边界。
# - STD: 纯色块(含 JPEG 噪声)灰度标准差通常 <4；实测最低的“非纯色”卡(暗底+角标)
#   为 2.5，而带内容的暗光帧最低约 8.9。取 6 位于两者之间且偏保守。
# 任一条件命中即丢弃：均值低=近黑，标准差低=纯色/近纯色。
BLANK_MEAN_THRESHOLD = 8.0
BLANK_STD_THRESHOLD = 6.0


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def get_duration(video: Path) -> float | None:
    """用 ffprobe 读时长(秒)，失败返回 None."""
    p = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ])
    try:
        return float(p.stdout.strip())
    except Exception:
        return None


def extract_frame(video: Path, t: float, out: Path) -> bool:
    """在 t 秒处抽 1 帧. -ss 放 -i 前面, 速度快, 精度对原型足够."""
    out.parent.mkdir(parents=True, exist_ok=True)
    p = run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{max(t, 0):.2f}",
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", "2",
        str(out),
    ])
    return out.exists() and out.stat().st_size > 0


def detect_scenes(video: Path) -> list[tuple[float, float]] | None:
    """优先用 PySceneDetect ContentDetector 切场景.

    返回 [(start_sec, end_sec), ...]，任何失败返回 None 让调用方回退 1fps.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except Exception as e:
        print(f"[scene] scenedetect 不可用，回退 1fps: {e}")
        return None
    try:
        vid = open_video(str(video))
        sm = SceneManager()
        sm.add_detector(ContentDetector())
        sm.detect_scenes(video=vid, show_progress=False)
        scenes = sm.get_scene_list()
        if not scenes:
            print("[scene] 未切出场景，回退 1fps")
            return None
        out: list[tuple[float, float]] = []
        for s, e in scenes:
            out.append((s.get_seconds(), e.get_seconds()))
        print(f"[scene] {video.name}: 切出 {len(out)} 个场景")
        return out
    except Exception as e:
        print(f"[scene] 切分失败，回退 1fps: {e}")
        return None


def timestamps_for_scene(start: float, end: float, duration: float | None, max_per_scene: int = 3) -> list[float]:
    """每场景留 1-3 中帧: <2s 取中点1帧, 2-5s 取2帧, >5s 取3帧(均分)."""
    dur = max(end - start, 0.0)
    if dur <= 0:
        return [round(start, 1)]
    if dur < 2.0 or max_per_scene <= 1:
        ts = [(start + end) / 2.0]
    elif dur <= 5.0 or max_per_scene == 2:
        ts = [start + dur * 1 / 3, start + dur * 2 / 3]
    else:
        ts = [start + dur * 0.25, start + dur * 0.5, start + dur * 0.75]
    if duration and duration > 0:
        ts = [min(max(t, 0.0), max(duration - 0.1, 0.0)) for t in ts]
    return [round(t, 1) for t in ts]


def fallback_timestamps(duration: float | None, fps: float = 1.0) -> list[float]:
    """纯 1fps 回退: t = 0,1,2,..."""
    if not duration or duration <= 0:
        return [0.0]
    n = max(int(duration * fps), 1)
    return [round(i / fps, 1) for i in range(n)]


def frame_gray_stats(img: Image.Image) -> tuple[float, float]:
    """转灰度后的 (mean, stddev)，用于黑场/纯色判定."""
    st = ImageStat.Stat(img.convert("L"))
    return float(st.mean[0]), float(st.stddev[0])


def is_blank_frame(
    img: Image.Image,
    mean_threshold: float = BLANK_MEAN_THRESHOLD,
    std_threshold: float = BLANK_STD_THRESHOLD,
) -> bool:
    """纯函数: 判断一帧是否为黑场/纯色卡（应丢弃）.

    灰度均值 < mean_threshold  = 近黑（片头尾黑场、黑底 logo 卡）
    灰度标准差 < std_threshold = 纯色/近纯色（纯色卡、纯字卡）
    阈值见模块顶部常量的取值依据（保守，避免误杀夜景/暗光正常帧）。
    """
    mean, std = frame_gray_stats(img)
    return mean < mean_threshold or std < std_threshold


def process_video(video: Path, frames_dir: Path, fps: float, max_per_scene: int) -> list[dict]:
    video_id = video.stem
    duration = get_duration(video)
    print(f"[video] {video.name} duration={duration}")

    scenes = detect_scenes(video)
    if scenes:
        ts: list[float] = []
        for s, e in scenes:
            ts.extend(timestamps_for_scene(s, e, duration, max_per_scene))
    else:
        ts = fallback_timestamps(duration, fps)
        print(f"[fallback] {video.name}: 纯1fps, {len(ts)} 帧")

    # 去重排序，避免同秒重复抽帧
    ts = sorted(set(ts))
    rows: list[dict] = []
    filtered = 0
    for t in ts:
        fname = f"{video_id}_{t:.1f}.jpg"
        out = frames_dir / fname
        ok = extract_frame(video, t, out)
        if not ok:
            print(f"[warn] 抽帧失败: {video.name} @ {t}s", file=sys.stderr)
            continue
        # 落盘前判定: 黑场/纯色卡直接删文件，不写 manifest
        try:
            with Image.open(out) as im:
                blank = is_blank_frame(im)
        except Exception as e:  # noqa: BLE001 — 读图失败宁可保留，不误杀
            print(f"[warn] 读帧失败保留 {fname}: {e}", file=sys.stderr)
            blank = False
        if blank:
            out.unlink(missing_ok=True)
            filtered += 1
            print(f"[filter] 丢弃黑场/纯色帧: {fname}", file=sys.stderr)
            continue
        rows.append({
            "video_id": video_id,
            "time": float(t),
            "frame_path": str(out.as_posix()),
            "video_path": str(video.as_posix()),
        })
    if filtered:
        print(
            f"[filter] {video.name}: 过滤 {filtered}/{len(ts)} 帧(黑场/纯色卡)",
            file=sys.stderr,
        )
    print(f"[done] {video.name}: {len(rows)}/{len(ts)} 帧")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="ffmpeg 1fps抽帧(优先场景切分,失败回退1fps)")
    ap.add_argument("--input-dir", default="videos_sample")
    ap.add_argument("--frames-dir", default="frames")
    ap.add_argument("--manifest", default="frames/manifest.jsonl")
    ap.add_argument("--fps", type=float, default=1.0, help="回退模式帧率")
    ap.add_argument("--max-per-scene", type=int, default=3, help="每场景最多保留帧数(1-3)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    frames_dir = Path(args.frames_dir)
    manifest = Path(args.manifest)
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"输入目录不存在: {in_dir}", file=sys.stderr)
        return 1
    videos = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file())
    if not videos:
        print(f"{in_dir} 里没有视频文件(支持 {sorted(VIDEO_EXTS)})，先把 mp4 放进去。")
        manifest.write_text("", encoding="utf-8")
        return 0

    all_rows: list[dict] = []
    for v in videos:
        all_rows.extend(process_video(v, frames_dir, args.fps, args.max_per_scene))

    with manifest.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[manifest] {len(all_rows)} 帧 -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
