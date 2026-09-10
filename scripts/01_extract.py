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

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}


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
    for t in ts:
        fname = f"{video_id}_{t:.1f}.jpg"
        out = frames_dir / fname
        ok = extract_frame(video, t, out)
        if ok:
            rows.append({
                "video_id": video_id,
                "time": float(t),
                "frame_path": str(out.as_posix()),
                "video_path": str(video.as_posix()),
            })
        else:
            print(f"[warn] 抽帧失败: {video.name} @ {t}s", file=sys.stderr)
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
