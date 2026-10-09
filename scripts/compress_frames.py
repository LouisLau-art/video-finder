#!/usr/bin/env python3
"""compress_frames.py — 抽帧目录 JPEG 关键帧批量重编码压缩到 480p 级别（原地替换）。

背景：data/local-runtime/frames_full 下 17 万+ 帧为平铺 .jpg（约 43GB），
含大量 4K/高分原图。scripts/05_api.py 直接按文件名经 /api/frames/{filename}
提供，Chroma 元数据与 state.json 里存的 frame_path 也是 .jpg 文件名。
因此本脚本【硬性约束】：保持原文件名不变、原地 os.replace 原子替换，
只改文件内容与分辨率，不改扩展名、不改名、不动任何元数据/数据库。

幂等/断点续跑：以“长边是否 ≤ --max-long-edge”为唯一判定，无需清单文件。
长边已达标直接跳过；重跑只会处理尚未达标的残帧。

缩放：只缩不放，短边按比例自动取偶数（-2），横屏/竖屏分别构造 scale 表达式，
    长边恰好收敛到目标值（默认 854，对应 480p 级别）。
编码：ffmpeg mjpeg，-q:v 控制质量（mjpeg 推荐区间 2-5，默认 5 体积优先），
    -an 去音轨、-map_metadata -1 去元数据。
并发：ThreadPoolExecutor（IO + ffmpeg 子进程混合，线程即可），默认 3 worker；
    每个任务用 pid+线程id+uuid 生成同目录唯一临时名，杜绝并发互踩。

用法:
    .venv/bin/python scripts/compress_frames.py --dry-run            # 只统计不写盘
    .venv/bin/python scripts/compress_frames.py --limit 5 \\         # 小样验证
        --frames-dir /tmp/opencode/frames_smoke
    .venv/bin/python scripts/compress_frames.py                      # 全量原地压缩
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_FRAMES_DIR = "data/local-runtime/frames_full"
DEFAULT_MAX_LONG_EDGE = 854  # 长边目标值（854x480 ≈ 480p）
DEFAULT_QUALITY = 5          # ffmpeg mjpeg -q:v（2-5，越小质量越高）
DEFAULT_WORKERS = 3
PROGRESS_EVERY = 200         # 压缩阶段每完成多少个打印一行
SCAN_PROGRESS_EVERY = 5000   # 扫描分类阶段进度间隔（17 万帧时避免静默）
DRY_RUN_SAMPLE = 8           # dry-run 估算压缩后均大的抽样数
TMP_SUFFIX = ".tmp.jpg"

# ffprobe/ffmpeg 探测/编码的超时秒数（帧图很小，正常远用不到这么久）
PROBE_TIMEOUT = 60
ENCODE_TIMEOUT = 180


# ---------- 环境检查 ----------

def require_tools() -> tuple[str, str]:
    """检查 ffmpeg/ffprobe 是否在 PATH；缺失给出清晰错误并退出。"""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    missing = [n for n, p in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not p]
    if missing:
        print(f"[error] 缺少系统命令: {' '.join(missing)}；请先安装 ffmpeg"
                "（Arch: sudo pacman -S ffmpeg）。", file=sys.stderr)
        raise SystemExit(1)
    return str(ffmpeg), str(ffprobe)


# ---------- ffprobe / ffmpeg ----------

def probe_size(path: Path, ffprobe: str) -> tuple[int, int]:
    """用 ffprobe 读帧宽高；失败抛异常，由调用方计入失败并保留原图。"""
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=PROBE_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe rc={proc.returncode}: {proc.stderr.strip()[-200:]}")
    parts = proc.stdout.strip().split(",")
    if len(parts) < 2:
        raise RuntimeError(f"ffprobe 输出无法解析: {proc.stdout.strip()!r}")
    return int(parts[0]), int(parts[1])


def scale_filter(width: int, height: int, max_edge: int) -> str:
    """构造“只缩不放、长边 ≤ max_edge、短边偶数”的 scale 表达式。

    横屏（含方形）：宽=min(目标,原宽)，高=-2 按比例取偶数；
    竖屏：高=min(目标,原高)，宽=-2。
    表达式内逗号在 ffmpeg filter 语法里需用反斜杠转义。
    """
    if width >= height:
        return f"scale=min({max_edge}\\,iw):-2"
    return f"scale=-2:min({max_edge}\\,ih)"


def tmp_path_for(dst: Path) -> Path:
    """同目录唯一临时名：pid + 线程标识 + uuid，保证并发任务互不冲突。"""
    return dst.with_name(
        f".{os.getpid()}.{threading.get_ident() & 0xffffff:06x}."
        f"{uuid.uuid4().hex[:8]}{TMP_SUFFIX}")


# ---------- 任务函数（线程内执行） ----------

def classify_file(path: Path, max_edge: int, ffprobe: str) -> dict:
    """探测单帧并分类：pending（待压）/ skipped（已达标）/ failed（探测失败）。"""
    try:
        size = path.stat().st_size
        width, height = probe_size(path, ffprobe)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
        return {"path": str(path), "name": path.name, "status": "failed",
                "error": f"{type(e).__name__}: {e}"}
    status = "pending" if max(width, height) > max_edge else "skipped"
    return {"path": str(path), "name": path.name, "size": size,
            "width": width, "height": height, "status": status}


def compress_one(item: dict, max_edge: int, quality: int, ffmpeg: str) -> dict:
    """重编码单帧到同目录临时文件，成功后 os.replace 原子替换；失败删临时文件、保留原图。"""
    path = Path(item["path"])
    size_before = int(item["size"])
    tmp = tmp_path_for(path)
    vf = scale_filter(int(item["width"]), int(item["height"]), max_edge)
    try:
        proc = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(path), "-vf", vf, "-an", "-map_metadata", "-1",
             "-q:v", str(quality), str(tmp)],
            capture_output=True, text=True, timeout=ENCODE_TIMEOUT)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            return {"name": path.name, "status": "failed",
                    "error": f"ffmpeg rc={proc.returncode}: {proc.stderr.strip()[-200:]}"}
        if not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return {"name": path.name, "status": "failed",
                    "error": "ffmpeg 成功但输出文件缺失/为空"}
        size_after = tmp.stat().st_size
        os.replace(tmp, path)  # 同目录原子替换：文件名不变，仅内容/分辨率改变
        return {"name": path.name, "status": "compressed",
                "size_before": size_before, "size_after": size_after}
    except (subprocess.SubprocessError, OSError) as e:
        tmp.unlink(missing_ok=True)
        return {"name": path.name, "status": "failed",
                "error": f"{type(e).__name__}: {e}"}


def encode_sample_size(item: dict, max_edge: int, quality: int, ffmpeg: str) -> int:
    """dry-run 专用：编码到 stdout 管道（不落盘），返回压缩后字节数。"""
    vf = scale_filter(int(item["width"]), int(item["height"]), max_edge)
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", item["path"], "-vf", vf, "-an", "-map_metadata", "-1",
         "-q:v", str(quality), "-f", "mjpeg", "pipe:1"],
        capture_output=True, timeout=ENCODE_TIMEOUT)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg rc={proc.returncode}: {proc.stderr.decode('utf-8', 'replace').strip()[-200:]}")
    return len(proc.stdout)


# ---------- 主流程 ----------

def fmt_size(n: int) -> str:
    """字节数按合适单位格式化。"""
    mb = n / 1024 / 1024
    if mb < 1024:
        return f"{mb:.1f}MB"
    return f"{mb / 1024:.2f}GB"


def clean_stale_tmp(frames_dir: Path) -> int:
    """清理上次崩溃可能残留的隐藏临时文件（.<pid>...tmp.jpg），返回清理数。"""
    n = 0
    try:
        names = os.listdir(frames_dir)
    except OSError:
        return 0
    for name in names:
        if name.startswith(".") and name.endswith(TMP_SUFFIX):
            try:
                (frames_dir / name).unlink()
                n += 1
            except OSError:
                pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser(
        description="抽帧目录 JPEG 关键帧批量重编码到 480p 级别（原地替换，文件名不变）")
    ap.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR,
                    help="帧目录（平铺 .jpg，无子目录），默认 %(default)s")
    ap.add_argument("--max-long-edge", type=int, default=DEFAULT_MAX_LONG_EDGE,
                    help="长边目标像素，只缩不放，默认 %(default)s")
    ap.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                    help="ffmpeg mjpeg -q:v，推荐区间 2-5（越小质量越高），默认 %(default)s")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="并发线程数，范围 1-8，默认 %(default)s")
    ap.add_argument("--limit", type=int, default=0,
                    help="只压前 N 个待压文件（按文件名排序，冒烟用），默认不限")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计待压/跳过数量并抽样估算压缩后大小，不写任何文件")
    args = ap.parse_args()

    # 参数校验：质量取 mjpeg 推荐区间，越界直接拒绝（避免静默产出非预期画质）
    if not 2 <= args.quality <= 5:
        ap.error("--quality 仅支持 ffmpeg mjpeg 推荐区间 2-5（值越小质量越高）")
    if not 1 <= args.workers <= 8:
        ap.error("--workers 允许范围 1-8")
    if args.max_long_edge < 16:
        ap.error("--max-long-edge 过小（至少 16 像素）")
    args.limit = max(int(args.limit), 0)

    ffmpeg, ffprobe = require_tools()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        print(f"[error] 帧目录不存在或不是目录: {frames_dir}", file=sys.stderr)
        return 1

    # 平铺枚举：只认 .jpg（大小写不敏感），忽略隐藏文件（含本脚本临时文件）
    names = sorted(
        n for n in os.listdir(frames_dir)
        if n.lower().endswith(".jpg") and not n.startswith("."))
    if not names:
        print(f"[warn] 目录下没有 .jpg 文件: {frames_dir}", file=sys.stderr)
        return 1
    print(f"[scan] 目录 {frames_dir} 共 {len(names)} 个 .jpg，开始探测分辨率…", flush=True)

    t_start = time.time()

    # 阶段一：并发探测分辨率并分类（幂等判定依据，不需要清单文件）
    pending: list[dict] = []
    n_skip = 0
    n_fail = 0
    total_bytes = 0
    n_seen = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(classify_file, frames_dir / n,
                               args.max_long_edge, ffprobe) for n in names]
        for fut in futures:
            item = fut.result()
            n_seen += 1
            if item["status"] == "pending":
                pending.append(item)
                total_bytes += int(item["size"])
            elif item["status"] == "skipped":
                n_skip += 1
                total_bytes += int(item["size"])
            else:
                n_fail += 1
                print(f"[warn] 探测失败: {item['name']} -> {item['error']}",
                      file=sys.stderr)
            if n_seen % SCAN_PROGRESS_EVERY == 0:
                print(f"[scan] 已探测 {n_seen}/{len(names)}，待压 {len(pending)}，"
                      f"已达标 {n_skip}，失败 {n_fail}，耗时 {time.time() - t_start:.0f}s",
                      file=sys.stderr, flush=True)

    # 排序保证 --limit / 处理顺序确定（文件名含 video_key，排序代价可接受）
    pending.sort(key=lambda x: x["name"])
    pending_bytes = sum(int(x["size"]) for x in pending)
    print(f"[plan] 待压 {len(pending)} 个（{fmt_size(pending_bytes)}），"
          f"已达标跳过 {n_skip} 个，探测失败 {n_fail} 个；"
          f"扫描耗时 {time.time() - t_start:.0f}s", flush=True)

    if args.dry_run:
        # 抽样把前若干待压文件编码到 stdout 管道估算压缩后均大（不写任何文件）
        if pending:
            samples = pending[:min(DRY_RUN_SAMPLE, len(pending))]
            s_before = 0
            s_after = 0
            with ThreadPoolExecutor(max_workers=min(args.workers, len(samples))) as pool:
                futs = {pool.submit(encode_sample_size, it, args.max_long_edge,
                                    args.quality, ffmpeg): it for it in samples}
                for fut in futs:
                    it = futs[fut]
                    try:
                        s_after += fut.result()
                        s_before += int(it["size"])
                    except Exception as e:  # noqa: BLE001 — 单个抽样失败不致命
                        print(f"[warn] 抽样编码失败 {it['name']}: {e}", file=sys.stderr)
            if s_before > 0:
                ratio = s_after / s_before
                avg_before = s_before / min(len(samples), DRY_RUN_SAMPLE)
                avg_after = s_after / min(len(samples), DRY_RUN_SAMPLE)
                est_after = int(pending_bytes * ratio)
                print(f"[dry-run] 抽样 {len(samples)} 个：平均 {avg_before / 1024:.0f}KB "
                      f"-> {avg_after / 1024:.0f}KB（压缩比 {ratio * 100:.1f}%）")
                print(f"[dry-run] 按抽样比例估算：待压部分 {fmt_size(pending_bytes)} "
                      f"-> 约 {fmt_size(est_after)}，预计节省 "
                      f"{fmt_size(pending_bytes - est_after)}（抽样估算，仅供参考）")
        print(f"[dry-run] 未写任何文件。正式跑可加 --limit 5 先冒烟。")
        return 0

    # 正式跑：先清理上次崩溃残留临时文件
    n_stale = clean_stale_tmp(frames_dir)
    if n_stale:
        print(f"[plan] 已清理残留临时文件 {n_stale} 个")

    todo = pending[:args.limit] if args.limit > 0 else pending
    if not todo:
        print("[done] 无待压文件，全部已达标。")
        return 0
    print(f"[run] 开始压缩 {len(todo)} 个（quality=q:v {args.quality}，"
          f"workers={args.workers}）"
          + (f"，--limit {args.limit} 仅处理前 {len(todo)} 个" if args.limit else ""),
          flush=True)

    # 阶段二：并发重编码，主线程统一计数（无需加锁）
    n_compressed = 0
    n_run_fail = 0
    saved_bytes = 0
    n_done = 0
    t_run = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compress_one, it, args.max_long_edge,
                               args.quality, ffmpeg) for it in todo]
        for fut in futures:
            res = fut.result()
            n_done += 1
            if res["status"] == "compressed":
                n_compressed += 1
                saved_bytes += int(res["size_before"]) - int(res["size_after"])
            else:
                n_run_fail += 1
                print(f"[warn] 压缩失败: {res['name']} -> {res['error']}", file=sys.stderr)
            if n_done % PROGRESS_EVERY == 0 or n_done == len(todo):
                rate = n_done / max(time.time() - t_run, 1e-9)
                print(f"[progress] 已压缩 {n_compressed}（本轮 {n_done}/{len(todo)}），"
                      f"已达标跳过 {n_skip}，失败 {n_fail + n_run_fail}，"
                      f"累计节省 {saved_bytes / 1024 / 1024:.1f}MB，{rate:.1f} 个/秒",
                      flush=True)

    # 汇总：压缩前后总体积按全部已探测帧计（未处理/跳过帧保持原大小）
    bytes_before_all = total_bytes
    bytes_after_all = total_bytes - saved_bytes
    el = time.time() - t_start
    total_fail = n_fail + n_run_fail
    pct = saved_bytes / bytes_before_all * 100 if bytes_before_all else 0.0
    print(f"[summary] 压缩 {n_compressed}，跳过 {n_skip}，失败 {total_fail}"
          + (f"（另有限额未处理 {len(pending) - len(todo)}）" if args.limit else ""))
    print(f"[summary] 总体积 {fmt_size(bytes_before_all)} -> "
          f"{fmt_size(bytes_after_all)}，节省 {fmt_size(saved_bytes)}（{pct:.1f}%）")
    print(f"[summary] 总耗时 {el / 60:.1f} 分；帧文件名未变，元数据/数据库未改动。")
    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
