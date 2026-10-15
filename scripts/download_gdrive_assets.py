#!/usr/bin/env python3
"""
download_gdrive_assets.py — 批量拉取张顺仪文档中的 5 个海外 Google Drive 公开相册。
保存至本地目录（默认 /home/louis/Pictures/海外活动素材_GoogleDrive/）。
"""
import os
import sys
import subprocess
from pathlib import Path

TARGET_DIR = Path("/home/louis/Pictures/海外活动素材_GoogleDrive")
PROXY = "http://127.0.0.1:7897"

FOLDERS = [
    ("25澳洲_Photos-Bra", "https://drive.google.com/drive/folders/1S6PRpYkX5n4W_49bVl7zk4J056LEfwKm"),
    ("25澳洲_QK-Melb-Spring-Summer", "https://drive.google.com/drive/folders/17R83G-PkIcR375GzYy8Y8EcV4kKJanJF"),
    ("26GCM_黄金海岸", "https://drive.google.com/drive/folders/1q9ZjkncL6kr8Su6_479WkkIRCgmMAKQf"),
    ("日本Parkrun_丸一女士", "https://drive.google.com/drive/folders/1i_qS3hSQ7_Zc_CBK6Btnb9zxtnLbzgc6"),
    ("26悉尼马_Models", "https://drive.google.com/drive/folders/1eSUND4cbudEOnmT1dqcGFNjQtmxDJCXE"),
]

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    python_bin = sys.executable

    print(f"=== 开始下载 Google Drive 海外相册到: {TARGET_DIR} ===")
    for name, url in FOLDERS:
        out_path = TARGET_DIR / name
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> 正在下载: {name} ...")
        cmd = [
            python_bin, "-m", "gdown",
            "--folder", url,
            "-O", str(out_path),
            "--proxy", PROXY,
            "--continue"
        ]
        try:
            subprocess.run(cmd, check=True)
            file_count = len(list(out_path.glob("*.*")))
            print(f">>> {name} 下载完成，当前文件数: {file_count}")
        except Exception as e:
            print(f">>> {name} 下载失败或部分完成: {e}")

    print("\n=== 全部 Google Drive 下载任务执行完毕 ===")

if __name__ == "__main__":
    main()
