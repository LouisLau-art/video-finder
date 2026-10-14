#!/usr/bin/env bash
# 图片全量索引：0摄影图（两共享）。幂等可续跑，用 --include-dir 圈定范围。
# 由 systemd 单元 vf-images.service 调用；也可手工执行。
set -uo pipefail
cd /home/louis/Projects/github/video-finder-prototype

LOG="data/logs/images_full_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/logs

echo "[run_images_index] 启动，日志: $LOG" | tee -a "$LOG"
echo "[run_images_index] 范围: 0摄影图（两共享），幂等" | tee -a "$LOG"

export HF_HUB_OFFLINE=1

.venv/bin/python scripts/06b_index_images.py \
  --include-dir "0摄影图" \
  --workers 2 \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "[run_images_index] 索引结束，exit=$RC" | tee -a "$LOG"
exit "$RC"
