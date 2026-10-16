#!/usr/bin/env bash
# run_skipped_and_asr.sh — 串行跑完「30 个 skipped 4K 视频补录」+「全量 ASR 语音台词提取」+「自动发布到新服务器」
#
# 优势：
# 1. 严格串行执行，避免 4K 视频解码与 ASR 音频抽取同时争抢 NAS SMB / 磁盘 / CPU；
# 2. 全程日志落盘至 data/logs/，支持 tail 实时查看；
# 3. 跑完自动触发 deploy_to_server.sh --index，热重载并预热云服务器。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/data/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/chain_skipped_asr_${TS}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 启动串行任务流水线 ===" | tee -a "$MAIN_LOG"

# 阶段一：补录 30 个 skipped 4K 视频（限宽 1280 抽帧）
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [阶段一] 开始补录 30 个 skipped 视频..." | tee -a "$MAIN_LOG"
.venv/bin/python scripts/06_index_all.py --only-skipped --max-frame-width 1280 2>&1 | tee -a "$MAIN_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [阶段一] 30 个 skipped 视频处理完毕！" | tee -a "$MAIN_LOG"

# 阶段二：全量 ASR 语音台词提取
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [阶段二] 开始全量 ASR 语音台词提取..." | tee -a "$MAIN_LOG"
.venv/bin/python scripts/06c_index_asr.py 2>&1 | tee -a "$MAIN_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [阶段二] 全量 ASR 提取完毕！" | tee -a "$MAIN_LOG"

# 阶段三：同步发布到新服务器
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [阶段三] 开始发布更新后的向量库到服务器..." | tee -a "$MAIN_LOG"
bash scripts/deploy_to_server.sh --index 2>&1 | tee -a "$MAIN_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 全流程圆满结束！===" | tee -a "$MAIN_LOG"
