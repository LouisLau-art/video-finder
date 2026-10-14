#!/usr/bin/env bash
# 等待图片索引服务完成，并自动触发向新服务器增量同步索引与缩略图
set -uo pipefail
cd /home/louis/Projects/github/video-finder-prototype

LOG="data/logs/chain_deploy_images_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/logs

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 启动图片索引守候与自动发布流程 ===" | tee -a "$LOG"

while systemctl show -p ActiveState --value vf-images.service | grep -qE '^(active|activating|deactivating)$'; do
    sleep 60
done

RC=$(systemctl show -p ExecMainStatus --value vf-images.service)
echo "[$(date +'%Y-%m-%d %H:%M:%S')] vf-images.service 结束，退出码: $RC" | tee -a "$LOG"

if [ "$RC" -ne 0 ]; then
    echo "[error] 图片索引未成功结束 (code=$RC)，中止自动部署发布。" | tee -a "$LOG"
    exit "$RC"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始执行增量发布至新服务器 (106.53.225.216)..." | tee -a "$LOG"
export HF_HUB_OFFLINE=1

bash scripts/deploy_to_server.sh --index --images 2>&1 | tee -a "$LOG"
DEPLOY_RC=${PIPESTATUS[0]}

if [ "$DEPLOY_RC" -eq 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 部署发布成功！核对公网接口状态..." | tee -a "$LOG"
    # 公网入口由站点配置决定（仓库不存真实域名/IP，见 docs/site.example.json）
    PUBLIC_BASE="$(python3 scripts/site_config.py get deploy.public_base)"
    if [ -n "$PUBLIC_BASE" ]; then
        HEALTH=$(curl -s "${PUBLIC_BASE}/api/health")
        echo "Health: $HEALTH" | tee -a "$LOG"
    else
        echo "[warn] site.json 缺 deploy.public_base，跳过公网核对" | tee -a "$LOG"
    fi
else
    echo "[error] 部署脚本失败 (code=$DEPLOY_RC)" | tee -a "$LOG"
fi

exit "$DEPLOY_RC"
