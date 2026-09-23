#!/usr/bin/env bash
# 保持本地 FastAPI 检索服务与腾讯云 SSH 反向隧道长生不老
#
# 凭证: 腾讯云 root 密码存于 ${HOME}/.config/video-finder/ssh_auth (600)，
#       绝不写进仓库。缺失时请先创建该文件。
set -u

ROOT_DIR="/home/louis/Projects/github/video-finder-prototype"
AUTH_FILE="${HOME}/.config/video-finder/ssh_auth"
TUNNEL_HOST="203.0.113.10"

# 1. 检查并启动 05_api.py
if ! pgrep -f "scripts/05_api.py" > /dev/null; then
    echo "[$(date)] 启动 05_api.py..."
    nohup /usr/bin/uv run python "$ROOT_DIR/scripts/05_api.py" --port 8000 > /tmp/05_api.log 2>&1 &
fi

# 2. 检查并维持 SSH 隧道 (转发腾讯云 18000 -> 本地 8000)
if ! pgrep -f "ssh.*18000:127.0.0.1:8000" > /dev/null; then
    if [ ! -f "$AUTH_FILE" ]; then
        echo "[$(date)] 缺少凭证文件 $AUTH_FILE，跳过隧道建立" >&2
        exit 1
    fi
    echo "[$(date)] 建立 SSH 反向隧道..."
    nohup sshpass -f "$AUTH_FILE" ssh -o StrictHostKeyChecking=no -N -R 18000:127.0.0.1:8000 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        "root@${TUNNEL_HOST}" > /tmp/ssh_tunnel.log 2>&1 &
fi
