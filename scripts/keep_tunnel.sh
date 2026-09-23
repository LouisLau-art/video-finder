#!/usr/bin/env bash
# 保持本地 FastAPI 检索服务与腾讯云 SSH 反向隧道长生不老

ROOT_DIR="/home/louis/Projects/github/video-finder-prototype"

# 1. 检查并启动 05_api.py
if ! pgrep -f "scripts/05_api.py" > /dev/null; then
    echo "[$(date)] 启动 05_api.py..."
    nohup /usr/bin/uv run python "$ROOT_DIR/scripts/05_api.py" --port 8000 > /tmp/05_api.log 2>&1 &
fi

# 2. 检查并维持 SSH 隧道 (转发腾讯云 18000 -> 本地 8000)
if ! pgrep -f "ssh.*18000:127.0.0.1:8000" > /dev/null; then
    echo "[$(date)] 建立 SSH 反向隧道..."
    nohup sshpass -p '***REMOVED-SECRET***' ssh -o StrictHostKeyChecking=no -N -R 18000:127.0.0.1:8000 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        root@203.0.113.10 > /tmp/ssh_tunnel.log 2>&1 &
fi
