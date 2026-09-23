#!/usr/bin/env bash
# 保持本地 FastAPI 检索服务与云主机 SSH 反向隧道长生不老
#
# 隧道参数（user/host/remote_port/local_port）来自仓库外的站点配置
# （`python3 scripts/site_config.py tunnel`），本文件不存任何真实地址。
# 站点配置示例见 docs/site.example.json。
# 凭证: 云主机密码存于 ${HOME}/.config/video-finder/ssh_auth (600)，
#       绝不写进仓库。缺失时请先创建该文件。
set -u

ROOT_DIR="/home/louis/Projects/github/video-finder-prototype"
AUTH_FILE="${HOME}/.config/video-finder/ssh_auth"
SITE_CLI="python3 $ROOT_DIR/scripts/site_config.py"

# 1. 检查并启动 05_api.py
if ! pgrep -f "scripts/05_api.py" > /dev/null; then
    echo "[$(date)] 启动 05_api.py..."
    nohup "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/05_api.py" --port 8000 > /tmp/05_api.log 2>&1 &
fi

# 2. 从站点配置读隧道参数
TUNNEL_USER="$($SITE_CLI get tunnel.user)"
TUNNEL_HOST="$($SITE_CLI get tunnel.host)"
REMOTE_PORT="$($SITE_CLI get tunnel.remote_port)"
LOCAL_PORT="$($SITE_CLI get tunnel.local_port)"
if [[ -z "$TUNNEL_USER" || -z "$TUNNEL_HOST" || -z "$REMOTE_PORT" || -z "$LOCAL_PORT" ]]; then
    echo "[$(date)] 站点隧道配置缺失（user/host/ports），跳过隧道建立（参见 docs/site.example.json）" >&2
    exit 1
fi

# 3. 检查并维持 SSH 隧道 (转发云主机 REMOTE_PORT -> 本地 LOCAL_PORT)
if ! pgrep -f "ssh.*${REMOTE_PORT}:127.0.0.1:${LOCAL_PORT}" > /dev/null; then
    if [ ! -f "$AUTH_FILE" ]; then
        echo "[$(date)] 缺少凭证文件 $AUTH_FILE，跳过隧道建立" >&2
        exit 1
    fi
    echo "[$(date)] 建立 SSH 反向隧道..."
    nohup sshpass -f "$AUTH_FILE" ssh -o StrictHostKeyChecking=no -N -R "${REMOTE_PORT}:127.0.0.1:${LOCAL_PORT}" \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        "${TUNNEL_USER}@${TUNNEL_HOST}" > /tmp/ssh_tunnel.log 2>&1 &
fi
