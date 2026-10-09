#!/usr/bin/env bash
# 把检索服务发布到公网服务器（腾讯云老数仓），使服务脱离工位机独立运行。
#
# 部署形态：
#   服务器 /opt/videofinder/
#     ├── app/          <- 本仓库代码（不含 data/、.git、.venv）
#     ├── index_full/   <- 向量库（Chroma，含 encoder.json）
#     ├── frames_full/  <- 帧缩略图（前端卡片封面）
#     ├── hf-cache/     <- CN-CLIP 模型权重（HF 缓存布局）
#     └── bin/ lib/     <- uv 建的独立 Python 3.11 虚拟环境
#   服务：systemd unit `videofinder.service`，监听 127.0.0.1:18000，
#         Nginx 的 /api/ 反代到该端口（无需隧道）。
#
# 站点私有参数（host/user/远端目录/端口）一律读仓库外的 site.json
#   `python3 scripts/site_config.py get deploy.host` 等。
# 凭证读 ${HOME}/.config/video-finder/ssh_auth（600）。
#
# 用法：
#   scripts/deploy_to_server.sh              # 同步代码 + 重启服务（最快，日常用）
#   scripts/deploy_to_server.sh --index      # 额外同步向量库
#   scripts/deploy_to_server.sh --frames     # 额外同步帧缩略图
#   scripts/deploy_to_server.sh --all        # 三样全同步
#   scripts/deploy_to_server.sh --dry-run    # 只看会传什么，不落盘
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTH_FILE="${HOME}/.config/video-finder/ssh_auth"
SITE_CLI=(python3 "${ROOT_DIR}/scripts/site_config.py")

DO_INDEX=0
DO_FRAMES=0
DRY=""
for arg in "$@"; do
    case "$arg" in
        --index) DO_INDEX=1 ;;
        --frames) DO_FRAMES=1 ;;
        --all) DO_INDEX=1; DO_FRAMES=1 ;;
        --dry-run) DRY="--dry-run" ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "[error] 未知参数: $arg" >&2; exit 2 ;;
    esac
done

# ---- 读站点配置 ----
cfg() { "${SITE_CLI[@]}" get "$1"; }
DEPLOY_USER="$(cfg deploy.user)"
DEPLOY_HOST="$(cfg deploy.host)"
REMOTE_BASE="$(cfg deploy.remote_base)"
APP_SERVICE="$(cfg deploy.service)"
[ -z "$DEPLOY_USER" ] && DEPLOY_USER="$(cfg tunnel.user)"
[ -z "$DEPLOY_HOST" ] && DEPLOY_HOST="$(cfg tunnel.host)"
[ -z "$REMOTE_BASE" ] && REMOTE_BASE="/opt/videofinder"
[ -z "$APP_SERVICE" ] && APP_SERVICE="videofinder"

if [[ -z "$DEPLOY_HOST" ]]; then
    echo "[error] site.json 缺 deploy.host / tunnel.host，无法确定目标服务器" >&2
    echo "        参见 docs/site.example.json" >&2
    exit 1
fi
if [[ ! -f "$AUTH_FILE" ]]; then
    echo "[error] 缺少凭证文件 $AUTH_FILE" >&2
    exit 1
fi

SSH_CMD="sshpass -f '$AUTH_FILE' ssh -o StrictHostKeyChecking=no"
TARGET="${DEPLOY_USER}@${DEPLOY_HOST}"
echo "[deploy] 目标 ${TARGET}:${REMOTE_BASE}  服务=${APP_SERVICE}  dry=${DRY:-no}"

rsync_to() {  # rsync_to <src> <dst_rel> [extra args...]
    local src="$1"; shift
    local dst="$1"; shift
    rsync -a --info=progress2 --partial $DRY \
        -e "sshpass -f '${AUTH_FILE}' ssh -o StrictHostKeyChecking=no" \
        "$@" "${src}" "${TARGET}:${REMOTE_BASE}/${dst}/"
}

# ---- 1. 代码（必做）----
echo "[deploy] 同步 app/ ..."
rsync_to "${ROOT_DIR}/" "app" \
    --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    --exclude '.pytest_cache' --exclude '*.egg-info' \
    --exclude 'data' --exclude 'chroma_db' --exclude 'frames' \
    --exclude '.scratch' --exclude 'eval_local' --exclude '.cortexkit' \
    --exclude '.opencode' --exclude 'videos_sample' --exclude 'docs/site.json'

# ---- 2. 向量库（可选）----
if [[ "$DO_INDEX" == "1" ]]; then
    echo "[deploy] 同步 index_full/ ..."
    rsync_to "${ROOT_DIR}/data/local-runtime/index_full/" "index_full"
fi

# ---- 3. 帧缩略图（可选）----
if [[ "$DO_FRAMES" == "1" ]]; then
    echo "[deploy] 同步 frames_full/ ..."
    rsync_to "${ROOT_DIR}/data/local-runtime/frames_full/" "frames_full" \
        --include '*/' --include '*.jpg' --exclude '*'
fi

# ---- 4. 站点配置（NAS 路径映射等，供服务拼 NAS 链接）----
if [[ -f "$HOME/.config/video-finder/site.json" ]]; then
    echo "[deploy] 同步 site.json ..."
    if [[ -z "$DRY" ]]; then
        sshpass -f "$AUTH_FILE" ssh -o StrictHostKeyChecking=no "$TARGET" \
            'mkdir -p /root/.config/video-finder'
        sshpass -f "$AUTH_FILE" scp -o StrictHostKeyChecking=no \
            "$HOME/.config/video-finder/site.json" \
            "$TARGET:/root/.config/video-finder/site.json"
    fi
fi

# ---- 5. 重启服务并健康检查 ----
if [[ -n "$DRY" ]]; then
    echo "[deploy] dry-run 结束，未改动服务器。"
    exit 0
fi

echo "[deploy] 重启 ${APP_SERVICE} ..."
# shellcheck disable=SC2029
sshpass -f "$AUTH_FILE" ssh -o StrictHostKeyChecking=no "$TARGET" \
    "systemctl restart '${APP_SERVICE}' && sleep 5 && systemctl is-active '${APP_SERVICE}'"

echo "[deploy] 等待服务就绪（模型加载约 10-60s）..."
for i in $(seq 1 12); do
    sleep 5
    code="$(sshpass -f "$AUTH_FILE" ssh -o StrictHostKeyChecking=no "$TARGET" \
        "curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:18000/api/health" || true)"
    echo "[deploy]   ${i}/12 health=${code}"
    [[ "$code" == "200" ]] && break
done

if [[ "$code" != "200" ]]; then
    echo "[deploy][error] 服务未就绪，最近日志：" >&2
    sshpass -f "$AUTH_FILE" ssh -o StrictHostKeyChecking=no "$TARGET" \
        "journalctl -u '${APP_SERVICE}' --no-pager -n 20" >&2 || true
    exit 1
fi

PUBLIC_BASE="$(cfg deploy.public_base)"
if [[ -n "$PUBLIC_BASE" ]]; then
    echo "[deploy] 完成。公网入口：${PUBLIC_BASE}/api/health"
else
    echo "[deploy] 完成。服务已在 127.0.0.1:18000 就绪（公网入口由站点配置 deploy.public_base 决定）。"
fi
