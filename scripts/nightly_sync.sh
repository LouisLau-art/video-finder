#!/usr/bin/env bash
# nightly_sync.sh — 每日夜间自动增量索引 + 发布到云服务器。
#
# 流程:
#   1) 前置检查 NAS 挂载（任一 site.json 配置的挂载点可列目录即可）
#      不可用 -> 记日志退出码 3，不做任何写操作
#   2) flock 独占锁，防止上一晚没跑完又叠一轮
#   3) 增量索引 scripts/06_index_all.py（幂等，自动跳过已索引素材），
#      完整输出 tee 到 data/logs/nightly_<date>.log
#   4) 有新增才发布 scripts/deploy_to_server.sh --all（零新增直接"今日无新增"结束）
#   5) 验收: curl deploy.host 的 /api/health，失败重试 1 次
#   6) 结果原子落盘 data/ops/nightly_status.json
#
# 站点私有参数一律读仓库外的 site.json（服务器地址等敏感值不进本仓库），
# 索引/发布日志含真实素材路径，只落 data/logs/（该目录确认不入库）。
#
# systemd 安装方式（unit 文件在本仓库 deploy/ 下）:
#   sudo cp deploy/videofinder-nightly.{service,timer} /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now videofinder-nightly.timer
#
# 用法:
#   scripts/nightly_sync.sh              # 正常夜间流程
#   scripts/nightly_sync.sh --no-deploy  # 只索引不发布（排障用）
#
# 退出码:
#   0 成功（含"今日无新增"分支）；3 NAS 挂载不可用；其余非零=对应步骤失败
set -euo pipefail

# ---- 工作目录固定为仓库根，后续全部用相对路径 ----
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SITE_CLI=(python3 scripts/site_config.py)

# ---- 开关参数 ----
DO_DEPLOY=1
for arg in "$@"; do
    case "$arg" in
        --no-deploy) DO_DEPLOY=0 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "[nightly][error] 未知参数: $arg" >&2; exit 2 ;;
    esac
done

# ---- 每步打带时间戳的日志，同步进 journal（systemd 下可见） ----
log() { echo "[nightly $(date '+%F %T')] $*"; }

# ---- 结果字段（异常路径也要能写状态文件，见 write_status） ----
INDEXED_NEW=""   # 数字；拿不到统计时写 null
PUBLISHED=false
HEALTH_OK=false

TS="$(date +%Y%m%d)"
LOG_FILE="data/logs/nightly_${TS}.log"
STATUS_FILE="data/ops/nightly_status.json"

mkdir -p data/logs data/ops

# ---- 状态文件原子落盘（tmp + mv），异常退出路径也会调用 ----
write_status() {
    local exit_code="$1" duration="$2"
    python3 - "$STATUS_FILE" "$TS" "$exit_code" "$duration" "$INDEXED_NEW" \
              "$PUBLISHED" "$HEALTH_OK" <<'PYEOF'
import json, os, sys
path, date, code, dur, new, pub, health = sys.argv[1:8]
try:
    new_val = int(new)
except (TypeError, ValueError):
    new_val = None
status = {
    "date": date,
    "exit_code": int(code),
    "indexed_new": new_val,
    "published": pub == "true",
    "health_ok": health == "true",
    "duration_sec": int(float(dur)),
}
tmp = f"{path}.tmp.{os.getpid()}"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(status, f, ensure_ascii=False)
    f.write("\n")
os.replace(tmp, path)  # 原子替换
PYEOF
}

# ---- 0. 开始计时 ----
START_TS="$(date +%s)"

# ---- 1. 前置检查: NAS 挂载可用（任一配置挂载点可列目录即可；便宜操作，5s 超时防挂死） ----
log "步骤1: 检查 NAS 挂载可用性 ..."
NAS_OK=0
while IFS=$'\t' read -r _share mount _name; do
    [ -n "$mount" ] || continue
    if timeout 5 ls "$mount" >/dev/null 2>&1; then
        log "  NAS 挂载可用: $mount"
        NAS_OK=1
        break
    fi
done < <("${SITE_CLI[@]}" shares)

if [ "$NAS_OK" -ne 1 ]; then
    log "[nightly][error] NAS 挂载均不可用（未挂载/无内容/超时），本次不做任何写操作"
    END_TS="$(date +%s)"
    write_status 3 "$((END_TS - START_TS))"
    exit 3
fi

# ---- 2. 独占锁: 防止上一晚没跑完又叠一轮 ----
LOCK_FILE="/tmp/videofinder-nightly.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "[nightly][error] 已有另一轮 nightly 在跑（锁 $LOCK_FILE 被占），本轮退出"
    END_TS="$(date +%s)"
    write_status 3 "$((END_TS - START_TS))"
    exit 3
fi
log "步骤2: 已获取独占锁 $LOCK_FILE"

# ---- 3. 增量索引（整体限时 4h；幂等，已索引素材自动跳过） ----
log "步骤3: 运行增量索引 scripts/06_index_all.py（限时 4h）..."
INDEX_RC=0
timeout 4h .venv/bin/python scripts/06_index_all.py 2>&1 | tee "$LOG_FILE" || INDEX_RC=$?
# 注意: 上面是管道，INDEX_RC 取的是整条管道退出码（tee 正常时即 06 的退出码）
if [ "$INDEX_RC" -ne 0 ]; then
    log "[nightly][error] 增量索引失败（退出码 $INDEX_RC，日志: $LOG_FILE），不发布"
    END_TS="$(date +%s)"
    write_status "$INDEX_RC" "$((END_TS - START_TS))"
    exit "$INDEX_RC"
fi

# ---- 从日志统计新增数: [plan] 行的"待处理 N"最准，退级数 [progress] 行 ----
INDEXED_NEW="$(grep -oE '待处理 [0-9]+' "$LOG_FILE" | head -1 | awk '{print $2}')"
if [ -z "$INDEXED_NEW" ]; then
    INDEXED_NEW="$(grep -c '^\[progress\]' "$LOG_FILE" || true)"
    [ "$INDEXED_NEW" -eq 0 ] && INDEXED_NEW=""
fi
log "步骤3完成: 新增待处理 = ${INDEXED_NEW:-统计失败(将记为 null)}"

# ---- 零新增: 跳过发布与验收，直接记"今日无新增"结束 ----
if [ -z "$INDEXED_NEW" ] || [ "$INDEXED_NEW" -eq 0 ]; then
    log "今日无新增素材，跳过发布与验收，正常结束"
    END_TS="$(date +%s)"
    write_status 0 "$((END_TS - START_TS))"
    exit 0
fi

# ---- 4. 发布（真跑，限时 4h） ----
if [ "$DO_DEPLOY" -eq 1 ]; then
    log "步骤4: 发布 scripts/deploy_to_server.sh --all（限时 4h）..."
    DEPLOY_RC=0
    timeout 4h scripts/deploy_to_server.sh --all 2>&1 | tee -a "$LOG_FILE" || DEPLOY_RC=$?
    if [ "$DEPLOY_RC" -ne 0 ]; then
        log "[nightly][error] 发布失败（退出码 $DEPLOY_RC，日志: $LOG_FILE）"
        END_TS="$(date +%s)"
        write_status "$DEPLOY_RC" "$((END_TS - START_TS))"
        exit "$DEPLOY_RC"
    fi
    PUBLISHED=true
    log "步骤4完成: 发布成功"

    # ---- 5. 验收: health 校验（失败重试 1 次） ----
    log "步骤5: 验收 /api/health ..."
    DEPLOY_HOST="$("${SITE_CLI[@]}" get deploy.host)"
    HEALTH_OK=0
    if [ -n "$DEPLOY_HOST" ]; then
        HEALTH_URL="http://${DEPLOY_HOST}/api/health"
        for attempt in 1 2; do
            BODY="$(curl -s --max-time 30 "$HEALTH_URL" || true)"
            if printf '%s' "$BODY" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    fc = d.get("frames_count", 0) or 0
except Exception:
    fc = 0
sys.exit(0 if fc > 0 else 1)
'; then
                HEALTH_OK=1
                log "  health 通过: $BODY"
                break
            elif printf '%s' "$BODY" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    ok = bool(d.get("ok")) and str(d.get("database", "")) == "ready" and str(d.get("version", "")) != ""
except Exception:
    ok = False
sys.exit(0 if ok else 1)
'; then
                # 兼容线上旧版 health（无 frames_count 字段）: ok=true + database=ready 即算通过
                HEALTH_OK=1
                log "  health 通过（旧版结构，无 frames_count 字段）: $BODY"
                break
            else
                log "  第 ${attempt}/2 次验收未通过: ${BODY:-<空响应>}"
                [ "$attempt" -lt 2 ] && sleep 15
            fi
        done
    else
        log "[nightly][error] site.json 缺 deploy.host，无法验收"
    fi

    if [ "$HEALTH_OK" -ne 1 ]; then
        log "[nightly][error] 验收失败（health 两次尝试均不通过），标记发布异常"
        END_TS="$(date +%s)"
        write_status 1 "$((END_TS - START_TS))"
        exit 1
    fi
    log "步骤5完成: 验收通过"
fi

# ---- 6. 成功落盘 ----
END_TS="$(date +%s)"
write_status 0 "$((END_TS - START_TS))"
log "全部完成: 索引新增=${INDEXED_NEW} 发布=${PUBLISHED} 验收=${HEALTH_OK}，状态已写入 $STATUS_FILE"
exit 0
