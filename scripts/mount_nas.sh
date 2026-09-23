#!/usr/bin/env bash
# mount_nas.sh — 幂等挂载 NAS 只读共享（video-finder 全量索引用）。
#
# 共享与挂载点来自仓库外的站点配置（`python3 scripts/site_config.py shares`，
# 每行 share<TAB>mount<TAB>name），本文件不存任何真实地址。
# 站点配置示例见 docs/site.example.json。
#
# 行为:
#   1) 把 "${SRC_AUTH}"（"key = value" 带空格格式）规范化为 /root/.smbcreds
#      （"key=value" 无空格，600），内容无变化才跳过写入；
#   2) 逐个检查挂载点（mountpoint -q），已挂载则跳过，未挂载才 mount -t cifs …ro…；
#   3) 可重复执行，第二次应全部 SKIP。
#
# 用法: sudo bash scripts/mount_nas.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_AUTH="/home/louis/.config/video-finder/smb_auth"
DST_CREDS="/root/.smbcreds"

# 1) 规范化凭证：去掉 " = " 两侧空格 -> "="
normalize_creds() {
    if [[ ! -f "$SRC_AUTH" ]]; then
        echo "[mount] 凭证源不存在: $SRC_AUTH" >&2
        return 1
    fi
    local tmp
    tmp="$(mktemp)"
    # 每行 "key = value" -> "key=value"（只处理首个等号两侧空白，保留值内空格）
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*=[[:space:]]*/=/' "$SRC_AUTH" > "$tmp"
    chmod 600 "$tmp"
    if [[ -f "$DST_CREDS" ]] && cmp -s "$tmp" "$DST_CREDS"; then
        echo "[mount] 凭证无变化，跳过写入: $DST_CREDS"
        rm -f "$tmp"
    else
        mv "$tmp" "$DST_CREDS"
        chmod 600 "$DST_CREDS"
        echo "[mount] 凭证已规范化写入: $DST_CREDS (600)"
    fi
}

# 2) 幂等挂载单个共享
mount_one() {
    local share="$1" mnt="$2"
    mkdir -p "$mnt"
    if mountpoint -q "$mnt"; then
        echo "[mount] SKIP 已挂载: $mnt ($(grep -E " $mnt " /proc/mounts | awk '{print $1}' | head -n1))"
        return 0
    fi
    mount -t cifs "$share" "$mnt" \
        -o "credentials=${DST_CREDS},ro,iocharset=utf8,vers=3.0"
    echo "[mount] OK 已挂载: $share -> $mnt"
}

# 3) 从站点配置读共享列表（无配置时友好报错退出 1）
SHARES_OUT="$(python3 "$SCRIPT_DIR/site_config.py" shares)" || {
    echo "[mount] 读取站点配置失败：请先配置 site.json（参见 docs/site.example.json）" >&2
    exit 1
}
if [[ -z "$SHARES_OUT" ]]; then
    echo "[mount] 共享列表为空：请先配置 site.json（参见 docs/site.example.json），或设置 VF_SITE_CONFIG" >&2
    exit 1
fi

normalize_creds
while IFS=$'\t' read -r share mnt _name; do
    [[ -z "$share" || -z "$mnt" ]] && continue
    mount_one "$share" "$mnt"
done <<< "$SHARES_OUT"
echo "[mount] done"
