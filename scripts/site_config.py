#!/usr/bin/env python3
"""site_config.py — 站点私有配置加载器（通用可公开，本文件不含任何真实值）.

真实值放在仓库之外，读取优先级:
    1) 环境变量 VF_SITE_CONFIG 指向的 JSON 文件
    2) ~/.config/video-finder/site.json
    3) ./site.json（仓库根目录，可选）
都不存在时返回空默认结构，不报错（调用方以降级模式运行）。

期望的 JSON schema（示例见 docs/site.example.json）:
    {"nas": {"smb_host": ..., "prefix": ...,
             "synology_web_base": ..., "path_map": {...}},
     "shares": [{"share": ..., "mount": ..., "name": ...}, ...],
     "tunnel": {"user": ..., "host": ...,
                "remote_port": ..., "local_port": ...}}

CLI（纯标准库，供 shell 脚本调用）:
    python3 scripts/site_config.py get nas.smb_host   # 点号取任意键，缺失打印空行
    python3 scripts/site_config.py shares             # 每行 share<TAB>mount<TAB>name
    python3 scripts/site_config.py tunnel             # 每行 key<TAB>value
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIG_ENV = "VF_SITE_CONFIG"
CONFIG_HOME = Path("~/.config/video-finder/site.json").expanduser()

TUNNEL_KEYS = ("user", "host", "remote_port", "local_port")


def _defaults() -> dict:
    return {
        "nas": {"smb_host": "", "prefix": "", "synology_web_base": "", "path_map": {}},
        "shares": [],
        "tunnel": {"user": "", "host": "", "remote_port": 0, "local_port": 0},
    }


def _merge(base: dict, over: dict) -> dict:
    """逐 section 浅合并：未知键原样保留，缺失 section 用默认补齐。"""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get(CONFIG_ENV, "").strip()
    if env:
        paths.append(Path(env))
    paths.append(CONFIG_HOME)
    paths.append(Path("site.json"))
    return paths


def load(path: str | Path | None = None) -> dict:
    """加载站点配置；文件缺失/损坏一律返回空默认结构，不抛异常。"""
    cfg = _defaults()
    targets = [Path(path)] if path else candidate_paths()
    for p in targets:
        try:
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return _merge(cfg, data)
                return cfg
        except Exception:
            continue
    return cfg


def get(cfg: dict, dotted: str):
    """点号路径取值，如 get(cfg, "nas.smb_host")；缺失返回空字符串。"""
    cur: object = cfg
    for part in (dotted or "").split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    if isinstance(cur, (dict, list)):
        return json.dumps(cur, ensure_ascii=False)
    return cur


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print("用法: site_config.py get <a.b.c> | shares | tunnel")
        return 0 if len(argv) >= 2 else 1
    cfg = load()
    cmd = argv[1]
    if cmd == "get":
        if len(argv) < 3:
            print("")
            return 1
        v = get(cfg, argv[2])
        print("" if v is None else v)
        return 0
    if cmd == "shares":
        shares = cfg.get("shares", []) or []
        for s in shares:
            if not isinstance(s, dict):
                continue
            print(f"{s.get('share', '')}\t{s.get('mount', '')}\t{s.get('name', '')}")
        return 0
    if cmd == "tunnel":
        tun = cfg.get("tunnel", {}) or {}
        for k in TUNNEL_KEYS:
            print(f"{k}\t{tun.get(k, '')}")
        return 0
    print(f"未知命令: {cmd}（可用: get/shares/tunnel）", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
