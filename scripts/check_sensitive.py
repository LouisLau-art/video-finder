#!/usr/bin/env python3
"""check_sensitive.py — 敏感信息自检（公开仓库纪律的执行工具）。

默认只扫描**可能被提交的东西**：git 跟踪的文件 + 未跟踪且未被 ignore 的文件，
再额外并入 `.scratch/`（它虽被 gitignore，但内容仍可能被外发）。

用法:
    python3 scripts/check_sensitive.py            # 扫"可能提交"集合 + .scratch/
    python3 scripts/check_sensitive.py --all      # 暴力扫全目录（含 gitignored，噪音大）
    python3 scripts/check_sensitive.py -q         # 只报总数

设计原则:
  * 特征词表本身**不含**真实值——这里用"正则形状"（内网 IP 段、私有域名、
    已知禁词的抽象规则），避免脚本自己成为泄露源。
  * 真实禁词表放在仓库外：~/.config/video-finder/sensitive_terms.txt
    （每行一个词，# 开头为注释，支持 utf-8）。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# --- 形状规则（不含真实值，可安全公开）-------------------------------

# 私有/内网 IP 段
IP_SHAPE = re.compile(
    r"(?<![\d.])("
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r")(?![\d.])"
)

# smb:// 共享引用（host 部分限定为合法主机名/IP 字符，避免匹配到正则源码）
SMB_SHAPE = re.compile(
    r"smb://[A-Za-z0-9][A-Za-z0-9._\u4e00-\u9fff-]*", re.IGNORECASE
)

# 锁文件/包元数据上下文：此处的 10.x.x.x 是包版本号，不是内网 IP
LOCKFILE_CONTEXT = re.compile(
    r"(?:version\s*=|name\s*=|registry\s*=|sha256|\.whl|/packages/|"
    r"pypi|dist-info|\d+\.\d+\.\d+\.\d+-(?:py|manylinux|cp|abi)\d*)",
    re.IGNORECASE,
)

# 私有域名后缀（常见企业内网/自建服务）
PRIVATE_DOMAIN_SHAPE = re.compile(
    r"\b[a-z0-9][a-z0-9.-]*\.(?:internal|local|lan|corp)\b", re.IGNORECASE
)

# 本仓库自身的公开白名单（这些是示例/文档里合法出现的）
ALLOWLIST = re.compile(
    r"(nas\.local|relay\.example\.com|example\.(?:com|internal)|"
    r"127\.0\.0\.1|0\.0\.0\.0|localhost|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)"
)

DEFAULT_TERMS_FILE = Path.home() / ".config" / "video-finder" / "sensitive_terms.txt"

# 默认不扫的目录
SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache",
}
# 默认不扫的二进制/大文件后缀
SKIP_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mp3", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".whl", ".so", ".bin",
    ".sqlite3", ".db", ".pickle", ".pt", ".onnx", ".pdf",
}
MAX_BYTES = 2_000_000

# 额外始终并入扫描的路径（即使被 gitignore）
ALWAYS_INCLUDE = [".scratch"]


def load_terms(path: Path) -> list[str]:
    """读仓库外禁词表；不存在则返回空（形状规则仍然生效）。"""
    if not path.is_file():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def _ok(p: Path) -> bool:
    if any(part in SKIP_DIRS for part in p.parts):
        return False
    if p.suffix.lower() in SKIP_SUFFIXES:
        return False
    try:
        return p.stat().st_size <= MAX_BYTES
    except OSError:
        return False


def git_candidate_files() -> list[Path]:
    """git 跟踪 + 未跟踪且未 ignore 的文件（== 可能被提交的东西）。"""
    out: list[Path] = []
    for extra in (["--others", "--exclude-standard"], []):
        cmd = ["git", "ls-files", "-z", *extra]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode != 0:
            continue
        for name in r.stdout.split("\0"):
            if name:
                out.append(Path(name))
    return out


def walk_files(roots: list[Path]):
    for root in roots:
        if root.is_file():
            yield root
            continue
        for p in root.rglob("*"):
            if p.is_file():
                yield p


def iter_files(roots: list[Path], scan_all: bool, explicit: bool):
    """产生待扫文件。

    * explicit=True：用户显式给了路径 → 直接遍历这些路径（尊重其意图）。
    * scan_all=True：暴力遍历 roots（含 gitignored）。
    * 否则：git 候选集 ∪ ALWAYS_INCLUDE（可能被提交的东西 + .scratch/）。
    """
    if explicit or scan_all:
        for p in walk_files(roots):
            if _ok(p):
                yield p
        return

    seen: set[Path] = set()
    for p in git_candidate_files():
        if p.is_file() and _ok(p) and p not in seen:
            seen.add(p)
            yield p
    # .scratch 等始终并入（即使被 ignore）
    for inc in ALWAYS_INCLUDE:
        d = Path(inc)
        if d.exists():
            for p in walk_files([d]):
                if _ok(p) and p not in seen:
                    seen.add(p)
                    yield p


def scan_file(p: Path, terms: list[str]) -> list[tuple[int, str, str]]:
    """返回 [(行号, 类别, 命中片段), ...]。"""
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except (OSError, ValueError):
        return []
    hits: list[tuple[int, str, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        # 锁文件/包元数据里的 x.y.z.w 是包版本号，不是内网 IP
        lockish = bool(LOCKFILE_CONTEXT.search(line))
        for label, rx in (
            ("内网IP", IP_SHAPE),
            ("SMB共享", SMB_SHAPE),
            ("私有域名", PRIVATE_DOMAIN_SHAPE),
        ):
            if label == "内网IP" and lockish:
                continue
            for m in rx.finditer(line):
                frag = m.group(0)
                if ALLOWLIST.search(frag):
                    continue
                hits.append((i, label, frag))
        for t in terms:
            if t and t in line:
                hits.append((i, "禁词", t))
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="敏感信息自检（公开仓库纪律）")
    ap.add_argument("paths", nargs="*",
                    help="要扫描的路径；省略则扫「可能提交集 + .scratch」")
    ap.add_argument("--terms", default=str(DEFAULT_TERMS_FILE),
                    help="仓库外禁词表路径（每行一个词）")
    ap.add_argument("--all", action="store_true",
                    help="暴力扫全目录（含 gitignored 本地数据，噪音大）")
    ap.add_argument("-q", "--quiet", action="store_true", help="只输出汇总")
    args = ap.parse_args(argv)

    roots = [Path(x) for x in (args.paths or ["."])]
    explicit = bool(args.paths)  # 用户显式给了路径 → 尊重其意图，直接遍历
    terms = load_terms(Path(args.terms))

    total = 0
    files = list(iter_files(roots, args.all, explicit))
    for p in files:
        hits = scan_file(p, terms)
        if hits:
            total += len(hits)
            if not args.quiet:
                for lineno, label, frag in hits[:20]:
                    print(f"  {p}:{lineno}  [{label}]  {frag}")
                if len(hits) > 20:
                    print(f"  {p}: … 另有 {len(hits) - 20} 处")

    if not terms and not args.quiet:
        print(f"  [note] 禁词表未找到：{args.terms}（仅形状规则生效）")

    print(f"\n扫描 {len(files)} 个文件（{'全目录' if args.all else '可能提交集 + .scratch'}），"
          f"命中 {total} 处" + ("  ✓ CLEAN" if total == 0 else "  ✗ 请脱敏后再提交"))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

