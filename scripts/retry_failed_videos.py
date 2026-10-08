#!/usr/bin/env python3
"""专门补跑 state.json 中失败的视频，跳过整盘扫描，强制离线加载模型。"""
import os
import sys
import json
import re
import shutil
from pathlib import Path

# 确保离线模式和模型路径
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/home/louis/.cache/huggingface"

# 磁盘保护：仅当剩余空间 >= 此值（GB）才继续处理下一个视频，否则停下报错。
# 大视频（最长 11GB）抽帧可能产出数百 MB~数 GB，磁盘写满会连累 state/Chroma 写入。
MIN_FREE_GB = float(os.environ.get("RETRY_MIN_FREE_GB", "2.0"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
spec_06 = importlib.util.spec_from_file_location("m06", ROOT / "scripts" / "06_index_all.py")
m06 = importlib.util.module_from_spec(spec_06)
spec_06.loader.exec_module(m06)

spec_02 = importlib.util.spec_from_file_location("m02", ROOT / "scripts" / "02_embed_index.py")
m02 = importlib.util.module_from_spec(spec_02)
spec_02.loader.exec_module(m02)

import chromadb
from PIL import Image

def main():
    state_file = ROOT / "data/local-runtime/index_full/state.json"
    frames_dir = ROOT / "data/local-runtime/frames_full"
    manifest_file = frames_dir / "manifest.jsonl"
    db_dir = ROOT / "data/local-runtime/index_full"

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    # 找出失败的视频并解析出路径
    # 优先从 error 里的 /mnt 路径解析；解析不到（如 ffmpeg 偶发报错）时，
    # 退回到此前盘点生成的 video_key -> relpath 映射表反查。
    rel_fallback = {}
    dk = ROOT / ".scratch/index-audit/disk_keys.json"
    if dk.is_file():
        try:
            rel_fallback = json.loads(dk.read_text(encoding="utf-8")).get("rel", {})
        except Exception:
            rel_fallback = {}

    targets = []
    roots = m06._default_roots()
    for vkey, info in state.items():
        if info.get("status") != "failed":
            continue
        err = info.get("error", "")
        vpath = None
        m = re.search(r"'(/mnt/[^']+)'", err)
        if m:
            vpath = m.group(1)
        elif vkey in rel_fallback:
            # 从映射表反查：relpath 已知，用 video_key 校验确定它属于哪个共享
            rel = rel_fallback[vkey]
            for root_dir in roots:
                if m06.video_key_of(m06.share_of(root_dir), rel) == vkey:
                    cand = Path(root_dir) / rel
                    if cand.is_file():
                        vpath = str(cand)
                    break
        if not vpath:
            continue
        p = Path(vpath)
        if not p.is_file():
            continue
        share = m06.share_of(vpath)
        relpath = p.name
        for r in sorted(roots, key=len, reverse=True):
            try:
                relpath = p.relative_to(r).as_posix()
                break
            except ValueError:
                continue
        targets.append({
            "video_key": vkey,
            "video_path": vpath,
            "share": share,
            "relpath": relpath,
            "stem": p.stem,
            "size": p.stat().st_size,
            "mtime": p.stat().st_mtime,
        })

    # 先小后大：小文件先跑，尽快消化、也便于观察单视频的帧产出量
    targets.sort(key=lambda t: t["size"])

    print(f"找到 {len(targets)} 个待补跑视频（本地可直接读取）")
    if not targets:
        return 0

    # 加载模型
    print("正在加载 CN-CLIP 模型...")
    encoder = m02.build_encoder("cnclip")
    chroma_path = str(db_dir / "cnclip")
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_or_create_collection(name="frames", metadata={"hnsw:space": "cosine"})

    # 准备追加 manifest
    manifest_keys = m06.load_manifest_keys(manifest_file)
    man_f = manifest_file.open("a", encoding="utf-8")

    success_count = 0
    total = len(targets)

    for i, t in enumerate(targets, 1):
        vkey = t["video_key"]
        # 磁盘保护：空间不足就停下，避免写满盘连累 state/Chroma
        free_gb = shutil.disk_usage(str(ROOT)).free / 1024**3
        if free_gb < MIN_FREE_GB:
            print(f"\n[磁盘保护] 剩余 {free_gb:.2f}GB < {MIN_FREE_GB}GB，停止处理。"
                  f"已完成 {success_count}/{total}，剩余 {total - i + 1} 个未处理。")
            break
        print(f"[{i}/{total}] 处理 {vkey} - {t['stem']} "
              f"({t['size']/1024/1024:.1f}MB, 盘余 {free_gb:.2f}GB)...")
        job = {
            "video_path": t["video_path"],
            "share": t["share"],
            "relpath": t["relpath"],
            "video_key": vkey,
            "stem": t["stem"],
            "frames_dir": str(frames_dir),
            "local_scratch": "",  # 磁盘紧张，NAS 直读
            "fps": 1.0,
            "max_per_scene": 3
        }

        res = m06._extract_job(job)
        if not res["ok"]:
            print(f"  -> 抽帧失败: {res.get('error')}")
            state[vkey]["error"] = str(res.get("error"))
            continue

        rows = res["rows"]
        frames = res["frames"]
        if not rows:
            print("  -> 0 帧有效（纯黑场/纯色卡），标记完成")
            state[vkey] = {
                "status": "done",
                "mtime": t["mtime"],
                "size": t["size"],
                "frames": [],
                "empty_confirmed": True
            }
            m06.save_state(db_dir, state)
            success_count += 1
            continue

        # 编码并写入向量库
        imgs = []
        keep_rows = []
        for r in rows:
            fp = Path(r["frame_path"])
            if fp.is_file():
                try:
                    with Image.open(fp) as img:
                        imgs.append(img.convert("RGB"))
                        keep_rows.append(r)
                except Exception as e:
                    print(f"    无法读取帧 {fp}: {e}")

        if keep_rows:
            embeddings = encoder.encode_images(imgs)
            ids = [f"{r['video_key']}@{r['time']:.1f}" for r in keep_rows]
            metadatas = [{
                "video_key": r["video_key"],
                "video_id": r["video_id"],
                "video_name": r["video_id"] + ".mp4",
                "time": float(r["time"]),
                "frame_path": str(r["frame_path"]),
                "video_path": str(r["video_path"]),
                "share": r["share"],
                "relpath": r["relpath"],
            } for r in keep_rows]

            col.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)

            # 写 manifest
            if vkey not in manifest_keys:
                for r in keep_rows:
                    man_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                man_f.flush()
                manifest_keys.add(vkey)

        # 更新 state（frames 必须是帧文件名列表，与主流程 06_index_all.py 保持一致；
        # 写成整数会让主流程续跑时 `for n in st["frames"]` 崩溃）
        state[vkey] = {
            "status": "done",
            "mtime": t["mtime"],
            "size": t["size"],
            "frames": [Path(r["frame_path"]).name for r in keep_rows]
        }
        m06.save_state(db_dir, state)
        print(f"  -> 成功写入 {len(keep_rows)} 帧向量！")
        success_count += 1

    man_f.close()
    print(f"\n全部处理完毕！成功补跑 {success_count}/{total} 个视频。")

if __name__ == "__main__":
    main()
