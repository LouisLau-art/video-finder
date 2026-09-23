# video-finder · 视频语义检索（文本 / 图片搜视频帧）

![Python 3.14](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Platform Linux](https://img.shields.io/badge/Platform-Linux-lightgrey?logo=linux&logoColor=white)
![Chroma](https://img.shields.io/badge/VectorDB-Chroma-orange)
![CN-CLIP](https://img.shields.io/badge/Encoder-CN--CLIP-purple)
![FastAPI](https://img.shields.io/badge/Service-FastAPI-009688)

> 一句话：给一堆本地视频，输入一句中文/英文描述，或直接上传一张参考图，就能找回 **文件名 + 秒数 + 关键帧截图**。

单机 CPU 可跑（无独显、无 Docker、不调付费 API）：`uv` + `ffmpeg` + `Chroma` + `CN-CLIP / SigLIP2` + `FastAPI`。
设计目标是从 10 个视频的验证集平滑扩展到 **数千个 视频 / 数百 GB+ 素材库**。

## 核心能力

- **中文文本搜帧**：自然语言直查，如「夜跑 红色背心 冲刺」「白色帐篷 草坪」「竞速瞬间瞬间」，返回视频级排名 + 命中秒数 + 关键帧图。
- **以图搜图**：上传 / 拖拽一张参考图（或 base64 JSON），不用任何文字，直接按画面特征检索到相同镜头；实测命中原视频同时刻帧相似度 100%。
- **场景感知抽帧**：PySceneDetect 场景切分、每场景取 1–3 个中间帧，切不出自动回退纯 1fps；片头尾黑场 / 纯色卡自动过滤，避免污染 top1。
- **多 encoder 可插拔 + 自动对齐**：`SigLIP2 → open_clip → dummy` 自动回退，可选中文原生 `CN-CLIP`；实际 encoder 写入索引元数据，查询时自动配对，保证图文同一向量空间。
- **视频级评测体系**：hit@1/5/10 + MRR，多配置横向对比（翻译层 / 中文直查 / 中文原生），结果落 JSON 报告。
- **REST 服务化**：FastAPI 暴露文本检索、图片检索、关键帧图、健康检查 4 个端点，开箱跨域，供 Web / 机器人直接调用。
- **全量索引管线**：NAS(SMB) 直读零拷贝、多进程并行抽帧、**断点续跑**、增量 upsert、`nice` 限速后台跑，10 万帧级索引无人值守。
- **工程细节**：凭证外置（配置文件 600 权限，绝不进仓库）、帧级全局唯一命名防撞名、评测集与生产索引物理隔离、单文件异常不中断整批。

## 架构

```text
视频库 (本地磁盘 / SMB 网络共享 / NAS)
  │
  ▼  06_index_all.py ── 全量索引管线（并行 + 断点续跑 + 限速）
  │     发现 → 抽帧(01) → 编码(CN-CLIP) → 批量 upsert
  │
  ├─▶ frames_full/*.jpg            关键帧落盘（含黑场过滤）
  ├─▶ frames_full/manifest.jsonl   帧级明细（video_key/share/relpath/time）
  └─▶ index_full/cnclip/           Chroma 向量库（cosine，帧级向量）
        + state.json               断点状态（done/failed/skip）
  │
  ▼  05_api.py ── FastAPI 检索服务
  │     POST /api/search            文本 → 编码 → 向量检索
  │     POST /api/search-by-image   图片 → 图像编码 → 向量检索（multipart / base64）
  │     GET  /api/frames/{name}     关键帧预览图
  │     GET  /api/health            健康检查 {"status":"ok","frames_count":N}
  ▼
调用方（Web 前端 / 聊天机器人 / CLI 03_search.py）
```

单机链路（最小验证集即可跑通）：

```text
mp4 (videos_sample/)
  → ffmpeg 抽帧（优先场景切分，失败回退 1fps，黑场过滤）
  → frames/{video_id}_{ss}.jpg + manifest.jsonl
  → embedding (CPU, batch)
  → Chroma 本地索引 chroma_db/ + encoder.json（记录实际 encoder）
  → 查询：文本/图片 → 同一 encoder 编码 → 余弦距离 → rank/score/video_id/time/frame_path
```

## 快速开始

环境：Arch Linux / Python 3.14（`.python-version` 已锁定）/ `uv` / `ffmpeg + ffprobe`，无独显可跑。

```bash
# 1) 建 venv
uv venv && source .venv/bin/activate

# 2) 先装 CPU 版 torch + torchvision（重要！默认轮子会混进 CUDA 版
#    torchvision，导致 open_clip/transformers 报 torchvision::nms does not exist）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) 装其余依赖
uv pip install -e .

# 4) 确认 ffmpeg / ffprobe
ffmpeg -version | head -n 1 && ffprobe -version | head -n 1
```

最小验证（1 个 5 秒合成视频，无需真实素材）：

```bash
# 生成假视频
ffmpeg -y -f lavfi -i "testsrc=duration=5:size=640x360:rate=30" videos_sample/fake_test.mp4

python scripts/01_extract.py                 # 抽帧
python scripts/02_embed_index.py --batch-size 2   # 建索引
python scripts/03_search.py "white tent lawn" --topk 5   # 查询
```

真实素材三步走（与 `scripts/` 实现一致）：

```bash
# 0) 放视频（先 10–20 个，别一上来就全量）
cp /path/to/*.mp4 videos_sample/

# 1) 抽帧：优先场景切分（每场景 1–3 中帧），失败回退纯 1fps
python scripts/01_extract.py
# 可选：--input-dir videos_sample --frames-dir frames
#       --manifest frames/manifest.jsonl --fps 1.0 --max-per-scene 3

# 2) embedding + 建索引（CPU 小 batch）
python scripts/02_embed_index.py
# 可选：--manifest frames/manifest.jsonl --db chroma_db --collection frames
#       --batch-size 4 --model siglip2|cnclip|openclip|dummy --rebuild

# 3) 查
python scripts/03_search.py "白色帐篷 草坪" --topk 10
# 可选：--db chroma_db --model siglip2|cnclip|openclip|dummy
#       --translate（启用 opus-mt 翻译层，默认直查） --topk 20
```

### 站点私有配置（NAS / 隧道）

内网的 NAS 地址、共享映射与隧道参数放在**仓库之外**，不进 git：

```bash
mkdir -p ~/.config/video-finder
cp docs/site.example.json ~/.config/video-finder/site.json
chmod 600 ~/.config/video-finder/site.json
# 按示例格式填写真实值：nas（smb_host/prefix/synology_web_base/path_map）、
# shares（share/mount/name 列表）、tunnel（user/host/remote_port/local_port）
```

读取顺序：`$VF_SITE_CONFIG` → `~/.config/video-finder/site.json` → `./site.json`；
缺失则以降级空配置运行（本地最小链路不受影响）。
相关脚本：`scripts/site_config.py`（`get nas.smb_host` / `shares` / `tunnel`）、
`scripts/mount_nas.sh`（幂等挂载）、`scripts/keep_tunnel.sh`（服务与隧道保活）。

## 检索服务 API（`scripts/05_api.py`）

```bash
uv run python scripts/05_api.py --port 8000
# 默认按顺序探测索引：全量索引库 → 评测库 → chroma_db
# 显式指定：--db data/local-runtime/index_full/cnclip --port 8000
```

```bash
# 健康检查
curl http://127.0.0.1:8000/api/health
# → {"status":"ok","frames_count":636}

# 文本搜帧
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"夜跑 红色背心 冲刺","top_k":5}'

# 以图搜图（multipart 上传）
curl -F "image=@frame.jpg" -F "top_k=3" \
  http://127.0.0.1:8000/api/search-by-image

# 以图搜图（base64 JSON，前端压缩后常用）
curl -X POST http://127.0.0.1:8000/api/search-by-image \
  -H 'Content-Type: application/json' \
  -d '{"image_base64":"data:image/jpeg;base64,...","top_k":3}'
```

统一响应结构：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "query": "夜跑 红色背心 冲刺",
    "total": 5,
    "elapsed_ms": 82,
    "encoder": "cnclip",
    "results": [
      {
        "rank": 1,
        "score": 0.71,
        "match_percentage": "71%",
        "video_id": "demo_clip",
        "video_name": "demo_clip.mp4",
        "time_seconds": 35.4,
        "time_formatted": "00:35.4",
        "frame_image_url": "/api/frames/demo_clip_35.4.jpg",
        "nas_path": "share/folder/demo_clip.mp4"
      }
    ]
  }
}
```

> 图片上传限制：常规后缀直通，非标准后缀告警放行；15MB 上限；坏图 / 缺字段 / 不支持的类型分别返回 400 / 415。

## 全量索引管线（`scripts/06_index_all.py`）

面向几千个视频的无人值守索引：**NAS 直读、不需要先下载视频**（实测 SMB 顺序读 60MB/s，4K 素材流式抽帧 ≈ 2.3× 实时速度）。

```bash
# 0) 挂载网络共享（幂等，可重复执行）
bash scripts/mount_nas.sh

# 1) 只统计不落盘：视频数、总体积、目录分布
.venv/bin/python scripts/06_index_all.py --dry-run

# 2) 小样验证（指定目录与索引位置，不动生产库）
.venv/bin/python scripts/06_index_all.py --limit 3 \
  --frames-dir /tmp/smoke/frames --manifest /tmp/smoke/manifest.jsonl --db /tmp/smoke/chroma

# 3) 按目录分片跑高价值内容（可断点续跑，重复执行自动 skip 已完成）
nohup .venv/bin/python scripts/06_index_all.py \
  --include-dir <目录关键字> --workers 3 --nice 10 \
  > /tmp/index_full.log 2>&1 &

# 4) 全量（数百 GB+ 素材，10 万帧级；CPU 12 核估算 6–10 小时）
.venv/bin/python scripts/06_index_all.py
```

关键机制：

| 机制 | 说明 |
|---|---|
| 断点续跑 | `state.json` 按视频记录 done/failed + 帧数，重跑自动 skip；失败记录原因不中断整批 |
| 全局唯一命名 | `video_key = md5(share/relpath)[:8]`，帧名 `{key}_{stem}_{t}.jpg`，避免不同目录同名视频撞名 |
| 并行 + 限速 | 多进程抽帧（默认 3 worker），worker 内 `os.nice(10)`；主进程批量编码（64 帧/批） |
| 增量 upsert | 按批 encode + upsert 到 Chroma，崩溃后从断点续 |
| 元数据自带定位 | manifest / 向量 metadata 携带 `share + relpath`，检索结果直接给出可点击的原始路径 |

## 评测

```bash
# 本地评测集（不入库；格式见 eval_local/queries.json 示例）
uv run python scripts/04_eval.py --queries eval_local/queries.json
# → 汇总表（hit@1/5/10 + MRR）+ 每查询 top1 视频 + eval_local/report.json
```

对比配置：`siglip2+tr`（本地翻译层）/ `siglip2+zh2en`（中文直查）/ `cnclip+raw`（中文原生），同一帧集分别建索引。

实测（20 个真实品牌视频 / 521 帧；短查询 17 条，长句描述型 12 条，视频级命中率）：

| 配置 | 短查询 hit@1 | 短查询 hit@5 | 长句 hit@5 | 长句 hit@10 | 短查询均耗时 |
|---|---|---|---|---|---|
| SigLIP2 + opus-mt 翻译 | 0.24 | 0.35 | 0.58 | 0.58 | ~340ms |
| SigLIP2 + 中文直查 | 0.29 | 0.71 | 0.75 | 0.75 | ~84ms |
| **CN-CLIP 中文原生** | **0.41** | **0.82** | **0.83** | **0.92** | **~77ms** |

结论：中文原生 CN-CLIP 命中率最高且最快，翻译层降级为可选路径（`--translate`）。
以图搜图实测：端到端 ~0.5–0.6s（含上传与编码），同图检索命中相似度 100%。

> 评测集与索引在 `eval_local/`、`data/`（已 `.gitignore`），仓库只提交代码。

## 目录结构

```text
video-finder/
├── scripts/
│   ├── 01_extract.py     # ffmpeg 抽帧（场景切分 → 1fps 回退 + 黑场过滤）
│   ├── 02_embed_index.py # embedding + 写 Chroma（SigLIP2/open_clip/cnclip/dummy）
│   ├── 03_search.py      # CLI 文本查询（中文直查 / 可选翻译 / encoder 对齐）
│   ├── 04_eval.py        # 视频级 hit@k / MRR 评测
│   ├── 05_api.py         # FastAPI 检索服务（文本 + 以图搜图 + 帧图 + 健康检查）
│   ├── 06_index_all.py   # 全量索引管线（NAS 直读 / 并行 / 断点续跑）
│   ├── mount_nas.sh      # 幂等挂载网络共享（凭证外置）
│   └── keep_tunnel.sh    # 服务与反向隧道保活（凭证外置）
├── tests/test_smoke.py   # 48 项冒烟测试
├── docs/                 # 前后端接口契约与页面规格
├── pyproject.toml
├── LICENSE               # MIT
└── README.md
```

关键文件说明：

- `01_extract.py`：`video_id = 文件名去后缀`；帧命名 `{video_id}_{ss}.jpg`（保留 1 位小数）；manifest 每行 `video_id/time/frame_path/video_path`。
- `02_embed_index.py`：Chroma collection 默认 `frames`，`hnsw:space=cosine`，id 形如 `{video_id}@{time:.1f}`；结束写 `encoder.json {encoder, model_id, dim}`。
- `05_api.py`：`score = 1 - dist / 2`（cosine ∈ [0,2] 归一化）；`ensure_state()` 懒加载模型；CORS 全开供前端直连。
- `06_index_all.py`：`--db` 为索引根，Chroma 实际写入 `<db>/<encoder>/`，与评测库结构对齐，API 默认探测顺序可直接命中。

## 常见坑

| 现象 | 原因 / 解法 |
|---|---|
| `torchvision::nms does not exist` | 混装了 CUDA 版 torchvision；按快速开始先用 CPU index 装 `torch torchvision` 再 `uv pip install -e .` |
| `manifest 为空或不存在` | 先跑 `01_extract.py`，确认 `frames/manifest.jsonl` 非空 |
| `collection 不存在` | 先跑 `02_embed_index.py` |
| 查询全是 `[文件缺失?]` | 帧文件被删但索引还在；重跑 01 + 02（02 加 `--rebuild`） |
| `dummy` 分数随机 | 无网兜底模式；有网后重跑 02 自动升级到 SigLIP2 / CN-CLIP |
| NAS 全量遍历慢 | 已在管线内剪枝 `@eaDir / #recycle / site-packages / node_modules` 等；建议先 `--include-dir` 分片跑 |

## Roadmap

- [x] 评测基础设施 `scripts/04_eval.py`（视频级 hit@k/MRR）+ 20 视频 / 521 帧实测：中文原生 CN-CLIP 最优。
- [x] 评测集扩展至 20 个视频（短查询 17 条 + 长句 12 条），确定默认路径为中文直查、CN-CLIP 为推荐 encoder。
- [x] 服务化：FastAPI 文本检索 + 以图搜图 + 关键帧回传。
- [x] 全量索引管线：NAS 直读、并行、断点续跑（面向 数千个 视频 / 数百 GB+）。
- [ ] 全量索引实跑 + 检索质量回归（50+ 视频真实查询重测）。
- [ ] 时间定位更细：场景内多帧去重 + 镜头边界微调（0.1s → 帧级）。
- [ ] 结构化元数据混合检索：目录/年份/品类字典 + 向量召回融合。
- [ ] 面向终端用户的交互层（聊天机器人 / Web 前端）。

## License

MIT License，Copyright (c) 2026 LouisLau-art，详见 [LICENSE](./LICENSE)。
