# video-finder · 视频搜索（文本搜视频帧，单机 CPU 原型）

![Python 3.14](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Platform Linux](https://img.shields.io/badge/Platform-Linux-lightgrey?logo=linux&logoColor=white)
![Chroma](https://img.shields.io/badge/VectorDB-Chroma-orange)
![SigLIP2](https://img.shields.io/badge/Encoder-SigLIP2-purple)

> 一句话：给一堆本地 mp4，一句文本（中/英文）就能找回 **文件名 + 秒数 + 截图（帧路径）**。

本仓库是在 Arch Linux 无独显机器上验证通的最小链路：只用 `uv` + 本地 `Chroma`，不依赖 Docker、不调付费 API。

## 功能特性

- **文本搜帧**：英文自然描述（如 `white tent lawn black man`）按相似度排序返回帧。
- **中文查询**：含汉字时优先走本地翻译模型 `Helsinki-NLP/opus-mt-zh-en`（懒加载，CPU）译成英文再编码；模型不可用/翻译失败自动回退内置关键词映射表，无网不断链。
- **场景优先抽帧**：优先 PySceneDetect `ContentDetector` 按场景切分，每场景留 1–3 个中间帧；切不出/报错自动回退纯 1fps，不中断。
- **黑场/纯色卡过滤**：抽帧后丢弃片头尾黑场、纯色卡帧（灰度均值 < 8 或标准差 < 6），不写 manifest，避免污染检索 top1。
- **Encoder 三档自动回退**：`SigLIP2（google/siglip2-base-patch16-224）→ open_clip ViT-B/32（laion2b_s34b_b79k）→ dummy 哈希（仅保链路）`，实际用哪个会写进 `chroma_db/encoder.json`，查询时自动对齐，保证图文同一向量空间。
- **本地持久化**：向量存 `chroma_db/`（cosine 空间），payload 含 `video_id / time / frame_path`，拿着 `frame_path` 直接打开 jpg 就是截图，`video_id + time` 可回跳原视频对应秒数。
- **CPU 友好**：默认 `--batch-size 4`，小内存可用 `--batch-size 2`。

## 架构链路

```text
mp4 (videos_sample/)
  │
  ▼
ffmpeg 抽帧 ──优先──▶ PySceneDetect 场景切分 (每场景 1–3 中帧)
  │                    ──失败──▶ 回退纯 1fps
  ▼
frames/{video_id}_{ss}.jpg + frames/manifest.jsonl
(video_id / time / frame_path / video_path)
  │
  ▼
embedding (CPU)
  SigLIP2 → open_clip ViT-B/32 → dummy哈希(兜底)
  │
  ▼
Chroma 本地索引 (chroma_db/, collection=frames, hnsw:space=cosine)
  + chroma_db/encoder.json (记录实际 encoder)
  │
  ▼
查询: 文本 → 中文翻译(含汉字时, 离线回退关键词映射) → 同一 encoder 编码 → Chroma query
  → 排序输出 rank / score / dist / video_id / time / frame_path
```

## 快速开始

环境：Arch Linux / Python 3.14（`.python-version` 已锁定）/ `uv` / `ffmpeg + ffprobe`，无 N 卡、无 Docker。

```bash
cd video-finder   # 或 video-finder-prototype（本地旧目录名）

# 1) 建 venv（uv 会按 .python-version 用 python3.14）
uv venv
source .venv/bin/activate

# 2) 先装 CPU 版 torch + torchvision（重要！直接装默认轮子会混进 CUDA 版
#    torchvision，导致 open_clip/transformers 报 torchvision::nms does not exist）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) 装其余依赖
uv pip install -e .

# 4) 确认 ffmpeg / ffprobe
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
```

三步跑通（命令与参数均与 `scripts/` 实际实现一致）：

```bash
# 0) 放视频（先 10–20 个，别一上来就全量）
cp /path/to/*.mp4 videos_sample/

# 1) 抽帧：优先场景切分（每场景 1–3 中帧），失败回退纯 1fps
python scripts/01_extract.py
# 输出：frames/{video_id}_{ss}.jpg + frames/manifest.jsonl（含 video_id,time,frame_path）
# 可选参数：--input-dir videos_sample --frames-dir frames
#           --manifest frames/manifest.jsonl --fps 1.0 --max-per-scene 3

# 2) embedding + 建索引（CPU 小 batch，默认 4）
python scripts/02_embed_index.py
# 可选参数：--manifest frames/manifest.jsonl --db chroma_db --collection frames
#           --batch-size 4 --model siglip2|openclip|dummy --rebuild
#   --batch-size 2 更省内存 / --rebuild 重建索引 / --model 强制指定 encoder
# 输出：chroma_db/（本地持久化）+ chroma_db/encoder.json

# 3) 查
python scripts/03_search.py "white tent lawn" --topk 20
python scripts/03_search.py "white tent lawn black man" --topk 20
python scripts/03_search.py "白色帐篷 草坪" --topk 10
# 03 可选参数：--db chroma_db --collection frames
#             --model siglip2|openclip|dummy（默认读 encoder.json自动对齐）
#             --topk 20（默认 20）
```

最小验证（1 个 5 秒假视频，无需准备真实素材）：

```bash
# 生成 5 秒测试视频
ffmpeg -y -f lavfi -i "testsrc=duration=5:size=640x360:rate=30" videos_sample/fake_test.mp4

python scripts/01_extract.py
python scripts/02_embed_index.py --batch-size 2
python scripts/03_search.py "white tent lawn" --topk 5
python scripts/03_search.py --help
```

## 查询示例

英文查询（推荐，语义最准）：

```bash
python scripts/03_search.py "white tent lawn" --topk 20
python scripts/03_search.py "white tent lawn black man" --topk 20
```

中文查询（优先翻译模型，离线回退内置映射）：

```bash
python scripts/03_search.py "白色帐篷 草坪" --topk 10
# 终端会打印：[query] 翻译(zh->en): '白色帐篷 草坪' -> 'white tent lawn'
# 离线/模型不可用时回退：[query] 映射: '白色帐篷 草坪' -> 'white tent lawn'
```

翻译不可用时用的兜底映射原理（`scripts/03_search.py` 顶部 `ZH2EN` 表，按长词优先做字符串替换，前后补空格避免粘连）：

| 中文 | 映射英文 | 中文 | 映射英文 |
|---|---|---|---|
| 帐篷 | tent | 草坪/草地 | lawn |
| 草原 | grassland | 白色 | white |
| 黑色 | black | 黑人 | black man |
| 白人 | white man | 男人/男子 | man |
| 女人/女子 | woman | 小孩/孩子 | child |
| 狗 | dog | 猫 | cat |
| 车/汽车 | car | 房子 | house |
| 树 | tree | 花 | flower |
| 水 | water | 天空 | sky |
| 夜晚 | night | 白天 | daytime |
| 室内 | indoor | 室外 | outdoor |

> 注意：只是关键词替换，不是翻译模型。`白色帐篷 草坪 黑人` → `white tent lawn black man` 这种简单组合没问题，长句/复杂语义请直接写英文。

输出示例（排序 + 分数 + 帧路径即截图）：

```text
[query] 'white tent lawn'
[encoder] 按 encoder.json 用 siglip2
[result] top5 (共 32 帧, encoder=siglip2):
rank score    dist     video_id              time     frame_path
1    0.8123   0.3754   wedding_a             12.5     frames/wedding_a_12.5.jpg
2    0.7901   0.4198   wedding_a             13.5     frames/wedding_a_13.5.jpg
...
```

说明：`score = 1 - dist/2`（Chroma cosine distance ∈ [0,2] 换算而来）；若某行尾部出现 `[文件缺失?]`，说明帧文件被删但索引还在，重跑 01 + 02（02 可加 `--rebuild`）即可。

## 目录结构

```text
video-finder/
├── videos_sample/      # 把 mp4 放这里（先 10–20 个，不提交，只留 README.md 说明）
│   └── README.md
├── frames/             # 抽出的帧 + manifest.jsonl（.jpg/.manifest 不提交，.gitkeep 占位）
│   └── .gitkeep
├── chroma_db/          # Chroma 本地持久化 + encoder.json（不提交，.gitkeep 占位）
│   └── .gitkeep
├── scripts/
│   ├── 01_extract.py   # ffmpeg 抽帧（优先场景切分，失败回退 1fps）
│   ├── 02_embed_index.py # embedding + 写 Chroma（SigLIP2 → open_clip → dummy）
│   └── 03_search.py    # CLI 查询（含中文映射 + encoder 对齐）
├── pyproject.toml
├── LICENSE             # MIT
└── README.md
```

关键文件说明：

- `01_extract.py`：`video_id = 文件名去后缀`；帧命名 `{video_id}_{ss}.jpg`（ss 保留 1 位小数，如 `demo_2.5.jpg`）；manifest 每行 `video_id/time/frame_path/video_path`；支持视频后缀 `.mp4/.mov/.mkv/.avi/.webm/.m4v/.mpg/.mpeg`。
- `02_embed_index.py`：Chroma collection 名默认 `frames`，`hnsw:space=cosine`，id 形如 `{video_id}@{time:.1f}`，upsert 写入；结束写 `encoder.json {encoder, model_id, dim}`。
- `03_search.py`：复用 `02` 的 encoder 实现（`importlib` 动态加载，保证两边一致）；`query` 为必填位置参数。

## 性能说明（无卡会慢是正常的）

- 本机约束按 12 核 / 30GB / 纯 CPU 设计：embedding 一个 batch 几秒到十几秒正常，先拿 10–20 个视频测通链路。
- 建议 `--batch-size 4`；内存吃紧用 `--batch-size 2`。
- HuggingFace 模型下载慢/失败是正常的，脚本会自动回退 `SigLIP2 → open_clip ViT-B/32 → dummy 哈希（仅保链路）`，`chroma_db/encoder.json` 会记录实际用的 encoder。
- `dummy` 模式分数无语义意义，仅证明“抽帧→索引→查询”链路是通的；看到 `[encoder][WARN] 用 dummy 哈希向量` 即表示当前是兜底模式。
- `chroma_db/`、`frames/*.jpg`、`videos_sample/*.mp4` 均已 `.gitignore`，不要提交大文件。

## 排错

```bash
# 看 manifest 是否有帧
wc -l frames/manifest.jsonl && head -n 2 frames/manifest.jsonl

# 看索引里多少帧 / 用的哪个 encoder
cat chroma_db/encoder.json
python -c "import chromadb; print(chromadb.PersistentClient(path='chroma_db').get_collection('frames').count())"

# 强制用轻量回退（没网/下载失败时）
python scripts/02_embed_index.py --model openclip
python scripts/02_embed_index.py --model dummy   # 仅验证链路，分数无意义

# 重建索引（帧删了重抽、或换了 encoder 之后）
python scripts/02_embed_index.py --rebuild

# 各脚本帮助（参数以 --help 输出为准）
python scripts/01_extract.py --help
python scripts/02_embed_index.py --help
python scripts/03_search.py --help
```

常见坑：

| 现象 | 原因 / 解法 |
|---|---|
| `torchvision::nms does not exist` | 混装了 CUDA 版 torchvision；按快速开始先用 CPU index 装 `torch torchvision` 再 `uv pip install -e .` |
| `videos_sample 里没有视频文件` | 01 扫描不到支持后缀；确认文件在 `videos_sample/` 且后缀在支持列表 |
| `manifest 为空或不存在，先跑 01_extract.py` | 02 找不到 manifest；先跑 01，确认 `frames/manifest.jsonl` 非空 |
| `collection 不存在，先跑 02_embed_index.py` | 03 找不到 Chroma collection；先跑 02 |
| 查询全是 `[文件缺失?]` | 帧文件被删但索引还在；重跑 01 + 02（02 加 `--rebuild`） |
| 中文查不准 | 已接入翻译层；仍不准时改用英文自然描述，或补 `ZH2EN` 表（离线回退路径） |
| `dummy` 分数随机 | 无网兜底模式预期行为；有网后重跑 02 自动升级到 SigLIP2/open_clip |

## Roadmap

- [x] 中文查询接入翻译层：`Helsinki-NLP/opus-mt-zh-en`（懒加载，离线回退关键词映射），2026-09-14。
- [ ] 中文查询下一步：按 Roadmap 评测集对比 `SigLIP2+翻译` vs `CN-CLIP`，再决定是否换中文原生 encoder。
- [ ] 时间定位更细：场景内多帧去重 + 镜头边界微调，`time` 精度从 0.1s 向帧级对齐。
- [ ] 检索体验：`03_search.py` 加 `--show` 直接拼图预览 / 输出 HTML 报告。
- [ ] 增量索引：01 抽帧增量追加、02 按 `video_id` 增量 upsert，避免每次 `--rebuild` 全量重建。
- [ ] 评测集：固定 20 个视频 + 20 条查询，记录各 encoder（SigLIP2 / open_clip / dummy）的 top-k 命中率。
- [ ] 打包：`uv run` 一键脚本 + 示例视频生成器，非技术用户也能三步跑通。

## License

MIT License，Copyright (c) 2026 LouisLau-art，详见 [LICENSE](./LICENSE)。
