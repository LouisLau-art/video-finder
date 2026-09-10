# videos_sample / 说明

把要检索的 `.mp4`（兼容 `.mov/.mkv/.avi/.webm/.m4v`）**直接放进本目录**即可。

```bash
cp /path/to/*.mp4 videos_sample/
ls videos_sample/
```

- `video_id` = 文件名去后缀（如 `a.mp4` -> `a`）。
- 抽帧脚本会扫描本目录所有视频文件，输出到 `../frames/` 并写 `../frames/manifest.jsonl`。
- 先放 **10–20 个**视频验证链路，无卡 CPU 会慢，大批量后面再说。
