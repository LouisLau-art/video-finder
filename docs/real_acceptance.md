# 真实专有名称验收

## 用途

`08_real_acceptance.py` 用于索引完成后，通过本地 HTTP `/api/search` 接口验收真实专有名称失败案例是否已经能够召回目标素材。它只调用接口，不读取 Chroma、不写入向量库、不依赖 NAS 或素材原文件。

## 准备用例

真实查询词和目标素材标识不能进入版本库。复制 `docs/real_acceptance_cases.example.json` 到 gitignored 的：

```text
eval_local/real_acceptance_cases.json
```

然后在本地填写真实值。每条用例需要：

- `id`：稳定且中性的用例编号；
- `query`：真实查询词，仅存在于本地文件；
- `target_video_id`：期望命中的目标素材标识，仅存在于本地文件；
- `notes`：本地备注，不写入报告。

用例文件应保持在 `eval_local/`，该目录已被 `.gitignore` 忽略。脚本在文件缺失、格式错误或路径未被忽略时以退出码 `1` 明确失败，不会静默跳过。

## 执行

默认服务地址是本地服务地址，默认 `top_k` 为 20，逐条顺序执行，不并发：

```text
.venv/bin/python scripts/08_real_acceptance.py
```

也可以显式指定参数：

```text
.venv/bin/python scripts/08_real_acceptance.py \
  --cases eval_local/real_acceptance_cases.json \
  --report eval_local/real_acceptance_report.json \
  --top-k 20 \
  --timeout 30
```

脚本只向 `/api/search` 发送查询，不修改服务配置或索引。单条请求超时或返回错误时会记录该条错误，并继续执行后续用例。

## 报告解读

报告默认写入 gitignored 的 `eval_local/real_acceptance_report.json`，只包含：

- 用例 ID；
- 查询和目标标识的哈希前缀；
- 是否命中；
- 目标名次；
- `match_type`；
- 是否由关键词通道召回；
- `matched_text` 是否非空；
- 错误类别或无法判定原因。

报告不会写入查询原文、素材名称、完整路径或 `matched_text` 原文。

## 退出码

退出码与 `07_hybrid_eval.py` 保持一致：

- `0`：全部用例通过；
- `1`：脚本、输入、HTTP 或响应契约错误；
- `2`：用例执行完整，但至少一条未召回或未被关键词通道召回；
- `3`：响应存在但无法判定，例如命中来源字段不符合契约。

`2` 表示验收不通过；`3` 表示结果不完整或契约状态未知，不应当当作通过。

## 安全边界

- 真实用例文件只放在 `eval_local/`，不提交到版本库；
- 模板只含 `placeholder_` 中性值；
- 报告只写哈希前缀，不写可逆的查询或素材名称；
- 脚本不写向量库、不重建索引、不重启服务；
- 运行前应确认索引已经完成，并使用同一份目标素材标识核对结果。
