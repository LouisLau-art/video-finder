# Issue tracker: 本地 Markdown

本仓库的 issue 与 spec 以 markdown 文件形式存放在 `.scratch/`。

> **为什么是本地而非 GitHub Issues**：本仓库是 public，而检索需求会频繁涉及公司素材名、内网环境与业务背景（这些内容不适合公开）。
> 若将来需要对外协作，可另开一个脱敏的公开 issue，但默认一律走本地。
> `.scratch/` 已在 `.gitignore` 中，不会随仓库上传。

## 约定

- 一个 feature 一个目录：`.scratch/<feature-slug>/`
- spec 是 `.scratch/<feature-slug>/spec.md`
- 实现 issue 一个 ticket 一个文件：`.scratch/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 编号，**不要**合并成单个 tickets 文件
- triage 状态记在文件靠上的 `Status:` 行（角色字符串见 `triage-labels.md`）
- 评论与对话历史追加到文件底部 `## Comments` 标题下

## 当某个 skill 说 "publish to the issue tracker"

在 `.scratch/<feature-slug>/` 下新建文件（目录不存在则创建）。

## 当某个 skill 说 "fetch the relevant ticket"

读取对应路径的文件。用户通常会直接给出路径或 issue 编号。

## 敏感信息纪律（写 issue / spec 时必须遵守）

`.scratch/` 虽已 gitignore，但**写进去的内容一律按"可能被外发"对待**（会被粘贴、分享、或误提交），所以 issue / spec / 评论同样必须脱敏：

- **不写**：公司/品牌名、内网与公网服务器 IP、真实 NAS 路径与共享名、真实素材文件名、同事姓名、业务规模数字（如总素材量/时长）
- **需要真实素材做证据时**：单独写进 `.scratch/<feature-slug>/local-context.md`（该文件仅本地参考），正文用中性占位指过去，如
  `实测素材 <示例素材 A> 命中率 100%`
- **中性占位示例**：`<示例素材 A>`、`<目录关键字>`、`<内网共享>`、`<样例站点>`
- **提交前自检**：`python3 scripts/check_sensitive.py`（扫描工作树与 `.scratch/`，命中即报错退出）

## Wayfinding 操作

供 `/wayfinder` 使用。**map** 是主文件，每个 ticket 是它的**子文件**。

- **Map**：`.scratch/<effort>/map.md`（Notes / Decisions-so-far / Fog 正文）。
- **子 ticket**：`.scratch/<effort>/issues/NN-<slug>.md`，从 `01` 编号，问题写在正文。`Type:` 行记录类型（`research`/`prototype`/`grilling`/`task`）；`Status:` 行记录 `claimed`/`resolved`。
- **阻塞**：靠上的 `Blocked by: NN, NN` 行。列出的文件全部 `resolved` 时该 ticket 解除阻塞。
- **Frontier**：扫描 `.scratch/<effort>/issues/`，找 open、无阻塞、未被认领的文件；编号小者优先。
- **Claim**：动手前先设 `Status: claimed` 并保存。
- **Resolve**：在 `## Answer` 标题下追加答案，设 `Status: resolved`，然后把上下文指针（要点 + 链接）追加到 `map.md` 的 Decisions-so-far。
