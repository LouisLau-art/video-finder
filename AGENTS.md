# 项目约定

## 算力与代码实现分工（Token 节约）
- 助手不负责具体实现、写业务代码、审读大量代码等重度消耗 Token 的工作。
- 凡是涉及前端、界面开发、交互实现等工作，尽可能用云端 AI 前端生成平台来做（走平台额度），避免消耗本地模型的个人 Token。
- 助手主要负责：
  1. 出方案、理架构；
  2. 撰写给前端生成平台的 Prompt 与界面/交互需求；
  3. 制定前后端接口契约与数据规格；
  4. 排查疑难与联调指导。

## Agent skills

### Issue tracker

issue 与 spec 以本地 markdown 形式放在 `.scratch/`（该目录已 gitignore，不进公开仓库）。见 `docs/agents/issue-tracker.md`。

### Triage labels

沿用五个规范角色名（`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`）。见 `docs/agents/triage-labels.md`。

### Domain docs

单 context：根目录 `CONTEXT.md` + `docs/adr/`。见 `docs/agents/domain.md`。

## 敏感信息纪律（重要）

仓库是 **public**。任何可能进仓库或被外发的内容（代码、文档、issue、spec、提交信息、给外部平台的 Prompt）都必须脱敏：

- 不写：公司/品牌名、内网与公网服务器 IP、真实 NAS 路径与共享名、真实素材文件名、同事姓名、业务规模数字
- 站点与环境相关的真实值一律放仓库外的 `~/.config/video-finder/site.json`，代码经 `scripts/site_config.py` 读取（示例见 `docs/site.example.json`）
- 需要引用真实素材作为证据时，单独写到 `.scratch/<feature>/local-context.md`，并在正文用中性占位（如 `<示例素材>`）指过去
- 改动后自检：`python3 scripts/check_sensitive.py`（若存在）或人工核对上述清单
