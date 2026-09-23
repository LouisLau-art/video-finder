# Triage Labels

各 skill 用五个规范化的 triage 角色表述。本文件把这些角色映射到本仓库 issue tracker 实际使用的标签字符串。

| mattpocock/skills 中的标签 | 本仓库 tracker 中的标签 | 含义 |
| -------------------------- | ----------------------- | ---- |
| `needs-triage`             | `needs-triage`          | 维护者需要评估这个 issue |
| `needs-info`               | `needs-info`            | 等待报告者补充信息 |
| `ready-for-agent`          | `ready-for-agent`       | 已完整规格化，可交给 AFK agent |
| `ready-for-human`          | `ready-for-human`       | 需要人工实现 |
| `wontfix`                  | `wontfix`               | 不予处理 |

当某个 skill 提到某个角色（例如 "apply the AFK-ready triage label"）时，使用本表中对应的标签字符串。

（本项目使用本地 markdown tracker，标签以 issue 文件靠上的 `Status:` 行记录。）
