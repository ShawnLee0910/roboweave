# rc3 本地验收结果

本次完成了 L1/L2 接线、公开包与冻结源码的进程隔离，以及 history v2 记录回放。
候选共 **257 个任务配置、8 个 Common 执行 profile、13 个固定源码 bundle**。
正式 benchmark / production eligibility 仍为 false，所有 profile 的 Live 状态均为 NOT_RUN。

| 改动 | 结果 |
| --- | --- |
| 公开包边界 | CLI / Python API 使用固定 JSON 子进程操作，公开进程不加载裸名旧模块或改写 `sys.path`。 |
| L1 A/M/V/P 与 L2 | 各自冻结源码、准备依赖、原生准入与结果归档接通；L2 37 题包含 23 Symbolic、14 Physical。 |
| 过程与 history | v2 initial → before → after → observe → final 真值顺序调用原 `ProcessAudit`；缺失证据拒绝。 |
| 结果证据 | 全部文件核验路径、大小、哈希和格式；V 身份、过程报告与最终评分交叉绑定；L2 保持自身联合验收。 |
| L2 超时 | 7200 秒从模型 worker 创建起算，仅在原安全阶段发送一次 SIGINT，保留原始暂停结果。 |

完成的验收：

- 整体 CPU：**1196 passed，3 skipped**；最后修改后的相关回归：**259 passed**。
  两次运行合计覆盖 **1275 个不同的通过测试标识**，不把重复运行相加。
- 新接入家族的 **157/157** 真实私有输入 CPU 迁移和原生静态检查通过；临时私有副本已删除。
  既有 L0 100 题检查保留历史证据，本次未重新导入。
- L0 固定输入行为对比 **75/75** 通过；不宣称这代表所有版本的行为对比。
- 已安装 wheel：8 个 profile、36 条 CPU 命令通过，75 项 L0 行为对比再次通过。
- wheel 内 2483 个包文件与当前源码逐字节一致；未发现私有数据实物、受限 NVIDIA MDL 或凭据模式。
- 源码包的 **2573** 份源码、测试、工具、配置、示例和文档与当前目录一致；包内 **77 项检查通过**。
- 原始三个源码目录的 HEAD、已跟踪状态和原有修改文件哈希均未变。

独立审查修正了 V Runtime 根目录绑定、V 公开/原生任务身份混用，以及官方源码预检遗漏未跟踪文件。
M 两题的迁移 JSON 改用紧凑写出，避免超过原 2 MiB 读取上限；未放宽加载器限制。

完整记录：[机器摘要](VALIDATION_SUMMARY.json)、[整体回归](pytest.log)、[最终相关回归](final-related.log)、
[AMV 输入检查](AMV_INPUT_CHECK.json)、[P/L2 输入检查](PHYSICAL_L2_INPUT_CHECK.json)、
[wheel 验收](WHEEL_VALIDATION.json)、[源码包验收](SDIST_VALIDATION.json)、[分发审计](SOURCE_AUDIT.json)。
源码包内的验收记录是构建时快照；最终产物哈希由源码包外的对应验收记录给出。

按本轮要求未复现仿真环境、未启动 Live/GPU/真实模型，未推送或发布。
Common 接口不替代历史 CODEX/AgentCLI；L2 watchdog 仍有原观察与发信号的竞态，不能当作硬截止。
许可证及私有输入再分发尚未解决；静态和 CPU 通过不能证明真实场景执行成功。
