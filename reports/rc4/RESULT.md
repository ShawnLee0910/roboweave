# rc4 独立审查与修复

本轮按运行入口、资源迁移、评分、轨迹归档、源码分发、模型可见信息六个方向并行审查，并交叉核对关键修复。
审查基点为 `744bb80990c72cc0311fb62d39f177bb3e3ffead`，同时核验并保留开始时已有的未提交修复。
修复版为 `0.1.0rc4`；257 个任务、8 个 profile、13 个源码 bundle 的上游版本与任务合同保持冻结。

## 已确认的问题和修复

| 优先级 | 修前可触发的问题 | 修复 |
| --- | --- | --- |
| P1 | 源码哈希通过后仍执行同时间戳的异内容 `.pyc`；外部 `lab` 包可抢占或扩展模块 | 冻结模块直接从源码编译，按既有文件索引解析；7 个动态 verifier 加载点同样采用该 loader。第三方加载器保留原行为。 |
| P2 | 源根和中间目录的软链接被 `resolve()` 隐藏 | 在解析前拒绝注册路径中的软链接。 |
| P1 | materialization 预检之后的初态改动仍能配上冻结哈希进入回执 | 对实际读取的同一份字节重新验 SHA256 后才解析和写出。 |
| P2 | 并发导入可覆盖其他 profile 归属，或在拒绝之前修改竞争者目录权限 | 源校验后独占创建目录，在目录锁内核验归属、目标内容并写回执。 |
| P1 | 低层入口可接收物化后改过的指令、公开 ID 或预算 | 准入、runtime 和模型入口共用 catalog 绑定校验；启动回执关联 profile、任务及原生 instruction。 |
| P1 | 归档可穿过私有目录软链接、覆盖任务绑定，或接受重复 JSON 成功字段 | 复用安全路径和严格 JSON 检查，拒绝改写已有绑定与证据；验证全部证据哈希和大小。 |
| P1 | P/process 缺 `task_success` 被默认成功；L2 联合成功可缺支持报告 | 对照原 collector 显式校验成功门槛，拒绝矛盾成功。 |
| P1 | 完整性失败可被同一评分记录后续快照清除 | 保持该 episode 的故障状态，后续不会恢复为成功。 |
| P2 | 互补谓词或 milestone 与 `_unary` 的矛盾读数可掩盖违规 | projected/process 共用互补状态检查；v1 扩展路径用原 `MilestoneProgress.read()` 核对里程碑。 |
| P2 | 旧 P 版本的合法单 milestone 因缺新常量报 ImportError | 按实际冻结版本读取扩展能力；继续拒绝旧版本不支持的 history/cycle。 |
| P1 | 提供商错误回显的 Bearer、JSON 或转义密钥进入错误预览 | 用实际发送的凭据和标准诊断格式脱敏；保留捕获错误体原有长度与 SHA256。 |
| P2 | Anthropic 凭据继承到 Simulator；模型名规范化差异使 watchdog 失效 | 清理子进程默认凭据；记录与 watchdog 使用 provider 的实际模型名。 |
| P1 | 实际环境异常退出缺正常终态字段而被误判为损坏数据 | 只接受各原版本明确的负分数 abort schema，要求恢复证据，保留 V abort 收据。 |
| P2 | legacy 记录忽略已核验的负分数而采用旧 summary 成功标志 | 原始已核验分数优先；unknown 模式继续排除在 Live 比率之外。 |
| P2 | 已归档预算、history 或 experiment 与原生记录不一致仍通过 | 交叉核对 run record、client manifest、summary、任务绑定及固定 catalog。 |
| P2 | `--mode replay` 仍先按 Live 校验；目录重叠或非法实验名可能加载场景后才失败 | 覆盖模式后再校验；在读取任务资源前拒绝输出/工作目录冲突和无效实验 ID。 |
| P2 | 远端凭据未声明或实际未设置，仍先消耗场景加载预算才失败 | 配置检查要求必需的变量名；实际 CLI Live 在读取资产前检查变量非空，离线配置和 dry-run 不读取凭据值。 |
| P2 | 重复 JSON 字段被静默取最后值；非普通输入文件可能阻塞 | CLI 评分/映射使用严格 JSON，先检查普通文件及大小；逐条异常不打断其余结果汇总。 |
| P2 | 分发审计漏扫顶层构建/说明文件、扫描根软链接及非 Python 文件中的历史路径 | 扩展到实际分发范围，仍仅输出位置与类型。 |

开始时已有的全 profile `pause_on_client_failure=True`、KeyboardInterrupt 协议响应、episode/initialization 失败区分及 wall-timeout 汇总也已核验。
V 协议失败仍保留在原始有评分结果的统计中，同时显示原生 `outcome_class/formal_score_eligible`；没有删除失败分母。

## 验收证据

完整 CPU 集成回归 **1593 passed、3 skipped**；最后增加的凭据前置检查已补充受影响模块回归 **92 passed**（[日志](final-related.log)），未把重复的定向测试相加。
已安装 wheel 的 **8 profile、36 条命令**通过，默认 pip 字节码缓存保留；L0 固定输入对比 **75/75** 通过。
3 个跳过项仍是未分发的私有 L0 资源与两个历史任务包。

最终分发结果以 [机器摘要](VALIDATION_SUMMARY.json) 为准。独立专项包括：

- [启动参数](STARTUP_ARGUMENTS.json)：8 profile、9 条原生路线的 CPU 参数解析。
- [正常请求边界](PROMPT_BOUNDARY.json)：46 个 API／history 组合，184 个实际序列化请求体；16 项历史裁剪检查。
- [凭据独立复验](CREDENTIAL_RECHECK.json)：实际 provider 错误、model loop 和 preflight 落盘链路，使用假密钥及内存 HTTP 替身。
- [轨迹与原生失败 schema](TRAJECTORY_REVIEW.json)、[运行入口](RUNTIME_REVIEW.json)、[源码身份](SOURCE_IDENTITY_CHECK.json)。

本轮复用已有 CPU 环境，未启动 GPU、仿真、真实模型、SSH、resident 或队列，未重导私有任务包，未推送发布。
场景/评分替身和提供的谓词记录不证明 Live、物理真值或外部录制真实性。直接模型 API 的调用方仍须提供经核验的 seed；CLI 通过原生准备输入校验 seed。
L2 watchdog 保留原安全阶段观测到发信号之间的竞态。许可证和私有输入再分发条件仍未解决，全部正式资格标记保持 false。
