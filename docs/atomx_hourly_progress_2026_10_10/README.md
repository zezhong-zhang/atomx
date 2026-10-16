# 飞书每小时任务与谱学对照

当前主线为[原生 MCDHF→RCI 与 Yb 实算](../yb_mcdhf_rci_2026_10_11/README.md)。
卡片从 `yb_mcdhf_result` 的独立输入链读取共同轨道五态驻值、模型与 RCI/NIST
各行；未通过原驻值门时显式标为诊断。历史 SD 另列保留。六方向调整为驻值
正确性、平衡核价/多参考、同态 RCI 修正、性质、实验/软件比较与性能。
选择器对当前 MCDHF proof 独立核对 SHA，旧 SD 路径继续兼容。原先注册的
有限复验服务仍按源变化运行，新 MCDHF 核价生产执行器尚未注册；卡片如实
区分 agent 正在进行的物理判别和服务已经准入的有限实验。

当前 Yb⁺ 五参考共同轨道 MCDHF 已从原初始化在原驻值门下收敛，残差
5.143e−8；主表明确显示五参考基线。另列同轨道平衡单核空穴家族
294→468→531-CSF 的实际 RCI 差能和 NIST gap，四条目标谱线仍未过原 1% 门。
扩展模型未完成 MCDHF 轨道优化，不能继承五参考驻值认证。独立同 003 轨道
Breit/QED 量级账保留为另一个诊断模型，不拼入 005/531 预测。

2026-10-11 最新用户裁定与[原生路线](../atomic_native_roadmap_2026_10_11/README.md)
将待办改为Atomx原生六方向；其他软件仅作方程、物理、精度和效率独立对照。
Ne成功基线改读后期3200Ha实际JSON，检查能量/参考/残差单位一致；Eu改读
来源绑定的历史标定/本地NIST半经验审计，输入变化拒绝显示旧结论。原早期
n12和裸CI记录保留，不能代表最佳实践。原生linked SD三模式新控制已实跑；
原生生产SD/CCSDT executor尚未注册，发卡仍按原有有界选择器如实显示前沿。

用户于 2026-10-10 要求将待办和各种计算谱／实验谱对照列成飞书卡片，
每小时同步。已安装用户级 systemd timer，使用当前 `aisi-codex` bridge 的应用身份，
仅发送到本次授权的私聊。卡片为 CardKit 2.0 展示卡，不含回调按钮。

[进度登记](progress.json)列出大规模 SD、范数、动态 Breit、相关尾部、
FS-CCSDT、性质算符和性能任务；八类谱区分实际实验对照、跨软件对照和未执行计划。
[发送器](sync_card.py)每次重读源证据，记录注册表、渲染器及证据 SHA256。
Yb 能级、历史寿命、Ne X 射线位置及动态 Breit 数字直接从保存的结果生成。
静态叙述和待办须随实际科学工作更新。2026-10-11 起，每次实际成功发卡后，
回执键持久化为迭代请求，并启动独立 systemd 科学服务；预览不触发计算。
当前服务先按实际 gap 选择方向，再以总预算准入独立科学任务并行执行；不可用的实现前沿明确报告。
有限 FS-CCSDT 的更严精度门和 strict xfail 也从保存的007／009结果重新读取，
避免只报旧容差的通过数而隐藏后续精度缺口。

2026-10-11 的[当前物理正确性审计](../atomic_physics_correctness_2026_10_11/README.md)
调整待办优先级：固定空间先验完整signed方程与独立二阶极限，补齐重要缺失
图项／激发秩／模型耦合，再做g分波和能窗筛选。[原量级账](../atomic_physics_budget_2026_10_11/README.md)
保留为候选物理响应证据；历史小池、文献和冻结CI的动态Breit量级分开，
不相加为当前修正。现有 `physics_budget_result` 字段兼容读取新审计汇总并绑定SHA；
新生产判别尚未注册为executor，发送卡片不等于已启动新Yb根，受控机制保留。
[批次选择器](parallel_planner.py)核对审计与当前五态比较的身份并重验全部绑定输入；
在实现前沿时，将物理正确性与覆盖判别作为生产方向；有已准入的源变化有限
复验时先完成复验，同时保留这一生产方向和各独立实现前沿。
这项优先级注释不创建任务、不改变现有任务或科学门。

本轮源变化已触发并完成四核 FSCC、有限性质和完整五态范数复验。NIST 原任务
正确拒绝旧范数证明；现增加显式 `--norm-run` 依赖选择，保留历史默认和旧失败，
注册路径指向当前实算证明。全部原审计门通过，实验精度门仍失败。
见[当前依赖修复与完整身份链](../atomic_physics_correctness_2026_10_11/hourly_dependency/README.md)。

## 已验证的实际发送

- 原生 Yb⁺ 五参考 MCDHF 驻值及平衡 531-CSF RCI 最新结果已发送并读回，
  [第二阶段回执](../yb_mcdhf_rci_2026_10_11/delivery_stage002/receipt.json)记录
  `om_x100b63807e496ca0dfa2a1789aa5dd8`。发卡后选择器实际运行，状态为
  实现前沿，没有将待实现的扩展 MCDHF 标为生产计算已启动。
- 后继 531 混合 CI 驻值／真实轨道步完成后，
  [第三阶段卡片回执](../yb_mcdhf_rci_2026_10_11/delivery_stage003/receipt.json)记录
  `om_x100b638022c610a4c1180808be04364`；原驻值门未过、DC 一步与静态 RCI
  分开。选择器实际运行但没有新准入生产 executor，保留实现前沿状态。

- 手动服务启动：香港时间 2026-10-10 23:57，消息
  `om_x100b639246d34ca4c4fd515e6ad693c`。
- 实际整点自动触发：香港时间 2026-10-11 00:00，消息
  `om_x100b63925ec128a0c2468294c17b43d`。
- 动态 Breit／FS-CCSDT 验证后更新：香港时间 00:24，消息
  `om_x100b6392248d80a0df31cfebec8371d`。
- 非 canonical 更严精度门诊断更新：香港时间 00:38，消息
  `om_x100b6392ef50c8a4dfa9dcd8e53ad50`，明确展示仍未通过的目标。
- [部署与回执摘要](deployment.json)记录 timer 状态、服务结果、实际回执及文件身份。

timer 名称为 `atomx-lark-hourly-aisi-codex.timer`，按
`*-*-* *:00:00 Asia/Hong_Kong` 运行；已启用 `Persistent=true`。
服务是独立的一次性前台进程，超时 180 秒、内存上限 256 MiB，
不依赖 agent 回合内的后台 shell。
当前用户管理器 `Linger=no`：机器和用户管理器须在运行，离线期间不能即时发卡；
恢复时 Persistent 允许补一次遗漏触发，不重放每个遗漏小时。

## 维护与排查

用户级 unit 位于 `~/.config/systemd/user/`。私有配置和回执位于
`~/.local/state/atomx-lark-hourly-aisi-codex/`，目录权限 0700，配置 0600。
配置只存聊天 ID、CLI 路径及 bridge 环境路径；不复制 token 或账号密钥。
保留全部 channel 环境，发送前 dry-run 验证应用身份；若 bridge 未绑定或身份失效，
记录失败并退出，不自动授权、bind 或改用普通 profile。

```bash
python3 docs/atomx_hourly_progress_2026_10_10/sync_card.py \
  --config ~/.local/state/atomx-lark-hourly-aisi-codex/config.json --render-only
systemctl --user status atomx-lark-hourly-aisi-codex.timer
journalctl --user -u atomx-lark-hourly-aisi-codex.service -n 30
```

发送采用当前 CLI 的 `im +messages-send`，消息类型为 interactive。
按聊天与 UTC 小时生成幂等键，文件锁防止并发重复；同一小时的主动阶段更新可用
`--cycle <唯一阶段名>`。`latest_card.json` 可能仅为预览，实际成功以
`last_success.json` 和对应 `receipt_*.json` 为准，`last_status.json` 另保留最新失败。
停用同步使用 `systemctl --user disable --now atomx-lark-hourly-aisi-codex.timer`。

## 发卡后按 gap 选择迭代

当前 [选择器](iteration_planner.py) 与 [策略](iteration_policy.json) 在每张卡片
渲染时读取带哈希的证据，送达后在串行领取锁内重新读取，领取完整批次后并行执行。卡片显示选择依据、
被延后的低信息方向与下一任务；执行采用领取时的最新证据，预览不是执行回执。
高精度、高效率、高复用、模块化与架构自洽为持续选择约束：

- 精度：完整 H／态／空间与原门限优先，不把算法残差、参考记录或 changed-H 诊断提升为实验通过。
- 效率：同模型至少三个径向分辨率下的未通过差距相对展宽不超过 2% 时，停止低信息网格扫描；执行前核对内存、磁盘和负载。
- 复用：沿用不可变 signed 积分、完整 Bloch 数组与公共积分 ABI，不为改变强度重建径向轨道。
- 模块化：选择器只作决定，worker 只负责回执／资源／生命周期，科学任务复用公共物理消费器。
- 自洽：平台按 Z、核、后端、完整磁空间、物理源与频率处方分组；改变 H 的对照各自使用自己的完整 FCI。

[原径向队列](iteration_queue.json)声明 12 个任务：ns=42/60/80/100 × 动态／静态／DC。
它只在尚无误差平台时作为补充对照使用。2026-10-11 08:00 已完成其中 10 项；
ns=42/60/80/100 的动态差距相对展宽仅 7.36e−6，剩余径向任务因此延后，
没有删除或改写旧结果。新方向转为 [完整算符分解与全横向强度对照](run_gap_step.py)。

旧径向任务
每轮使用 Z=4 的完整 1s/2s/2p 磁空间、两核电子、八个价虚自旋轨道，
重新计算 DF、虚轨道、完整 signed 积分和真实后端 CCSDT；独立指数核簇／Bloch
算符对照保留全部 120 个三电子行列式和八个模型根。
这是非 canonical 方程定位与径向细化，不冒称大规模 Yb 实验精度认证。
更严 1e−10 Ha 目标即使失败也保存，不修改门限。

[worker](iteration_worker.py)在锁内领取批次、独立子进程并行执行，保存实际卡片回执、队列哈希、输入、日志、
运行前源码快照、完整有效矩阵和结果哈希。任务脚本
[run_iteration_step.py](run_iteration_step.py)拒绝未收敛核 DF，另外记录其停止残差；
该停止残差也不等于完整态精度。后端一次 120 秒／512 MiB／一线程，
任务总超时 600 秒、进程地址空间 4 GiB；独立服务当前为 26 GiB／六个 CPU 配额，最多六个独立任务，各一线程；旧单任务仍为 4 GiB 地址上限，
`KillMode=control-group`。当前有限基准适合 CPU；不据此宣称 GPU 或端到端加速。

服务：`atomx-atomic-iteration-aisi-codex.service`；持久请求监视：
`atomx-atomic-iteration-aisi-codex.path`。它们独立于 bridge 子进程；当前机器与
用户管理器仍须在线。重复回执不重复科学任务；中断领取标为 interrupted，
失败保留且不自动重跑。现在可执行诊断完成后显示 `implementation_frontier`，
输出需要补齐的图项、生产空间、性质算符或性能证据；证据无效显示
`waiting_for_evidence_repair`，资源不足显示 `deferred_resources`。
没有安装策略文件时才保留旧队列耗尽状态 `waiting_for_queue_expansion`。
这个流程自动选择并执行已经注册的科学实验，尚不能自动推导任意新方程或无限修复代码。
它每次仍重新审视实际 gap，禁止重复旧扫描冒充进展。

控制器的源码、策略和进度登记另保存为私有 `iteration_decisions/*_sources.tar.gz`；
每项科学动作保存其源码、job、输入哈希、完整数组、原始后端结果与失败。
规则不执行任意生成的 shell 或未经验证的方程；资源上限不因预估失败而自动缩小物理空间。

实际验证：

- 00:59 手动卡片 `om_x100b6392a1efeca4c3387195540140b` 触发
  [ns42 dynamic](../atomic_pci_sd_native_2026_10_09/hourly_iterations/finite_ns42_dynamic/result.json)。
- 01:00 整点卡片 `om_x100b6392bfd650a4c346fae28784204` 触发
  [ns42 static](../atomic_pci_sd_native_2026_10_09/hourly_iterations/finite_ns42_static/result.json)。
- 两轮独立 Bloch 差分别 6.312e−15／5.914e−15 Ha；后端完整有效矩阵差
  8.750e−10 Ha；原 1e−9 门通过，更严 1e−10 门仍失败。

卡片读取 `iteration_status.json`，显示上轮真实结果／失败；当前新卡触发的
任务完成情况在后续卡片同步。远程 NIST 页面为已发布快照，需要登录对应
ChatGPT 账号；后续小时任务不自动重发云端网站版本。

本次 08:17 起的三次真实卡片均自动选择并完成新动作：完整 signed 核孔分解闭合
8.882e−15 Ha；全横向 λ=0.5／1.5 的 CCSDT／FCI 差分别为
4.402e−10／1.304e−9 Ha，相对原差为 0.5031／1.4907。
下一方向由这些实际响应决定，优先审计非 canonical 核—价耦合的线性依赖项。
2% 线性响应偏离只用于选择审计方向，不是物理精度通过门；这些 changed-H 对照
不证明某个图项缺失，也未修复原 1e−10 Ha 目标。
详见 [本增量证据与边界](adaptive001/README.md)。


## 三方向并行首轮（2026-10-11，历史配置）

用户明确要求并行推进后，已加入[批次选择器](parallel_planner.py)、
[固定探针登记](parallel_registry.json)和[有界执行器](run_registered_step.py)。
旧 gap 规则和原严格精度门保留；注册表只准入已审阅的独立 Python 探针，
不能生成任意命令或把预检通过当作生产方法完成。三方向是实际幅度方程审计、
横向 J 耦合适配／有限 SD 锚、完整 Yb 尾空间资源预检。

首轮一张成功卡片最多准入三项独立任务，单线程 BLAS／Numba／OMP；
每探针 120 秒／4 GiB 地址上限，总预留 12 GiB、服务 14 GiB、主机额外余量 8 GiB。
负载、CPU 亲和性、主机内存和磁盘须在领取前通过；大规模 Yb 根计算尚不在这个
有界探针注册表中。来源／物理输入／科学门限决定任务身份，完成的相同任务不重复执行。
每项有独立 source/input archive、回执、日志、产物哈希与门限；
超时终止整条子进程组，失败不取消已完成同批任务，重启只标记未完成领取为中断。

[并行增量证据](parallel001/README.md)记录实际卡片批次、资源合同变化、
源身份和回归。有限图项修复、原生完整横向 SD 和生产尾部收敛分别判断。

## 六方向并行增量（2026-10-11）

用户随后明确要求六方向并行，当前登记和服务上限已改为六项；
[前瞻合同](parallel_six002/contract.json)保留旧配置与当前机器资源，固定原物理空间和精度门。
六方向为FSCC扩大核／扇区验证、完整度量与相关性质、原生SD横向图项、
CPU/GPU完整signed消费与实际准入、完整Yb有界范数／尾空间消费、NIST匹配覆盖与误差闭环。
每方向先有独立实现、同H控制与完整source/input合同，再由主任务审阅后注册；
未完成实现保持disabled，不会以旧重复扫描充当新增方向。

每项有限探针仍为一线程／120秒／4GiB地址上限，六项实际预留20GiB、池上限24GiB，服务26GiB，
主机额外余量8GiB。GPU可用性与显存属于独立实际准入；受控计时还须另过负载门，
不能将CPU进程数或有限kernel计时换算为端到端加速。三／六配置均有回归，
[当前控制器检查](parallel_six002/controller-checks.xml)包括六个真实OS子进程相互等待的屏障，
验证没有隐藏的三任务串行限制。失败隔离、持久身份与超时后代清理沿用原契约。

[六方向实际结果](parallel_six002/README.md)已由真实回执启动并全部完成；
六个独立科学PID共同重叠0.814115秒，244份数组/归档等输出SHA与151项回归通过。
[交付proof](parallel_six002/final_delivery_proof.json)保留最初CPU资源拒绝、随后整批准入、
源epoch链、NIST失败的accuracy gates与GPU失败的timing guard。已完成身份重评后不重复运行；
每方向卡片给出具体后续gap，未经审阅的新生产求解不能自动沿用有限探针预算。

Yb 主线现为原生 MCDHF→RCI。[最新卡片回执](../yb_mcdhf_rci_2026_10_11/delivery_stage003/receipt.json)
展示五参考驻值通过、531-CSF 扩展驻值失败及其真实轨道步；实际选择器报告
实现前沿，没有伪报新增扩展 MCDHF 生产执行。
[远程周期表版本 2](../yb_mcdhf_rci_2026_10_11/atlas_update/PUBLICATION.md)
已成功发布同源谱对照，保留全部原始 NIST 和历史模型。
