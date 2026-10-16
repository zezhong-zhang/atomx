# 六方向并行推进

六个方向已完成独立实现、源与输入冻结、审阅注册。[前瞻合同](contract.json)固定完整 signed 空间、Hamiltonian、metric、原精度门和整批资源预算。三位科学子任务各负责两个方向，共享调度与指南由主任务整合。

| 方向 | 本轮实际实现和证据 | 尚未闭合的缺口 |
|---|---|---|
| [FS-CCSDT](../../atomic_pci_sd_native_2026_10_09/parallel_six_fscc_scope001/README.md) | 显式修复后端扩大到4core/12spinor，完整495/792行列式与8模型列；保留rank投影约6.5e−14 Ha、完整signed HEFF投影5.1e−15 Ha | T4与attachment Q3/Q4残差约2.1/2.3e−3 Ha；FCI方法差约9.1e−6 Ha；生产J分块、更大空间及其他扇区 |
| [相关性质](../../atomic_pci_sd_native_2026_10_09/parallel_six_properties001/README.md) | 新公共FiniteCorrelatedWave强制完整full-H/metric/parent/operator身份；native E1/E2/M1全部11个signed分量与CI差≤2.7e−15 a.u. | 只认证完整有限wave；截断CC响应、核响应、rates/lifetimes尚未认证 |
| [SD横向图项](../../atomic_pci_sd_native_2026_10_09/parallel_six_sd_diagrams001/README.md) | 新公共scalar T2 source、PP及normal-ordered HH ladders和linked能量；10 doubles、完整495独立Fermi行，DC/static/dynamic/复相位差≤1.4e−15 Ha | Fock/singles/mixed-rank/ring、其余core、valence与Sigma2未补齐；不是完整横向全阶SD |
| [CPU/GPU](../../atomic_pci_sd_native_2026_10_09/parallel_six_gpu001/README.md) | 同源ordered FP64 CPU/CUDA，在原4GiB地址预算下实际跑全部密度rank和DC/static/dynamic PP/HH；逐位相等；完整owned device payload 1,361,840 B | 有其他GPU compute进程，受控负载门未过、speedup=null；小矩阵不证明大规模/端到端加速 |
| [完整Yb norm](../../atomic_pci_sd_native_2026_10_09/parallel_six_norm001/README.md) | 当前helper实际消费原完整五态全部18M、28,218,500核坐标；fresh104case/832component独立锚；原1e−11与64MiB门通过 | 原saved wave的当前norm消费，不是新core replay、根、l4/E10尾收敛或新谱 |
| [NIST](../../atomic_pci_sd_native_2026_10_09/parallel_six_nist001/README.md) | 新可复用共同身份/完整panel/严格matching/coverage/u/covariance接口；消费完整旧五态和全部本地数据库，修复极小u的协方差界 | 四激发3失败、两分裂均失败、四目标线只有2个唯一observed；未知u保留未知；没有新实验精度改善 |

原三方向配置及实际回执见[历史结果](../parallel001/README.md)。当前六项各一线程/120s/4GiB地址，实际总预留20GiB，池上限24GiB，独立服务26GiB/6CPU，额外8GiB主机余量。[首次资源拒绝](failed_preflight001.json)保留：one-minute load 29.19+6超过32 CPUs，因此当时没有发卡或启动任务。随后[整批准入](pre_send_resources.json)通过，门限与空间未改。

[注册审阅](registration_review.json)验证六份handoff、完整payload及所有预声明SHA。FSCC/properties原广域源码包中的无关SD模块随后仅做格式变化；[显式源epoch链](source_epoch_transition.json)保留旧字节和完整AST相等证明。旧frozen证书继续绑定旧epoch；本轮回执运行绑定当前实际字节并重新认证，不修改旧proof来声称新源通过。

[调度33项回归](controller-checks.xml)包括六个真实OS子进程互等屏障、失败隔离、重启持久claim、地址/负载预算及三/六宽度。[最终150项面板](final-checks.xml)与[CUDA依赖变更拒绝控制](runtime-guard-checks.xml)均无失败/跳过。449项CUDA实现/驱动/编译器/loader/libdevice依赖在import前和使用后验证SHA与大小；它们是固定运行依赖manifest，不是完整运行时副本。

[启动卡](start_card.json)及[真实飞书回执](start_card_receipt.json)触发六个真实科学探针。实际六个独立PID均完成，共同重叠0.814115秒，244份完整输出文件（含数组/源码与输入归档）SHA通过。整批实际process/输出SHA与完成后不重复准入证据记录于[最终交付proof](final_delivery_proof.json)；每项结果的report gates、NIST accuracy gates和GPU timing guard独立列示。

[结果卡](results_card.json)与[真实回执](results_card_receipt.json)已发送；重新核验六方向均completed、零重复任务。小时timer仍active，下一次12:00 HKT。

每张成功卡片按当前gap重评所有方向；完成的同source/input/action不重复。卡片列出具体后续实验：更大共同空间/扇区与T4、截断CC完整响应、剩余SD图项、生产热点GPU批量及传输、完整五态共同尾求根、d双重态/分裂与缺失observed/u。未审阅的新实现继续拒绝自动执行，不能将一个有限探针套用为生产求解许可。最高水准目标尚未实现。
