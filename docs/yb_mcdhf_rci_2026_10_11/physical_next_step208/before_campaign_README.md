# Atomx 原生 MCDHF→RCI：Yb⁺ 与精度、效率主线

用户于 2026-10-11 明确选择原生 MCDHF→RCI 为原子结构开发主线，Yb⁺ 示例
实际采用 MCDHF。其他软件用于参考方程及独立物理、精度和效率对照。此前
[原生路线](../atomic_native_roadmap_2026_10_11/README.md)中的 Ne、Eu 成功基线、
Yb SD 失败谱与完整 signed 方程证据继续保留；SD/CC 不再是 Yb 示例的主线。

[本轮合同](contract.json)区分轨道目标、CSF 空间、no-pair 投影、最终 RCI 算符
和观测态，保持原数值门与实验 1% 门。此前指南、进度和实际源文件的字节快照
见[前版清单](previous_sources/manifest.json)。

## 当前实际结果：29轨道 / 1046目标CSF，累计208步，原驻值仍失败

当前已闭合末态是 [run005 local128 / global208](correlated29_optimization_run005_preparation/RESULTS.md)，
在同一 f043 源、原七个核 no-pair 子空间、原五态 UID 等权0.2及
**3451坐标原逆度量**下，共同优化全部29轨道和五个完整目标扇区1046有符号CSF。
原残差 **0.0009553265963620529 > 1e−7**，能量指示量
**1.4594365307472928e−6 Ha > 1e−8 Ha**，两驻值门仍失败。
四线gap RMS为 **2054.3164606344617 cm⁻¹**，低于第80步2105.099，
仍高于本29轨道模型最初2048.365；D3/D5/P1/P3 gap为
**+2936.98/+2438.91/−1096.18/−1051.26 cm⁻¹**，原实验1%门均失败。
声明父空间2163 CSF的其余扇区未计算，没有全谱、全局基态或完整物理认证，
未移植旧 Breit/QED/反冲修正。此次新增128接受步、全部129点与8233实时控制通过，
[完整闭合](correlated29_optimization_run005_preparation/completion.json)保留全部有限stored-H接受账。
实际187H/187CI/129梯度、14hS/7验证Dirac；858.410秒是未受控墙钟，不是加速比。

[第二十二阶段汇总](results_stage022.json)与
[280项元数据整合](reporting_stage022/integration_proof.json)保持上述物理范围；
报告整合没有新增科学计算。远程周期表[版本7实际发布](atlas_update006/root_publication.json)
已追加第208步末态及完整129点，总19模型76行、209个不同global点。
原18模型72行、全部6019 NIST gzip、hosting及页面源码保持，未按最小gap选点；
[本地准备](atlas_update006/README.md)与[发布说明](atlas_update006/PUBLICATION.md)区分本地验证和root发布。
[版本6](atlas_update005/root_publication.json)作为历史发布证据保留，不重写旧卡或冻结阶段内容。

stage022卡片成功送达后，root按独立合同执行[private径向复用资格003](correlated29_radial_reuse_qualification003_preparation/actual/result.json)，
1126项完整H/CI根、signed密度/梯度及原度量比较通过，实际2H/2CI/2g、88.298秒未受控总耗时。
这是原corrected29同点的新私有数值源资格；没有新增轨道优化、没有共享采用或性能认证。
[原资格002失败](correlated29_radial_reuse_qualification002_preparation/README.md)仍按原epoch保留。
另[保存g/pg分解](correlated29_saved_gpg_decomposition_preparation/actual/result.json)149项控制通过，
只检验第80/208步保存坐标代数：g·pg/2是原能量指标，不是谱线误差界，
不能从残差、分块贡献或整体总能推出哪项物理缺失主导约2000 cm⁻¹ gap。

[stage022实际卡片读回](delivery_stage022/delivery_proof.json)绑定第208步报告与之后独立资格003的真实退出；
[00:00小时卡](delivery_hour24/delivery_proof.json)实际发送的是stage021第80步内容，按原范围保存。
两次服务均为implementation_frontier/job=null/jobs=[]，不冒充自动物理作业。
零点小时卡没有新的独立native launch，已有并行source/math准备持续；不把此前run005写成零点新启动。

以下531/320步、run003/16步、run004/80步及早期导数、积分、固定轨道对照均为各自源和模型的历史记录，
不作为当前29轨道第208步末态，也不把不同维度量残差直接排列为收敛速度。

历史完整 531-CSF 模型曾通过公共轨道优化器累计接受 **320 步**。
[第二轮终点独立重验](public_mixed_resume_run003/independent_endpoint_verification.json)
1,011 项来源、完整 H/CI、signed g/pg、原 P/逆度量与态身份核验通过。
原残差从第160步的 4.71705e−5 降到 **1.94348e−6，仍未过 1e−7**；
指标 **9.22537e−12 Ha** 通过 1e−8 门，不能据此认证驻值。
第二轮160步的四条 DC 跃迁改变量仅为 D3/D5/P1/P3
**+0.099/+0.321/−0.350/−0.525 cm⁻¹**，实验千cm⁻¹差距仍在。
[第二十阶段汇总](results_stage020.json)记录修正源基线、共享积分修复和真正的
29 共同轨道迭代；[第十九阶段汇总](results_stage019.json)保留积分修复的五态量级判别；
[第十八阶段](results_stage018.json)保留完整模型、完整密度梯度控制与
共同求积缺陷的原失败；[第十七阶段](results_stage017.json)记录新29轨道的
完整归一化密度梯度实算，[第十六阶段](results_stage016.json)保留条件约束检验和
接受步算术保护，[第十五阶段](results_stage015.json)保留保存审核与
原旋转导数失败，[第十四阶段](results_stage014.json)及
[21:00小时卡片读回](delivery_hour21/delivery_proof.json)保留当时范围。
第十三阶段的[汇总](results_stage013.json)与[实际卡片读回](delivery_stage013/delivery_proof.json)
保留屏蔽完整CI、586项独立复核与19项He有限增量保护资格的原范围。
[第六阶段卡片证据](reporting_stage006/current_card_evidence.json)按原范围保留。
下文五参考及各阶段归档均保持各自物理范围。

历史 run003 的29共同轨道/1046完整目标扇区完成[16步原生联合优化](correlated29_optimization_run003_preparation/RESULTS.md)。
17 个接受点、全部 16 个有限 stored-H 下降检查及 1175 项实时控制通过；
实际 27H/27CI/17 梯度，前台 116.92 秒，属于含检查与归档的非受控耗时。
等权五态 EOL 降低 0.01299662 Ha，但同一原度量残差 **0.021499→0.040910**，
指标 **0.00340818 Ha**，均未通过原驻值门。四线 RMS **2048.365→2203.862 cm⁻¹**；
6s 下态降低较多，使四条激发能全部升高，D 谱差增大、P 谱差减小。
全部点的上下态、重心和分裂保留；后继从第 16 步终点续算，不按最小实验 gap 挑点。
这些残差属于新 3451 维度量，不与旧 531-CSF 轨迹直接比较优化速度。
两处内积分阶数修复已进入[共享模块](rank_aware_cell_discriminator_preparation/shared_fix_implementation/RESULTS.md)，
三个独立有符号解析测试通过；原高阶解析失败归档不改，未认证全局积分误差或加速比。

新29共同轨道、完整1046目标扇区的[实际梯度探针](correlated29_mcdhf_probe_preparation/actual/progress.json)
完成4次H、4次CI、1次完整梯度、7次验证Dirac及28次h/S；279项通过后，
首个reported-root中心导数误差1.26484e−6超过原1e−8门，立即停止。
原Rayleigh门及第二旋转方向未执行，零优化器调用，原失败保持不变。
随后[341项独立保存数组审核](correlated29_mcdhf_probe_preparation/saved_array_audit/verification.json)
确认原P、完整3451坐标及复制逆度量的g/pg重构，驻值残差为
**0.0214991**、指标为**1.20703e−4 Ha**，两门均未通过；这属于新度量，
不能与原320步残差直接排列收敛速度。
[Decimal80能量分解](correlated29_mcdhf_probe_preparation/saved_array_audit/decomposition.json)
独立确认reported−Rayleigh导数差主要来自保存CI端点的归一化残差偏移；
归一化保存H商的导数误差6.470e−9在原1e−8范围内，但不升级旧运行资格。
共同径向能量变化的完整曲率原因尚未证明。

随后旋转资格[001原失败](correlated29_rotation_rayleigh_qualification/actual/failure.json)
在实时owner的tuple与JSON列表比较处停止：142项通过、1项失败，实际1H/1CI/0g、
7次Dirac与14次h/S。两份完整owner保存字节相同，
[独立表示审阅](correlated29_rotation_rayleigh_qualification/failure_identity_review.json)
确认序列化比较问题；原失败没有改写。仅修此比较的
[002后继](correlated29_rotation_rayleigh_qualification002/actual/progress.json)实际完成
3H/3CI/0g、7次Dirac与28次h/S，222项通过后在原同κ新旧轨道旋转方向的
**归一化能量导数门失败**：解析值0.001494376612，有限增量中心值
0.001494167475，误差**2.09137e−7 > 1.49438e−8**。
[端点分解](correlated29_rotation_rayleigh_qualification002/actual/direction1_diagnostics.json)
保留显式径向、重构和有限CI响应项。随后
[同点单体与运动分解](correlated29_rotation_failure_diagnosis/actual/result.json)
把实际端点的归一化能量导数偏差解释为条件梯度沿微小法向运动的作用，
重构后剩余差约1.30e−13；该阶段只测单体密度系数，完整两体梯度差随后另行实测。
[完整Gram中心化检验](correlated29_constraint_centering_diagnosis/actual/result.json)
保留全部七个同κ块、signed非对角项、有限二次项和CI响应，不改变原点、步长或阈值。
中心导数为0.001494376493974，与原解析导数差**−1.18227e−10**，
通过原1.49438e−8门；全部Gram恒等式通过，实际新增H/CI/梯度均为零。
原外部轨道方向的[独立后继](correlated29_constraint_centering_diagnosis002/actual/result.json)
也以原1e−6步长通过原1e−8门，中心导数与原解析值差**−1.09314e−10**；
两次均只消费原保存点，保留原失败，未接受新的轨道步。
这是同一保存三点、固定条件乘子的Q-Lagrange诊断，
[独立数学审查](correlated29_constraint_centering_review/MATH.md)说明其范围；
不能升级旧reported/Rayleigh失败，也不提供完整归一化混合CI梯度或统一误差界。
[完整归一化密度梯度实算](correlated29_full_normalized_gradient_control/actual/result.json)
已在同一真实owner点完成1H、1CI、2次公共梯度收缩，104项数值检查通过；
完整H、全部signed本征向量与原点逐位一致。逐态归一化、未删小系数的完整密度差
在原逆度量中的范数为**4.51526e−15**，本次实测不支持其解释当前残差。
新的完整梯度中心化方向误差为**1.09314e−10/1.18207e−10**，均通过原阈值。
非对称原始h及ket归一化修正对径向法向分量影响很大，但原切空间差约1.57e−10、
逆度量差约1.64e−13；原驻值两门仍FAIL，有限CI响应及径向重构误差范围仍分别保留。
上述早期归档时29轨道优化尚未启动；当时的[独立后继准备](correlated29_optimization_run002_preparation/README.md)
保持原度量、五态等权和接受步能量保护，461项源码与元数据检查只认证该准备范围。
此后已完成修正源完整H/CI/梯度与原两方向控制，以及run003/16步和run004/64步实际共同轨道更新，
累计208步的当前末态见本页首段。下述固定点积分量级判别也属于其原点，不能升级旧失败或提供
全局误差界；现有末态仍未驻值、未通过实验门，也未认证空间收敛或最高水平。

[接受步元数据实现](accepted_line_search_metadata_implementation/README.md)
只记录公共线搜索实际采用的能量、步长、斜率和原Armijo比较，不改变目标或接受式。
独立有限保存H保护的[19项算术资格](accepted_line_search_metadata_implementation/array_qualification002/result.json)
通过signed干涉、实际归一化、Decimal80商与舍入下降拒绝检查；
原数组选择失败另存，尚未执行Yb新优化器。
[两体势复用提案](correlated29_radial_reuse_design/README.md)复用已有semiseparable核，
保持四分量和完整空间。[唯一资格运行](correlated29_radial_reuse_qualification_preparation/RESULTS.md)
在225项解析锚通过后，于原点rank6单元失败：偏差**3.65867e−5 Ha**，
原完整矩阵核与候选核均有相同偏差。后续Yb H/CI/梯度及阶段profile均未执行，
共享补丁尚未应用，没有29轨道加速比。
[独立数学审查](coulomb_inner_quadrature_review/MATH.md)确认内层GL点数固定为
B-spline阶数，乘r^k后多项式次数增加，非负整数k需要order+k//2个内层点；
该例的相对偏差**1/4900**与实测吻合。外层负幂与负rank的误差仍须分别检验。
[独立修复量级实算](rank_aware_cell_discriminator_preparation/RESULTS.md)只改私有核的两处
内积分阶数；有符号普通/RKB解析锚通过，然后逐源计算全部7398个受影响径向项，
逐态保留全部signed系数及原完整CI范数。五态固定CI算符期望变化约**1.64e−14 Ha**；
D3/D5/P1/P3的上下差能变化分别为**+7.68e−12/+5.38e−12/−1.16e−12/−6.39e−13 cm⁻¹**。
这些名义差接近浮点舍入，不能解释约2000cm⁻¹实验gap，也不构成误差界或新对角化谱。
实际14次h/S、14796次公共径向求值，前台17.23秒；零H/CI/梯度/优化器/GPU，
原失败和驻值门不变。应保留数值修复，但当前优先推进尚未自洽的29共同轨道关联优化。
[本次实际卡片](delivery_stage018/delivery_proof.json)送达并读回后立即执行上述判别；
服务自身的实施前沿状态与root实际科学调用分开记账。
[22:00实际小时卡](delivery_hour22/delivery_proof.json)已送达并唯一读回，保留送达时
第十六阶段内容；本轮后继计算与报告另列。

[完整续算报告](public_mixed_resume_run003/RESULTS.md)与新增的
[4,963项保存数组复核](public_mixed_resume_run003/completion/verification.json)
覆盖全部161个接受点，原坐标/逆度量及g/pg逐位重放通过，未新增物理计算。
最后接受步的记录下降为1个float64间隔；这项观察不能证明舍入导致终止。
公共同点H复用现已[实现](checkpoint_hamiltonian_reuse_implementation/README.md)，
85项小模型数值测试通过，默认关闭；完整531同点资格现已实算，
[1,491项运行检查](hamiltonian_snapshot_qualification002/result.json)和
[3,217项独立数组核验](hamiltonian_snapshot_qualification002/verification.json)通过。
普通/复用路径的完整H、全部本征对、signed RDM及24轨道完整梯度相同；
实际3H/2CI/2梯度的计数与原残差失败保留。首次JSON归档失败及其
[保存数组核验](hamiltonian_snapshot_qualification002/old_saved_array_review/verification.json)
另列，不把旧失败改写为通过。随后[受控五组暖态配对](retained_hamiltonian_performance/actual/result.json)
实际完成21H/14CI/14梯度；链含trial、完整本征对检查、完整梯度、
物理身份、同codec压缩保存/fsync/回读，加速中位数为**1.24259倍**，
范围1.19672–1.29885，对应链用时约降低19.52%，全部负载与物理门通过。
[10,041项独立数组和统计核验](retained_hamiltonian_performance/verification.json)
另证14份完整归档、全部signed根子空间、原梯度与配对统计；两个独立进程冷启动仅为观察值。
这是该受检链的收益，不是整个优化器、GPU或并行效率认证。
[第七阶段证据](results_stage007.json)分别绑定旧科学353源码与新候选354源码，
不改变原驻值或实验失败状态。[第八阶段卡片读回](delivery_stage008/delivery_proof.json)
保留原范围；[第九阶段实际卡片读回](delivery_stage009/delivery_proof.json)
已确认当前320步、核场虚层及径向诊断、复用资格和版本4发布。
[第十阶段实际卡片读回](delivery_stage010/delivery_proof.json)另证屏蔽势矩阵锚、
全谱失败及613项保存数组复核；成功发卡后继续两个有界实现方向。
19:00小时卡另存于[独立送达记录](delivery_hour1900/delivery_proof.json)。
[20:00小时卡实际送达与读回](delivery_hour2000/delivery_proof.json)已同步首κ独立
54项与三尺度113项证据；按实际新卡触发其余四κ/完整屏蔽CI和17项保护复验。
登记复验与新的MCDHF物理实验分别记账。

固定第160步轨道的完整5p双空穴判别已实际完成两次完整目标块CI与四次H，
基线261/全模型531和扩展1240/全模型2483均不按振幅删减。
[运行结果](rank_only_discriminator/result.json)中D3/D5/P1/P3转移增量为
**+461.246/+596.713/+194.486/+148.672 cm⁻¹**，名义实验gap的RMS从
2022.830增至2225.974 cm⁻¹；这是有限非驻值点的固定轨道响应，
不能声称双激发家族解决主gap；
[3,247项独立数组审核](rank_only_discriminator/saved_array_audit/verification.json)
另证完整signed嵌套H、全部根、五态和上下差能，保留D5负相位。
原虚层判别在最低根报值检查处中止，尚未执行其CI；
[独立小诊断](rank_vs_virtual_discriminator/minimum_root_diagnostic/saved_array_audit/verification.json)
122项确认κ=−3的旧/新P补空间逐字相同，raw与最低根向量的Rayleigh能量相同，
但gvx报值偏3.8087e−8 Ha。完整谱报值偏约6.21e−11 Ha，
原1e−8 Ha门未更改。后继已核实全部五个实际向量的Rayleigh、相对残差、
能隙及最低根身份，并在同一第160步点完成平衡虚层实算：旧24轨道逐位保留，
新增5个核场补空间轨道；完整目标块261→1046（全模型531→2163）。
[运行的1,517项检查](virtual_only_discriminator/result.json)及
[3,112项独立数组审核](virtual_only_discriminator/saved_array_audit/verification.json)
通过，旧失败原件不变。四线增量仅为
**−4.782/−4.796/+2.485/+2.511 cm⁻¹**，RMS gap从2022.830降至2019.201。
这个有限核场虚层的收益远小于主gap，不证明径向收敛，也不代表相关驱动的
虚轨道或扩展模型轨道弛豫已计算。
[保存数组的径向与耦合诊断](virtual_only_discriminator/radial_support/result.json)
显示新增核场虚轨道均径0.287–0.348 a₀，原价轨道3.008–4.690 a₀；新增CSF
对角能量间隔最小约50 Ha。对角EN估计只解释这个指定基的有限响应，
不能作为Q空间谱或截断误差界。
[完成决策](virtual_only_discriminator/decision.json)选择检验闭68电子核屏蔽Fock场
生成的虚轨道，保持同一个原no-pair空间、旧24轨道、第160步点和完整1046目标块。
首轮[屏蔽势实算](screened_core_virtual_discriminator/failure.json)已完成10次J/K、
3,584个标量Slater积分；全部80个有符号PP/QQ/PQ/QP独立矩阵锚通过。
首个κ=−3的116根完整谱中49根未通过原相对残差1e−8门，最大为
**3.113086e−8**，因此在CI前停止，尚无屏蔽谱线或RMS。
[613项保存数组独立复核](screened_core_virtual_discriminator/saved_failure_audit/verification.json)
确认最低根实际残差3.021294e−8 Ha主要来自求解器返回向量本身；该根投影与
轨道映射贡献仅约3.35e−14和3.37e−14 Ha，S条件数上界约1.000000000000084。
矩阵高能尺度使低能向量精度不足成为当前数值问题，不能据此归罪于物理屏蔽场，
也不能由重新投影精度推断问题已解决。下一步
[确定性全谱精化](screened_core_virtual_refinement001/result.json)保持原保存算符、
原P/补空间及全部116根，实际一轮6,670个Jacobi旋转后，最大原物理相对残差
**2.046064e−14**、Gram误差**3.107323e−16**，原门通过。没有新增H/CI或积分。
这是首个κ的数值资格，[54项独立保存数组重验](screened_core_virtual_refinement001/saved_array_audit/verification.json)
另证全谱残差、Gram、signed根身份、完整旋转合同、原投影与最低根。
后继[保存场屏蔽虚层实算](screened_core_virtual_discriminator002/result.json)
已复用首κ资格及五个场矩阵，另外四κ各一轮全谱Jacobi通过原门。
运行内1,546项检查通过，实际2次完整CI/4次H、7次内部单体组装，
没有新增J/K、外部h/S、梯度或优化器调用；
[586项独立保存数组复核](screened_core_virtual_discriminator002/saved_array_audit/verification.json)
已通过完整573个辅助根、signed嵌套H/CI、五态、上下差能与RMS检查。
它不重复角动量或物理行列式重叠收缩；原公共tracker输出与独立同点CI系数重叠分别记录。
在同一第160步点、原P、旧24轨道和完整1046目标扇区，D3/D5/P1/P3增量为
**+274.483/+237.067/+240.135/+240.724 cm⁻¹**，RMS gap从2022.830变为
**2048.365 cm⁻¹**。下态降低0.002370059 Ha，上态分别降低
0.001119423/0.001289902/0.001275924/0.001273242 Ha；必须由上下差能判断作用。
D重心增加252.033、分裂减少37.416 cm⁻¹；P重心增加240.527、分裂增加
0.589 cm⁻¹。P的实验差距减小而D增大，整体尚未改善。
这是固定非驻值点的径向表示判别，下一步优先检验相关轨道的自洽MCDHF优化；
不据此认证空间收敛。原失败保留，不把双空穴与虚层增量相加。

优化数值工作并行推进：
[混合CI有限增量提案](mixed_ci_finite_increment_proposal/README.md)已完成413项零物理
准备检查，限定源实现已由
[后继审阅记录](mixed_ci_finite_increment_proposal/root_source_edit_authorization002.json)
准入。[候选实现](mixed_ci_finite_increment_implementation/README.md)现已冻结，
1,189项准备与[1,193项独立源审查](mixed_ci_finite_increment_source_review/verification.json)
通过；保留完整signed径向项、CI系数响应、有限残差和独立归一化，分别报告
本征值报值与Rayleigh差量。有限数值资格已按独立Decimal80算术解释器、13个
完整CI点、18次H及3次梯度的有界合同实算，
[26项通过后导数检查失败](mixed_ci_finite_increment_implementation/actual/result.json)。
完整径向/系数响应与Decimal门、旧H/梯度逐位一致性通过；中心差分
−0.008720635851与梯度−0.008720467438的差超过原门，后续拒绝非法输入的测试
因停止而未执行。[113项保存数组尺度审核](mixed_ci_finite_increment_saved_scale_audit/verification.json)
检查两个方向的全部±1e−4/1e−6/1e−8数据；1e−6误差为
−2.592768e−11和+2.251926e−10，原门均过，粗/中步观测误差阶为1.906/1.986，
支持粗步中心差分截断。该解释不把旧资格改写为通过，也不认证最小步长的误差界。
[1,341项独立保存数组核验](mixed_ci_finite_increment_implementation/saved_array_audit/verification.json)
另证完整13个谱、归一化、signed响应及旧H/梯度逐位一致；没有重复径向Decimal求值。
随后[保护后继002](mixed_ci_finite_increment_guard_qualification002/actual/result.json)
在fixture对照处失败，实际仅3次h/S、0次H/CI/梯度，19项测试未进入。
原保存h是原始单体矩阵，functional采用下三角对称完成形式
`L(h)=tril(h)+tril(h,-1).T`；直接将两种表示逐字节比较是fixture错误。
[138项只读表示审计](mixed_ci_one_body_mapping_audit/actual/result.json)
已核验原P、全部13点和两方向切向中的raw−L影响。完整空间raw−L最大约137Ha，
原P矩阵元最大3.736e−10Ha低于原1e−8门；实际波函数与切向影响低于对应原门，
不能称两种矩阵在全空间严格等价。启动环境错误另存，不计作科学运算。
随后[保护后继003](mixed_ci_finite_increment_guard_qualification003/actual/result.json)
实际19项通过：由公共producer重建一次基线H获得严格owner，全部6个原始h/S、
functional的L(h)与旧H逐位相同；复用保存数据检验两方向h=1e−6原导数门及非法输入拒绝。
实际1H、6h/S、2个初始重叠，0CI/梯度/径向增量。
旧粗步与002失败记录保持不变，不绕过owner拒绝反序列化的原合同。
该资格属于保存的He有限模型，Yb完整29轨道的导数尚待单独验证，尚未接入优化器。
旧fixed-density/constrained门、默认行为和原驻值指标保持原定义。

Yb⁺ 已从原始初始化完成 Atomx 公共 `MCDHFSolver` 五态共同轨道计算：
69 电子、24 径向轨道、固定等权 EOL，68 步后原驻值残差
**5.143009e−8 < 1e−7**，估计能量误差 **3.447850e−15 Ha < 1e−8**。
原坐标与 residual metric 未更换，独立梯度、CI 及原 metric 重放通过。
这是五个 one-CSF 参考的数值驻值基线，尚不认证核价关联、径向收敛或实验精度。
完整来源见[005 实算与重放](native_run/README.md)及
[五参考归档卡片证据](native_run/card_evidence_stationary_baseline.json)。

同一组驻值轨道上的完整静态 Breit RCI 与 NIST 名义值比较如下；实验来源
未提供这些记录的不确定度，保留原 1% 比较门，整体未通过。

| 态 | 五参考 RCI−NIST / cm⁻¹ | 加入 68-CSF 空穴后的谱线增量 / cm⁻¹ | 68-CSF RCI−NIST / cm⁻¹ |
| --- | ---: | ---: | ---: |
| 5d₃/₂ | +796.491 | −2.824 | +793.667 |
| 5d₅/₂ | −107.144 | −3.523 | −110.667 |
| 6p₁/₂ | −2078.221 | −667.608 | −2745.829 |
| 6p₃/₂ | −2593.151 | −597.984 | −3191.135 |

68-CSF 判别只加入 `4f13 6s5d`、`4f13 6s6p`，未重新优化扩展模型的
MCDHF 轨道。全部 13 个 J/parity 块、68 根保留，公共 DC／完整静态 Breit
矩阵与原裸参考 principal submatrix 重放通过；五个目标按完整性格绑定，
6s 仍是此有限模型的真实基态，P 参考权重保持约 98.6%。该家族几乎不修正
6s，却显著降低 P 态，因此这次扩大空间使 P 的实验差距增加。
它支持优先做平衡核价关联的判别，不能支持“扩大截断必然解决问题”。
[固定轨道物理判别](cv_discriminator/README.md)保存原数组、匹配及独立重放。

随后在同一 24 径向轨道池，补齐每个裸参考允许的单核空穴／双价粒子家族，
实际计算了 4f→4f+5p→4f+5p+5s 三个嵌套模型。有限家族完备枚举与算符、
匹配、全部谱的重放通过；这仍是固定五参考 MCDHF 轨道的 RCI，未优化扩展
模型轨道，也不包含核双激发、完整高阶关联或更完整的虚轨道空间。

| 加入家族／CSF 数 | D₃/₂ 线增量 | D₅/₂ 线增量 | P₁/₂ 线增量 | P₃/₂ 线增量 |
| --- | ---: | ---: | ---: | ---: |
| 完整允许 4f／294，相对五参考 | +566.568 | +633.476 | −269.163 | −46.185 |
| 再加 5p／468，相对 294 | +754.054 | +807.891 | +304.286 | +493.788 |
| 再加 5s／531，相对 468 | −34.182 | −31.329 | +15.810 | +16.410 |
| 531 最终 RCI−NIST | **+2082.932** | **+1302.894** | **−2027.289** | **−2129.138** |

单位均为 cm⁻¹。完整 4f 家族恢复了此前遗漏的 6s 基态耦合，基态降低
0.003714 Ha；531 全部五态绑定无歧义，6s 仍为有限模型真实基态。P 差距
较五参考略有改善，D 差距增加，四条均未过原 1% 门。这些同模型差能说明
当前家族扩展的物理影响，尚不能证明轨道松弛、核双激发或径向截断哪项主导
剩余差距。下一步先检查混合 CI 的 EOL 轨道驻值及上下态响应，再估算缺失
关联通道。[平衡家族实算](balanced_cv_replay/README.md)与
[前瞻枚举](balanced_cv_frontier/README.md)分别保留计算和预算证据。

同一 531 模型的 D 重心高于 NIST 1614.909 cm⁻¹，精细分裂为
591.853 cm⁻¹（NIST 1371.890）；P 重心低 2095.188 cm⁻¹，分裂为
3228.561 cm⁻¹（NIST 3330.410）。重心与分裂须分别诊断，不能用总能量降低
或某条线较近来判定整个物理模型合格。

受影响科学回归 **160 项通过**，卡片／选择器回归 **45 项通过**。
修复前的科学失败、编译缓存异常及所有重跑记录见
[测试记录](test_attempts.json)；旧 001–004 驻值失败与第一阶段卡片仍保留。
性能内核 component matvec 从 **1728 降到 224**，signed 梯度逐位相同；
负载准入未通过，没有端到端、GPU 或并行效率认证。

[第二阶段来源绑定汇总](results_stage002.json)已实际发卡并读回，
[回执](delivery_stage002/receipt.json)绑定当前模型和全部可见结果。
发卡后的[实际选择器状态](delivery_stage002/iteration_result.json)为实现前沿：
已准入复验没有新的源变化，未伪报扩展 MCDHF 生产执行。后继有界混合 CI
驻值探针已完成，另存[第三阶段汇总](results_stage003.json)。

531-CSF 混合 CI 等权 EOL 使用原 005 projector、inverse blocks 和 residual
metric，真实轨道残差为 **0.0367607**，一个原 Armijo 接受步后为
**0.0249933**，均远高于原 **1e−7** 门。两方向的双边完整 CI 重解检验
解析梯度；保存矩阵、原梯度和原 metric 的 432 项独立数组复核通过。
平均总能下降 0.000264687 Ha，但该步 DC 的 D3/D5/P1/P3 跃迁分别改变
**+176.099/+225.193/−44.971/−4.972 cm⁻¹**，四条实验差距均增加。
这一步不是完整公共 MCDHF 求解，不能拼入上述静态 RCI 或同 003 修正账。
[混合 CI 驻值诊断](mixed_ci_stationarity_probe/README.md)保留合同、全部候选、
身份和来源范围。运行的 68 项检查副本与另做的数组复核分开，driver 的哈希
为完成后的归档，未声称其在运行前已绑定。

第三阶段辨别出原度量公共续接接口与包含 CI 系数响应的有限能差需求。
原 fixed-density API 对变化的混合 CI 继续返回不支持；不能
冻结密度代替真实目标。实际保存矩阵的一步分解中，径向项约 −0.000583157 Ha，
CI 响应约 +0.000318470 Ha，响应不可省略。该分解尚不验证近舍入尺度的径向
增量内核，归一化缺陷、CI 残差、完整 signed 项和退化子空间仍须独立验证。

[后继公共接口与能差评估](mixed_ci_stationarity_probe/PUBLIC_OPTIMIZATION_SUCCESSOR.md)
保留具体接口边界和完整有限残差公式：45 个真实目标对的 239 项保存矩阵
代数检查通过；这不等于已实现混合 CI 径向稳定增量或完整优化。

公共原度量接口现已实现并完成 40 项独立 Rayleigh／逆度量／检查点边界测试，
旧默认路径的 12 步数值轨迹逐位相同；[实现及验证](public_optimizer_resume/README.md)
与[根代理受影响回归](public_resume_integration/contract.json)另存，受影响的
115 项回归已通过。首次完整 531-CSF 公共入口在优化前因梯度比较点不同而
[中止](public_mixed_resume_run/execution_failure_pre_solve.json)；完整 H/CI、原度量
与裸参考驻值检查均通过，尚未接受新优化步。
[同点旧源码参考](matched_recovered_point_reference/README.md)完成两次梯度和
625 项独立保存数组核验，确认旧保存 g 来自物理轨道而非精确 P×x。
[后继公共续算](public_mixed_resume_run002/README.md)改用真正同点的整向量比较，
保留原 g/pg 阈值 1e−9/1e−10 与首次失败。实际入口的两向量逐元素差均为 0，
公共调用已完成 160 个接受步，因原迭代上限返回；163 次完整梯度含两次入口控制，
202 次能量试算，没有 line-search fallback。约 329 秒是本机资源观察，非受控性能对照。
[独立终点重放](public_mixed_resume_run002/independent_endpoint_verification.json)988 项身份、
Hamiltonian、CI 与梯度检查通过；原残差 **4.71705e−5 未过 1e−7**，能量指标
**6.88972e−9 Ha 低于 1e−8 Ha**，因此不能认证驻值。[第四阶段](results_stage004.json)
保留全部接受点和同态 DC 谱；D3/D5/P1/P3 的 NIST gap 分别为
**+2310.754/+1736.426/−2083.959/−1915.650 cm⁻¹**，仍均未过 1%。
第160步同点的完整静态 Breit RCI 已实算，并独立核验 3,710 项；
四条 static−DC 跃迁改变量按 D3/D5/P1/P3 为
**+2.384/−28.751/−20.697/−44.262 cm⁻¹**。
原第160步 DC 驻值门失败仍保留，这个结果不拼入第320步、固定005谱或003修正账。
完整矩阵、谱、同态上下差能和范围见[同点静态RCI](public531_endpoint_static_rci/README_result.md)。
原 005 最终优化坐标未保存；本次直接采用归档的
recovered point，不称其为旧坐标逐位恢复。两条原驻值门仍分别为 1e−7 与
1e−8 Ha，不从接口测试推断新的 Yb⁺ 驻值、实验通过或性能加速。
[第三阶段实际回执](delivery_stage003/receipt.json)已读回，选择器的
[实现前沿状态](delivery_stage003/iteration_result.json)没有被写成新生产执行。

[远程 Yb⁺ 周期表](https://atomx-atomic-spectrum-atlas.black-fairy-6027.chatgpt.site/#Yb:1:models)
已发布版本7；19个独立模型、76行谱对照保留原始NIST和全部历史行。
[版本7实际发布证据](atlas_update006/root_publication.json)绑定同一提交与私有部署，
新增原生29共同轨道第208步末态及完整129点，保留第16/80步历史末态与轨迹；
原3451度量两驻值门和实验门仍失败。
历史[版本5发布](atlas_update004/PUBLICATION.md)的16模型64行均保持原值；其中同一第160步种子的
核场与闭核68屏蔽场虚层RCI仍是其原固定轨道模型，并未改写为后继优化末态。
原 owner-private 范围不变。[版本4](atlas_update003/PUBLICATION.md)、
[版本3](atlas_update002/PUBLICATION.md)、
[原版本2发布](atlas_update/PUBLICATION.md)与[第三阶段总核验](verification_stage003.json)
另存诊断、后继评估、卡片与站点证据，不覆盖第二阶段科学归档。

完整531与五个完整目标J/parity块261的同点谱、EOL、全部24轨道梯度、
signed CI/RDM和原metric已实算且[独立1,795项数组核验](target_sector_equivalence002/saved_array_audit/verification.json)
通过，差值均为0。范围仅为预声明的五目标EOL；不认证全谱、全局基态或速度。
首次tuple/list身份比较失败及修正后的前瞻合同保留。

第5阶段实际卡片的选择器因仍绑定五参考005身份而进入证据修复前沿。
[身份后继](results_stage005b.json)仅修正当前物理证据绑定并保留已发送原件；
18:00小时卡片成功发出且选择器重新验收当前身份。
可执行队列与需要新实现的物理前沿分别记录，不把0个登记作业写成生产计算。

## 实际路线

1. 固定共同轨道五态 EOL：`[Xe]4f14 6s/5d/6p` 的 6s₁/₂、5d₃/₂、5d₅/₂、
   6p₁/₂、6p₃/₂，以预先声明的 `CIStateAverage` 权重优化完整 CI 密度的
   Dirac–Coulomb 目标。要求真正受约束梯度驻值，不能以能量平台替代。
2. 按同一上下态的能移、重心和分裂辨别核价关联：平衡 4f/5p/5s 空穴、
   价相关和近简并多参考，先估算 CSF/行列式/径向工作量，再逐物理通道推进。
   五个裸参考本身不含完整核价关联，绝不标为关联空间收敛。
3. 用已优化共同轨道做 RCI：完整静态或明确能量转移合同的动态 Breit，
   QED 与反冲按实际公共消费层记账，重新求同一 Hamiltonian 的上下态。
   当前公共 MCDHF 轨道目标不含 Breit/QED；部分 QED/反冲仍是一阶期望。
4. 在同态波函数上验证 E1/E2/M1、寿命、长度/速度约定和径向响应，逐条绑定
   NIST/实验记录、参考分类及不确定度。未拟合结果与 Eu 历史标定分列。
5. 优先去掉精确冗余收缩，复用共享角结构/径向作用/接受的 CI；随后在相同
   模型、驻值门和负载准入下测端到端及并行效率。
6. 小空间 FCI、同积分 FAC/GRASP 等独立锚检验方程、算符与数值。参考程序
   不替代 Atomx 原生求解，也不提供没有实测的统一精度排名。

[原前瞻物理判别合同](physical_frontier004/decision_and_prospective_experiment.json)声明的
完整5p双核空穴与平衡s/d/p核场虚层已在同一冻结第160步点分别实算。
前置的五个完整目标J/parity块精确等价性及两分支完整signed空间均已核验，
没有按配置或振幅筛选；范围仍不认证全谱、全局基态、扩展驻值或完整物理。
双空穴总体使实验gap增加，首层核场虚轨道仅有几cm⁻¹收益；后继先做物理
质量与差能量级判别，不重复低信息量的截断扫描。

## 本轮实现与验证

屏蔽 no-pair 构造重复核项已由独立矩阵锚定位，修复为完整 seed Fock 矩阵的
占据加权平均。核投影主模型与屏蔽投影属于不同明确模型；投影不会在优化
过程中自动更新。数值修复的实验谱影响须另算，不从核矩阵元素大小推断。

声明的 `max_determinants` 现在传入 B-spline、shooting、共同电荷目标的
Hamiltonian 与 RDM/梯度，超过上限显式拒绝，不缩掉物理空间。原始公共
初始化 CI 已传入此参数，此前相关优化路径丢失了它。

性能增量仅复用同一 Coulomb source pair 对同一轨道的矩阵作用，保留 signed
RDM 累加顺序与所有空间、算符、数值阈值。核阶段计时与生产端到端加速分开。

在约 −14067 Ha 的 Yb 总能量上，直接相减会淹没后期轨道下降。新增同一径向
算符的完整二次／四次能量增量，仅对 one-CSF 各对称块且目标密度未变的情形
使用；混合 CI、态跟踪与跨电荷目标显式返回不支持。独立长精度全能量锚验证
公式，未把梯度积分近似作为接受规则。近单位 Gram 的 polar trial 后继则避免
退化本征矢重组单位矩阵，仍计算同一正交化映射；seed 与原始残差度量不变。
合同见[稳定 polar 判别](stable_polar/contract.json)。

真实 endpoint 判别确认正交约束与两次径向重建的舍入噪声。另一个受约束增量
API 使用原 projector 的显式坐标增量，扣除含完整二次项的实际 S-Gram 缺陷，
采用 CI `h/n` 的真实 normal 导数；在物理正交流形上与原能量严格相同。
两端大于浮点尺度的约束改变、混合 CI 响应或不同密度均不适用，原完整差分
API 保留。独立 Yb 重放见[线性增量判别](native_run/linear_map_increment_endpoint/result.json)，
下一完整求解的[前瞻约束](centered_increment_contract.json)保留原初始条件与门限。

[同 003 轨道修正账](correction_budget/README.md)实际计算了完整静态／明确实频
动态 Breit、可用 VP/SE 模型及反冲。动态与静态之差约 0.37–4.23 cm⁻¹，
两个 SE 模型差约 361–450 cm⁻¹；模型差不是严格误差界，不能按更接近实验来
选择模型。完整非局域 SE、屏蔽多体 QED 覆盖不足保留，不能拼成合格预测。

实际 Yb 运行、源快照和诊断由 [native_run](native_run/) 保存，物理独立锚由
[physics_audit](physics_audit/) 保存，性能测量由 [performance](performance/) 保存。
平衡单核空穴有限家族的29共同轨道/1046目标扇区已实际优化到累计208步，仍未满足原MCDHF驻值门。
当前按同源、同空间、原度量从末态有界续算，先辨别未驻值与物理遗漏；后续核双激发、高阶关联与
虚轨道方向须按同上下态差能另立合同，不从EOL下降推断实验gap必然缩小。
任何新增模型均保留独立的合同、数值门与实验门。
