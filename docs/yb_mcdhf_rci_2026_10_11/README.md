# Atomx 原生 MCDHF→RCI：Yb⁺ 与精度、效率主线

[论文输入对齐与完整5p关联对照](physical_budget31_after_gradient/paper_reproduction001/paper_alignment004/README.md)
已固定原DC轨道和18虚自旋轨道，打开完整5p₃/₂：22关联核电子、46冻结电子。
1171项旧模型恢复核验、531项新实时核验和12项张量测试通过；全部1292379个SDT方程
及12个目标磁伙伴满足原门限。两条线为25979.234799/26667.252046cm⁻¹，
比18核模型上移815.156156/1094.925635cm⁻¹，实验偏差扩大到13.1460%/9.59434%。
精细分裂增至688.017247cm⁻¹，仍低于实验1371.89cm⁻¹。此条件关联增量不能移植；
f/g虚空间、GTO指数及论文实际核/虚截断仍未对齐，不作论文精度或性能提升声明。

[Complete6p virtual continuation](physical_budget31_after_gradient/paper_reproduction001/native_virtual_p003/README.md)
原生RCC对照已继续加入完整6p₁/₂、6p₃/₂多重态：18个关联核电子、18个虚自旋轨道。
全部689589个SDT方程和12个目标磁伙伴通过；两条线为25164.078643/25572.326411cm⁻¹，
比此前完整4f模型上移581.387755/716.411634cm⁻¹，实验差距继续增大。
新增冻结4f对照补齐四模型面板：4f关联的谱线增量随虚空间改变，相差441.120927/489.197981cm⁻¹，
不能当成固定修正相加。局部DIIS迭代经有限FCI和旧Yb解核验，大模型14次完整残差调用收敛；
生产源码和接口未变，12项当前张量测试通过。缺失f通道仅有角动量准入证据，尚未计算其能量修正。

用户于 2026-10-11 明确选择原生 MCDHF→RCI 为原子结构开发主线，Yb⁺ 示例
实际采用 MCDHF。其他软件用于参考方程及独立物理、精度和效率对照。此前
[原生路线](../atomic_native_roadmap_2026_10_11/README.md)中的 Ne、Eu 成功基线、
Yb SD 失败谱与完整 signed 方程证据继续保留；SD/CC 不再是 Yb 示例的主线。

用户于2026-10-14追加授权RCCSDT对照。[原生小空间试算](physical_budget31_after_gradient/paper_reproduction001/yb_native_rccsdt001/README.md)
已完成同算符CCSD/CCSDT、两种附着秩和完整CI比较，连通T₃非零。
CCSDT/3p2h的6s→5d₃/₂谱线为24387.437987cm⁻¹，与同空间CI差−0.001114cm⁻¹。
此模型冻结64电子（包括4f¹⁴）、仅关联四核电子，未包括5d₅/₂；实验差距反而增大。
它是有限EOM-EA方程／集成锚，尚非完整论文FS-RCCSDT复现，MCDHF→RCI主线继续。

[原生张量CCSDT续算](physical_budget31_after_gradient/paper_reproduction001/native_tensor_ccsdt001/README.md)
已避开生产求解中的完整粒子扇区／生成元银行，加入完整5d₅/₂多重态并计算两条D线。
CCSDT/R₃为24372.156946/24708.853564cm⁻¹，与同模型CI差−0.001356/+0.004524cm⁻¹；
全部2/4/6目标磁伙伴、196项实时及353项保存数据核验通过，54项测试通过／2项外部程序测试跳过。
仍冻结64电子（含4f¹⁴），物理full-H缺陷与实验差距保留；此独立对照不放行MCDHF优化。

[完整4f关联与性能续篇](physical_budget31_after_gradient/paper_reproduction001/native_tensor_4f002/README.md)
现已关联18核电子、冻结50电子，保留完整4f¹⁴和原有12个S/D虚自旋轨道。
全部189834个SDT方程与12个目标磁伙伴通过；两条线为24582.690888/24855.914777cm⁻¹，
比冻结4f时上移210.533942/147.061213cm⁻¹，未缩小实验差距。
精确最高权重投影使三个目标态用28次矩阵作用收敛；此前两次求根失败完整保留。
同输入ABBA计时的完整核残差8.87→5.10秒（1.74倍），输出逐位相同；
59项相关测试通过／2项外部程序测试跳过，514项实时／96项保存数据核验通过。
这是当前RCC对照的有限虚空间结果，不提供大模型full-H、实验收敛或MCDHF放行。

用户后续已取消卡片：小时timer已停止并禁用，自动卡片授权已撤销。
物理研发继续；“推进”不恢复卡片。当前
[Eu/Yb比较核查](eu_yb_comparison001/README.md)区分Eu³⁺参数标定与Yb⁺未拟合、未驻值的谱。
按[论文方法开发](physical_budget31_after_gradient/paper_reproduction001/RESULTS.md)，
原生有限 CCSDT、电子附着能与 g/h/i 通道已完成独立验证（39项测试、127项高-l控制）。
这不是69电子Yb⁺论文数值复现；当前MCDHF/RCI谱和原失败门保持原记录。
[最新论文DHF基线对照](physical_budget31_after_gradient/paper_reproduction001/yb_dhf_table001/README.md)
已在两套原生基底实算三项附着能、两条D线及两项静态四极矩；D线与论文DHF行分别差
−0.259/+0.323cm⁻¹，四极矩符合表中小数位。原始物理矩阵回读、完整虚电子列和独立径向／角因子
核验通过；这只核验独立DC起点，完整关联谱、论文RCCSDT及生产MCDHF精度仍未完成。
[AI辅助物理迭代](ai_assisted_iteration001/README.md)现已建立完整配对证据账与建议器：
八个整体记录、六个合格历史配对，仅三个点组上界。当前同条件样本不足，建议器实际拒绝
学习预测，按物理优先级排序；失败不自动重试，NIST不进入训练，不恢复卡片。

[本轮合同](contract.json)区分轨道目标、CSF 空间、no-pair 投影、最终 RCI 算符
和观测态，保持原数值门与实验 1% 门。此前指南、进度和实际源文件的字节快照
见[前版清单](previous_sources/manifest.json)。

[最新文献指导的解析响应](physical_budget31_after_gradient/physical_next_decision012/README.md)
原生实现受约束极分解路径、DC径向一／二阶导数及完整有限CI的2n+1响应。
两套原模型10个目标态预测全部16个有限CI响应量，最大差9.144e−13Ha，155项核验通过。
新增15项独立测试通过；完整回归67通过／2失败，改动前同样失败且39个H／本征数组逐位相同。
原普通差分失败保留；此处只核验CI响应项，未算固定向量H3或总余项，未放行轨道优化或证明实验gap改善。

[此前差分诊断](physical_budget31_after_gradient/physical_next_decision011/README.md)
用精确二进制有理数区分根值／Rayleigh商差、有限CI系数响应及固定向量算符响应。
两项大步长D3/2偏差含约1.07e−8 Ha的CI响应项，步长减半后约缩小四倍。
现有原生中心化求解器在原十个保存H上重放后，普通差分为38/40，
B小步长误差降至4.50322e−9 Ha；两项大步长仍失败。旧005原失败保持，
未改生产源码、未重算当前源物理H／梯度、未放行优化，下一判别是独立的受约束CI响应曲率。

[最新密度／梯度收缩](physical_budget31_after_gradient/physical_next_decision010/README.md)
新增显式流式DC密度与批量有符号双线性收缩，默认保持缓存路径。
113项相关测试、357项原模型／完整4f¹²密度梯度实时核验和102项保存结果核验通过。
同方向D3/2差能响应降至原模型约18–25%；完整五态EOL原度量预条件范数
仍为A/B的0.189697/0.244299，远高于1e−7。原三项普通差分门与驻值失败保持。

[此前数值 H 构建](physical_budget31_after_gradient/physical_next_decision009/README.md)
显式选择流式装配，直接形成数值 CSR，避免保留大型角向系数表；默认路径保持。
262 项相关测试、完整当前 A/B 的523项实时及222项独立保存结果核验通过。
完整4f¹²新增82个NR组态在两臂全部保留，五扇区均已算完；265项实时、554项保存结果核验通过。
B臂D3/D5/P1/P3跃迁分别上移272.969/264.847/186.217/131.426 cm⁻¹，
虽各态总能下降约七千cm⁻¹，大部分关联贡献在跃迁中抵消；详见当前报告。
[前轮角向构建](physical_budget31_after_gradient/physical_next_decision008/README.md)
及[先前稀疏 H／密度核验](physical_budget31_after_gradient/physical_next_decision006/README.md)
仍是独立源版本的证据。固定轨道的组态贡献与完整MCDHF弛豫分开；
原驻值、三项导数门及实验精度失败保持，accepted13结果不移植到terminal16。

## 当前物理结果：31轨道原生弛豫完成16步；驻值仍失败

原29轨道 MCDHF 的同一208点，现已完成[一切向、两尺度约束导数核验](stationarity_next_step208/instrumentation_repair001/CLOSURE.md)：
7053项实时控制通过，两个导数误差约6.0e−11及5.3e−11，低于原门3.06e−8。
这支持该方向的相关梯度；原驻值仍失败，不能推广为完整Hessian或误差界。
首轮观测钩子失败继续按原记录保留。

[原生屏蔽f固定 RCI](screened_f_next_step208/execution001/RESULTS.md)完成941项实时控制：
先32项独立有符号PP/QQ/PQ/QP矩阵元，再两完整118维辅助场谱及1046→1595嵌套完整目标CI。
使用31径向轨道，新增f平均半径约9.01a0；之前核场种子约0.30a0。
D3/D5/P1/P3的跃迁位移为 **−207.30/−202.41/+15.94/+19.77 cm⁻¹**，
四线RMS从2054.32降至 **1915.98 cm⁻¹**，原实验门仍失败。
这说明径向种子会明显影响有限f空间响应；不能根据核场种子的小增量排除f关联。
闭核68电子J/K只构造辅助种子，没有加入最终多电子DC Hamiltonian，亦不是相关Euler场。
该固定RCI记录尚未优化31轨道；后续16步弛豫另列如下，驻值仍失败。
完整父空间为3789 CSF，实际计算五个完整目标扇区1595，
其余2194未求解；原29父空间2163中计算1046，其余1117未求解。

[完整31轨道相关梯度](screened_f_next_step208/gradient_proposal001/execution_preparation001/CLOSURE.md)
已完成438项实时控制及44项保存元数据闭合：实际2H/2CI/3完整signed梯度，
canonical点与公开坐标恢复点分别保存，恢复的最大径向波差约2.34e−14，
最大跃迁差约1.60e−6 cm⁻¹。新3689坐标保留原29块，新f复制同κ原4f逆度量仅作诊断；
新残差 **0.005162180923375838 > 1e−7**、g·pg/2指示量
**9.089562518880953e−5 Ha > 1e−8**，两门均失败。
该梯度的原记录是HF偏导，不能与原3451残差比较改善。
完整归一化与raw-h商法则修正对约束梯度的改变很小；完整4f↔5f旋转的偏导明显非零。
[31轨道约束导数后继](screened_f_next_step208/constrained_fd_proposal001/representation_repair001/actual/result.json)
已完成730项实时控制：基准点与两个步长的正负扰动共5次完整H/CI，五态UID及0.2等权保持。
普通报告能量的导数误差为+1.38e−7及−2.26e−7，均低于原门3.71e−7；
独立中心化有限CI响应的误差约6.00e−11及1.90e−11，分别保存，没有替换生产目标。
这仅核验所声明4f↔5f联合旋转方向，不是全空间导数、驻点或实验精度认证。
首轮驱动误读径向梯度形状的[失败](screened_f_next_step208/constrained_fd_proposal001/execution_preparation001/actual/failure.json)
保留；后继只修正连续31×246向量的读取，原物理模型、门限及失败记录均不变。
[31轨道原生MCDHF有限弛豫](screened_f_next_step208/native31_f043_relaxation_preparation001/actual/result.json)
已在原f043数值源前台闭合exit0：16接受步、17点、1582项实时控制通过，
实际23H/23CI/51完整signed梯度，原五态UID及0.2等权、原3689坐标度量保持。
全部16次原Armijo比较通过完整stored-H精确商账复核；未使用能量回调或实验拟合。
最后点四线RMS为 **1283.7699121117464 cm⁻¹**，初始为1915.9833377599443；
D3/D5/P1/P3 gap为 **−1496.50/−1480.16/−1336.54/−612.81 cm⁻¹**，原实验1%门仍失败。
原驻值残差 **0.68254770188387 > 1e−7**、指示量
**0.6018594158445788 Ha > 1e−8**，均失败，终止原因为maximum_iterations。
总EOL下降0.07226676172285806Ha；12→13步切向梯度和残差突增，
不能从能量下降或某个中间点更小gap认证优化收敛。完整轨迹保留，未按gap选末态，
[77项保存标量闭合](screened_f_next_step208/native31_f043_relaxation_preparation001/closure001/completion.json)
零新科学计算。下一步先判别此局部响应及同点RCI修正量级，不自动增加优化步数。
若迁移到当前共享源，需要重新核验同点及两尺度导数。
当前共享CI采用标量中心化对角化，与保存f043源的E/C末位可能不同，不能自动继承旧源资格。

[第12→13步完整源码重放](screened_f_next_step208/native31_f043_relaxation_preparation001/growth_step12_source_diagnosis001/stageA_execution_preparation001/RESULTS.md)
已通过200项控制、精确重现第13点：L-BFGS方向范数为预条件最速下降方向的64.29倍，
新增5f占方向平方范数73.48%。这排除历史恢复错误，未证明物理梯度错误或过冲。
[第13点实际方向导数](screened_f_next_step208/native31_f043_relaxation_preparation001/growth_step12_source_diagnosis001/stageB_execution_preparation001/RESULTS.md)
已闭合exit1：743项通过，两个普通报告能量导数门失败；5H/5CI/3完整梯度及四扰动账全部保留。
包含完整有限CI响应的独立中心化导数在两个步长误差约+1.48e−10/−1.94e−10，
低于原3.83e−7门；普通报告能量误差−1.41e−6/−1.77e−6仍超门。
这支持该局部方向的物理导数，未认证全空间、驻值或原alpha1路径曲率，不自动增加优化步数。
[保存标量分解](screened_f_next_step208/native31_f043_relaxation_preparation001/growth_step12_source_diagnosis001/stageB_execution_preparation001/reporting_difference_diagnosis001/README.md)
表明普通与中心化差异主要来自径向增量与完整stored-H商能增量的闭合斜率：
−1.26e−6/−2.05e−6Ha每单位方向；不是仅靠EO末位分辨率就能解释。
[两个保存点的完整矩阵重放](ai_assisted_iteration001/gradient_producer_source_audit001/saved_contraction_preparation001/RESULTS.md)
已闭合96项控制：完整H径向增量加有限CI响应与直接商能增量相差约1.53e−20Ha，
径向多项式与矩阵径向项仍相差+2.8954e−11Ha。此点对的差异已定位在CI响应与商能导出之前，
[完整端点产生路径对比](ai_assisted_iteration001/gradient_producer_source_audit001/endpoint_radial_contraction_preparation002/RESULTS.md)
已闭合191项控制及全部19589键：普通径向密度收缩与stored-H相差2.6673e−11Ha，
占总差约92.12%；多项式与普通端点相差2.2809e−12Ha，占7.88%。
原生forward角表signed-pair增量与stored-H各态相差不超过2.38e−13Ha，
保存基线密度及原多项式标量精确重现。这缩小到收缩/构造算术路径，未认证源码修正或严格误差界。
首轮准备误用展示态序的失败保留；后继只修正原账本累计顺序。
总差约6.35e−6cm⁻¹，不能解释上千cm⁻¹的实验gap；原两导数失败、驻值失败和生产目标保持。
[完整前向增量判别](ai_assisted_iteration001/gradient_producer_source_audit001/forward_linear_increment_preparation001/RESULTS.md)
进一步闭合98项控制及399个源/输入守卫：完整T@ΔR与转置径向收缩各态一致至8.24e−20Ha，
但与分别组装端点后相减的加权差仍为2.6692e−11Ha。这在同一1595态点对定位了
端点装配的有限精度非可加性，未证明某个具体舍入操作或认证生产修正。
全部19589键的五态带符号密度系数现已归档；下一数值工作可直接针对增量装配与导数资格验证。
原生产91个NR组态的[占据审计](eu_yb_comparison001/correlation_scope_audit001.json)为7个零核空穴、84个单核空穴；
没有显式双核空穴组态。固定此限制只增加径向截断无法补入该关联。
[旧24轨道5p双空穴判别](rank_only_discriminator/decision.json)已经测得四线位移
+461.25/+596.71/+194.49/+148.67cm⁻¹，但RMS从2022.83增至2225.97；不能移植到当前31点。
[当前完整组态增量准备](eu_yb_comparison001/core_core_discriminator_preparation001/README.md)
保留全部91组态，增加九个5p双空穴组态，目标CSFs1595→2574；
[实际完整构造](eu_yb_comparison001/core_core_discriminator_preparation001/execution_preparation002/RESULTS.md)
已在accepted13点按明示2659上限完成2574完整H，157项通过后，原嵌套1595子块逐位门失败：
最大差8.88e−16Ha。原门未放宽，原实验在求解扩展CI前停止；独立保存矩阵诊断另列，不能升级此失败。
[独立保存矩阵诊断](eu_yb_comparison001/core_core_discriminator_preparation001/saved_matrix_diagnostic001/RESULTS.md)
已完成五个完整谱及原signed追踪，56项通过后最后模块导入边界检查失败；全部输出保存。
accepted13点D3/D5/P1/P3位移为+468.06/+673.70/+149.14/+160.82cm⁻¹，
四线RMS910.72→1113.17cm⁻¹恶化。嵌套旧子块算术差对固定排序跃迁的严格界小于4.973e−10cm⁻¹，
不含求解器、物理截断和假想算符下的态重识别误差。两原失败均保持；这不是末态16或已驻值谱。
[已完成的四点物理判别](physical_budget31_after_gradient/physical_next_decision004/execution002/RESULTS.md)
以原accepted13和预先固定α=1/16的同一完整径向位移，完成A1595/B2574四个owned H/CI，
260项控制全部通过。四线关联×弛豫交互项为+1.445904/+0.768975/+1.432059/+2.711302cm⁻¹；
这不是其他方向或terminal16的交互上界。
[后续完整五态导数实验](physical_budget31_after_gradient/physical_next_decision005/STATUS.md)
保留674/677通过及原三项D3/2普通线导数失败。新固定轨道4f¹²结果也不替代导数与驻值资格。
[扩展梯度源码准备度](physical_budget31_after_gradient/physical_next_decision003/expanded_gradient_readiness001/README.md)
已明确先比较原筛选EOL梯度、完整实际范数梯度及其差值，保留31轨道与全部带符号项。
扩展角表1530万非零项。[实际完整梯度比较](physical_budget31_after_gradient/physical_next_decision003/expanded_gradient_execution_preparation002/RESULTS.md)
已在同点完成三次原生31轨道梯度，1355项控制及400个源/输入守卫通过：
完整实际范数梯度的驻值残差为0.221316，原空间为0.153978；两者均未驻值。
切向梯度方向仅差1.965°，同一原逆度量预条件后方向差46.451°；
原下降方向在扩展空间仍下降，保留98.506%的总EOL一阶下降量。
原微小密度系数筛选与实际范数处理的合并残差影响约1.19e−14，不能解释此点的主要驻值残差。
这份条件梯度记录本身不含新谱线；后续有限位移及五态导数结果已分别列于004/005。
不能从总EOL梯度推断四条谱线的各自导数，也不能将无owned快照的保存态直接交给优化器。
首轮[记录层失败](physical_budget31_after_gradient/physical_next_decision003/expanded_gradient_execution_preparation001/RESULTS.md)保留，后继只修正UID容器及标量JSON导出。
[文献精度及物理贡献预算](physical_budget31_after_gradient/literature_precision_budget001/README.md)
核对了2025年Yb⁺ RCCSDT结果：两条5d激发能中心值偏差为+69/+3cm⁻¹，
该方案中的三激发、h/i补充、Breit激发能贡献分别约−456/−430、−561/−512、−70/−96cm⁻¹。
当前31轨道只含s/p/d/f，尚无g/h/i；100个NR组态也没有完整核关联及系统高阶收敛资格。
后续资源选择应同时对齐文献的关联通道及谱线贡献，不能仅围绕驻值残差决定物理路线。
这些是文献模型的量级，不能直接移植修正；两条D线的中心值吻合不认证6p或全谱精度。
更进一步的[分区关联原生源码评估](physical_budget31_after_gradient/partitioned_correlation_assessment001/README.md)
明确已有分J/宇称DF→RCI、非正交重叠及一体跃迁处理；完整PCFI仍缺跨分区两体H、
联合度量广义CI与多轨道库态契约，应先用两电子完整空间资格验证，尚未实现或迁移Yb。
末态16的[保存观测量分解](eu_yb_comparison001/terminal_observable_decomposition001.json)给出
5d重心gap−1486.70、分裂gap+16.34cm⁻¹，6p分别−854.06/+723.73cm⁻¹；
组态间平衡与精细分裂需分别判别，不能只看总RMS。

[RCI横向矩阵所有权扩展](rci_retained_transverse_implementation001/README.md)
已在Atomx原生实现中加入静态Breit及频率相关横向相互作用的完整矩阵复用。
频率相关项替换静态Breit，按完整轨道能量映射及精确binary64输入绑定；
DC schema1与原MCDHF算符边界保持。[63项实际测试](rci_retained_transverse_implementation001/primary001/root_execution.json)
全部通过；[26项受影响集成测试](rci_retained_transverse_implementation001/integration001/root_execution.json)
也已通过，覆盖原signed梯度、径向缓存、频率provenance与独立完整多电子矩阵。
更完整的[有限CI响应集成](rci_retained_transverse_implementation001/mixed001/root_execution.json)
实际为42通过、2失败：与历史source002的整个H逐位相等门失败；一个h=1e−4方向的
导数误差约1.68e−7，超过原门8.72e−8。13点独立Decimal有限响应控制通过，
原完整调用账与失败均保留。[保存数组诊断](rci_retained_transverse_implementation001/mixed_failure_diagnosis001/actual/result.json)
消费全部13点和两方向原三尺度，未新建H/CI/梯度：五个历史比较H均非逐位相等，
最大绝对差2.22e−16；完整signed梯度22个分量不同，最大差1.08e−19。
全部13个完整谱残差不超过4.09e−15，Gram误差不超过2.43e−15。
两个方向在h=1e−4都超过原导数门，在h=1e−6与1e−8均通过。
这支持有限步长截断的解释，但不构成统一误差界；rank2内积分阶数9→10是源码候选。
[同源基准复验](rci_retained_transverse_implementation001/matched_preowner_baseline_replay_preparation001/actual/result.json)
现已前台闭合：两份完整365文件源仅ci及owner不同，径向核相同；各自真实构造一个DC owner，
完整29×29 H及7×70 signed梯度均与彼此及保存当前基准逐位相等，schema1头也相等。
完整29根残差最大2.64e−15、Gram误差2.00e−15；实际2H/2g/0CI求解。
这隔离了归属改动在此基准点的作用，不能认证历史源比较、全路径导数或大步长门。
两个原失败仍然失败，
不能把当前共享实现写成全部通过，或据此直接执行31空间优化。
[Yb31同点横向对照](transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/scalar_request_observation_successor001/actual/failure.json)
已闭合失败：65通过后，原DC整个H逐位相等门失败；静态及零频Breit均未进入。
[保存数组判别](transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/scalar_request_observation_successor001/saved_DC_failure_diagnosis001/RESULTS.md)
确认相同整个owner及31轨道P/Q，最大H元素差7.28e−12Ha，全部1595根残差及完整Gram通过。
该有限矩阵差的激发谱保守上界约0.0051cm⁻¹，不能解释当前千cm⁻¹缺口；原逐位门仍失败。
相应源资格及横向修正量尚待独立闭合。
[同源码DC基准后继](transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/scalar_request_observation_successor001/same_epoch_DC_baseline_successor001/RESULTS.md)
已前台闭合exit1：完整DC H逐位复现、86项控制通过；静态构造在261889次标量返回后触及
预设1200秒上限，静态H未完成、独立零频未进入，没有新增可用Breit谱位移。
原失败及所有已完成DC数组保持；[源码复用调查](transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/source_batch_reuse_note001/README.md)
指出静态Gaunt成对波/势重复求值候选。现已完成[私有布局保持复用资格](transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/source_batch_reuse_note001/layout_preserving_successor001/RESULTS.md)：
26项控制通过，全部16384完整积分键逐位相等；初始波求值65536→8、Yk求值32768→256。
原连续复制候选的逐位失败保留；这次只改变私有快照布局，未采用到共享源码、未重算完整31 H，
工作量减少不是整段Yb实测加速比，当前没有新增Breit谱位移。
此项不是频率相关MCDHF梯度或完整QED认证。

[物理贡献与六方向决策](physical_budget31_after_gradient/README.md)区分驻值、核关联、
径向/角空间、Breit/反冲、QED完备性及性能。目前千cm⁻¹ gap的主导物理来源尚未确定。
[CPU／CUDA私有批处理资格](gpu_mcdhf_next_step208/qualification_representation_repair001/actual/result.json)
已完成16760项实时控制：同一f043完整31轨道点、19589项signed表下，
CPU1、CPU4及CUDA的全部2824带矩阵逐位一致，完整径向及3689坐标梯度控制通过。
CUDA实际使用FP64顺序scan/row核及6个完整源分块，D2H为25009344字节；
整个资格墙钟50.60秒、采样进程组RSS峰值约1.76GiB，均为未受控观测。
CPU1首梯度包含JIT而CPU4复用编译核，不能由该数据计算公平并行效率或加速比。
GPU只计算带矩阵，H/CI、signed收缩及约束度量仍在CPU；未移植至当前共享生产源码，
没有GPU全H/CI、全空间有限响应、31轨道驻值或效率认证。
首轮[196通过/1失败](gpu_mcdhf_next_step208/qualification_preparation001/actual/result.json)
保留：tuple与保存JSON list直接比较失败；后继仅规范化完整头字段并提前保存实际头，
没有改变物理计算、数值门或资源上限。

[stage029实证整合](reporting_stage029/README.md)95项元数据/规划器/渲染检查通过；
[stage030历史卡片](reporting_stage030/README.md)更正过时谱表标签，保留全部原始科学字段及失败。
原能量指示量g·pg/2不构成能级误差界。
[02:00实际小时卡](delivery_hour26/delivery_authorization_proof.json)内容为stage028；
[03:00实际小时卡](delivery_hour27/delivery_authorization_proof.json)内容为stage030，
均已发送并读回资格闭合，各自严格格式门的失败仍保留。
02:00/03:00当时的自动服务为implementation_frontier、0科学作业；
[04:00实际小时卡](delivery_hour28/delivery_authorization_proof.json)使用stage032，
已发送并读回；原八操作格式门失败保留，保存JSON另按既有九操作协议闭合。
04:00服务完成四项既有FSCC、性质、SD图及GPU数值探针，Yb MCDHF自动作业为0；
root原生Yb迭代继续按独立合同准入，不能将旧小时的零作业推广到当前服务。
[stage031元数据整合失败](reporting_stage031/root_execution.json)仅因新增panel引用采用SHA字符串
而非公开渲染器要求的path/sha256对象，发生于预览时；registry前像仍保持stage030。
原失败和未发布输出保留，修复使用新后继，不能把stage031写成已送达或已整合。
[stage032后继整合](reporting_stage032/integration_proof.json)已前台闭合exit0，
实际预览13元素、87引用，公开规划器指向新汇总；全部stage030科学字段与原失败保留。
它作为04:00卡片的实际输入，不改变03:00卡片的历史内容。
[stage033历史汇总](reporting_stage033/integration_proof.json)已前台闭合exit0，
绑定实际31方向导数730门、三后端同点一致性16760门、共享回归42通过/2失败及
04:00四项自动探针；原驻值、实验失败和谱值保持。它更新下一小时输入，
没有新增科学、发卡或定时器，也不代表05:00卡片已经送达。
[stage034后继整合](reporting_stage034/integration_proof.json)已闭合，绑定31轨道16步实际末态；
它更新保存汇总，没有新增科学或发送。用户取消后不再供自动卡片调用。

原29轨道已闭合末态是 [run005 local128 / global208](correlated29_optimization_run005_preparation/RESULTS.md)，
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

[stage025 当前引用修复](reporting_stage025/README.md)将实际自动规划器的当前输入与
[stage024 同208/双240分支主表](reporting_stage024/current_card_evidence.json)统一，
[632项来源与公开规划入口控制](reporting_stage025/integration_proof.json)通过，零新增科学或发送。
[stage026汇总](reporting_stage026/README.md)及其[696项元数据检查](reporting_stage026/integration_proof.json)
保留同208点、两条240分支、CPU负载失败和版本8发布；该报告冻结时独立5f资格尚未闭合。
随后已完成的[389项独立保存数组资格](complete5f_fixed208_import_repair_preparation/completion_counter_audit/saved_array_qualification_preparation/output_serialization_repair/RESULTS.md)
作为另一个闭合证据记录，不回写stage026的历史状态。
[stage028历史报告](reporting_stage028/README.md)已绑定389项资格和实际第208步首次导数尝试失败，
[1973项来源、公开规划器与渲染检查](reporting_stage028/integration_proof.json)通过。
面板由stage026加六项主要证据组成，预览13个元素、76条引用；完整来源清单保留在结果manifest。
[stage027](reporting_stage027/README.md)的1966项元数据通过及726引用展开作为历史快照保留，
不再作为当前卡片面板。报告整合零新增科学或API，尚不代表实际小时卡送达。
[第01:00小时卡](delivery_hour25/delivery_authorization_proof.json)实际仍是stage024内容；
发送成功，但该轮自动服务因旧输入不一致而拒绝，状态为waiting_for_evidence_repair、0作业。
引用修复不回写该拒绝，也没有自动重跑科学；root原生迭代另按合同准入。
[stage022](results_stage022.json)及其[280项整合](reporting_stage022/integration_proof.json)作为第208步历史汇总保留。

远程周期表[版本9已私有发布](atlas_update008/PUBLICATION_CLOSURE.md)，当前23模型92行。
新增两组固定31/1595 RCI，均未进行31轨道优化；原驻值和实验失败保持。
61项静态及54项实际桌面/移动浏览器控制通过，发布提交676adfc663e6be447e5097722d4e10113c3e1731，
版本9部署成功，访问范围保持owner-only；全6019份NIST gzip与原21模型84行保留。
此前[版本8](atlas_update007/publication_manifest.json)的
两条history10/40末态与各33点完整轨迹分别保留，没有按gap选择分支；
277条轨迹记录对应262个不同保存点身份，共同208路线仍是209个不同点。
[57项实际桌面/移动浏览器检查](atlas_update007/browser_result.json)通过，
[部署成功](atlas_update007/deployment_result.json)，访问范围保持owner-only。
原19模型76行、全部6019 NIST gzip、hosting及renderer字节保持，
[版本7](atlas_update006/root_publication.json)与[版本6](atlas_update005/root_publication.json)仅为历史发布证据。

stage022卡片成功送达后，root按独立合同执行[private径向复用资格003](correlated29_radial_reuse_qualification003_preparation/actual/result.json)，
1126项完整H/CI根、signed密度/梯度及原度量比较通过，实际2H/2CI/2g、88.298秒未受控总耗时。
这是原corrected29同点的新私有数值源资格；该资格运行本身没有新增轨道优化或性能认证。
随后两处数值复用/对角化改进已[采用到共享实现](correlated29_radial_reuse_qualification003_preparation/shared_implementation/README.md)，
[123项受影响回归](correlated29_radial_reuse_qualification003_preparation/shared_implementation/root_test_execution.json)
全部通过，35.556秒属于回归耗时。当前共享采用与保存run005的immutable f043源分开，
没有改变该末态、原驻值/实验失败或旧资格失败。
[实际暖链比较](correlated29_radial_reuse_qualification003_preparation/warm_benchmark_preparation/RESULTS.md)
完成14H/14CI/14完整梯度；21891条科学控制通过，
[138项保存元数据闭合](correlated29_radial_reuse_qualification003_preparation/warm_benchmark_preparation/completion.json)通过。
原预声明pair3负载门失败（CPU30忙碌率0.97>0.05），全部五对均保留，未剔除或重跑。
中位ordinary/candidate耗时比1.188720782仅是观测值，**没有受控加速认证**，
也未测优化至驻值、并行效率或MCDHF GPU加速。
[原资格002失败](correlated29_radial_reuse_qualification002_preparation/README.md)仍按原epoch保留。
另[保存g/pg分解](correlated29_saved_gpg_decomposition_preparation/actual/result.json)149项控制通过，
只检验第80/208步保存坐标代数：g·pg/2是原能量指标，不是谱线误差界，
不能从残差、分块贡献或整体总能推出哪项物理缺失主导约2000 cm⁻¹ gap。
[第208步物理路线](physical_next_step208/README.md)先分辨共同/分parity EOL驻值，
随后完整5f新角通道，再单独4d单孔或第二核孔类，最后同点修正账。
整数估计5f/4d/5p双孔分别为完整1595/1576/9298目标CSF，零新科学求解，
不保证任何一类补好实验gap。

[stage028历史卡片主表](reporting_stage028/current_card_evidence.json)明确展示上述
同一第208步基线，并另列从该点分出的[history10/40各32步末态](correlated29_history_comparison_preparation/completion.json)。
两支各到自己的global240，残差分别为0.0003557923/0.0006485366，
能量指示量3.68579e−7/3.31256e−7 Ha，两支各自的两驻值门均失败；四线RMS分别为
2051.371/2051.470 cm⁻¹，两支原实验门也均失败。没有按实验gap选唯一末态或新种子。
[stage024汇总](results_stage024.json)只整合已有闭合证据。完整5f固定点判别保持
literal208旧29轨道、原P/核、五态UID及完整1595目标CSF，未换成任一240分支。
[原驱动导入失败](complete5f_fixed208_preparation/actual/failure.json)发生在所有科学调用之前，原失败保持。
仅修导入的[后继准入](complete5f_fixed208_import_repair_preparation/root_review.json)先通过14项真实导入检查，
随后唯一[前台运行已闭合exit 1](complete5f_fixed208_import_repair_preparation/root_execution.json)：
实际2H/2CI，原最后计数门因h/S完成7次而声明14次失败。
此前完整谱、嵌套H/Ritz、径向和UID控制的保存证据保留，**原运行仍失败，不作为通过的物理判别或新MCDHF驻值**。
[缓存调用链审查](complete5f_fixed208_import_repair_preparation/completion_counter_audit/README.md)
确认原两次CI共用同核与工作区的七个h/S算符缓存；没有漏掉新f轨道收缩。
独立保存数组审计先在389项数值门通过后因longdouble列表JSON输出失败，
[该失败](complete5f_fixed208_import_repair_preparation/completion_counter_audit/saved_array_qualification_preparation/actual/failure.json)保留。
仅修两处输出转换的[版本后继](complete5f_fixed208_import_repair_preparation/completion_counter_audit/saved_array_qualification_preparation/output_serialization_repair/completion.json)
已独立闭合exit0：18次唯一NPZ读取、389项全H/根、signed嵌套1046块/Ritz、原29波/投影/度量、
两套完整118维径向谱及五UID与上下态差能核验通过；零新H/CI/梯度或本征求解。
**有限同点DC响应取得独立资格，原14计数门及两个旧执行失败均不升级。**
D3/D5/P1/P3的上下态差能改变量为**−6.396/−3.185/+8.180/+9.102 cm⁻¹**，
四线RMS **2054.316→2048.834 cm⁻¹**；D/P重心改变−4.469/+8.795，分裂改变+3.212/+0.922 cm⁻¹。
这些量对应固定literal208、完整1595目标CSF、31条径向波的单次核场5f候选，
原驻值和实验门仍失败，未执行扩展MCDHF驻值；其余父扇区只枚举，未认证全谱。
新f平均半径约0.30 a0、原4f约0.75 a0，不能据小响应排除完整f关联不足或宣称径向收敛。
下一步先分辨同208约束导数/驻值与原生屏蔽f径向候选，不重复扩大截断，
不把该有限空间响应作为约2000 cm⁻¹ gap的完备性或误差界。
随后[实际同208导数尝试](stationarity_next_step208/actual/result.json)因不可变tracker的
观测钩子写入触发FrozenInstanceError，[唯一前台退出](stationarity_next_step208/root_execution.json)为exit1。
58项先行身份/空间门通过，实际完成7次初始h/S；H/CI/完整梯度均为0，方向导数尚未测量。
这是观测工具失败，不构成物理导数失败或通过；原失败保留，钩子修复后须另行准入，
没有以保存的389项RCI资格代替MCDHF导数/驻值判别。
旧第六阶段531-CSF/320步卡表作为历史保留，不再是当前卡片主谱表。

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
该失败epoch当时没有应用共享补丁；后继共享采用见页首，没有29轨道加速比。
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
