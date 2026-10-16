# 屏蔽f闭合后继：先测31轨道相关Euler

本页是[FD+核场f原路线](../README.md)之后的独立后继，只读新实际JSON，不改旧scope。
[屏蔽f唯一前台receipt](../../screened_f_next_step208/execution001/root_execution.json)已exit0，
[941条native控制](../../screened_f_next_step208/execution001/actual/result.json)全部通过：
32个独立有符号标量矩阵项最大误差8.5265e−14Ha、完整两套118谱及完整1046→1595CI。
实际2H/2CI、7hS producer、1680公共scalarSlater；0梯度/优化器。
59.614秒为未受控墙钟，不能用于加速认证。

原29波、literal208与七个原核P不变，closed68 Fock只构造两个新f径向seed，未改变no-pair边界。
全部完整双jj/config族与匹配规则在见到NIST结果之前声明；没有按较小gap挑seed。
[保存径向支撑](../../screened_f_next_step208/execution001/actual/radial_support.json)中新f平均半径
9.011/9.019a0，对照核场f0.297/0.299a0；这说明构造产生不同支撑，不证明屏蔽seed已径向完备。

| 同原208下态的跃迁变化 / cm⁻¹ | 核场f | 闭核屏蔽f | 屏蔽−核场 |
|---|---:|---:|---:|
| 5d_3/2 | -6.396354 | -207.298240 | -200.901886 |
| 5d_5/2 | -3.184771 | -202.408498 | -199.223726 |
| 6p_1/2 | +8.180120 | +15.937703 | +7.757582 |
| 6p_3/2 | +9.101962 | +19.771323 | +10.669360 |

屏蔽f四线RMS2054.316→1915.983cm⁻¹，原实验1%门仍FAIL；旧核场分支2048.834原样保留。
屏蔽f下态变化−0.00030086171318544075Ha，D3/D5上态变化约−0.00124538/−0.00122310Ha，
P1/P3约−0.000228244/−0.000210777Ha，表中都是各模型内upper−lower差。
D/P重心变化−204.364/+18.493cm⁻¹，分裂+4.890/+3.834cm⁻¹。
没有把总EOL下降或两个seed的能量差当可相加修正。

**新实际反证了“核場f小响应即可排除f径向重要性”。**
在同一有限1595配置族和原P内，构造不同径向候选可使D响应从数cm⁻¹变为约200cm⁻¹；
但仍不能识别哪个缺失关联类主导千cm⁻¹gap，P仍有约−1000cm⁻¹差，D仍高约+2700/+2200cm⁻¹。
两seed未做31轨道自洽MCDHF；原29/208驻值FAIL保持，当前结果不是更广空间驻值。

## 完整父空间与未计算部分的口径

[结构化覆盖账](current_model_coverage.json)更正初版decision的字段口径：
29模型完整枚举父空间2163，已计算五个完整目标扇区1046，其他未算1117；
31模型对应3789、1595、2194。初版`uncomputed_parent_CSFs=2163`实际指完整parent，
字段名不精确，保留其历史字节并在这里更正；并非2163或3789全部未算。
不改变已算target范围，也不认证父空间全谱或全局基态。

## 下一最小测量

最高信息项是**31径向、完整1595目标扇区的native correlated Euler全signed梯度**。
publicCSFOrbitalFunctional的梯度依赖完整CI径向密度；closed68 J/K不是该相关Euler算符。
辅助场最低根满足的是造seed条件，不能用它的Fock残差认证MCDHF驻值。

先建立明确的31轨道point/layout与3689坐标新metric/objective identity，保留原七P、UID与五权0.2。
3451原metric只作来源；多两列后不能伪称旧metricresume。rawscreened波与publicP@x若不逐位一致，
必须测量并声明新canonicalpoint，不能转贴旧owner或隐藏额外baselineH/CI。

最小首控制候选为fresh1H/1CI及完整31生产gradient、全targetnorm/unfilteredsignedRDM的独立公共收缩对照，
含rawh/ketnorm归一化修正、实际fullSGram与原P切向。若全部同点资格过，
再对一个预声明算法方向做±h（候选固定h1e−5），完整CI响应和有限Gram二次项保留，
总量候选3H/3CI/2梯度、0optimizer；如额外canonicalpoint需重验，另明确加H/CI，不能伪装在此预算内。
原abs1e−8/rel1e−5方向门不变，失败保存停止，不扫h、不调NIST权。

这是待精确准入的方案，[本页决策](decision.json)不是执行合同。
已观测1595固定CI峰值1442.078MiB（约1.408GiB）不能保证完整31RDM/gradient成本；须先核公共CSF/determinant/overlap、
全signedRDM与workspace实际规模，再声明CPU1/thread1/AS<=3.95GiB和有界墙钟。
若相关f/同κ旋转梯度明显，就在此完整空间作受控轨道松弛；若某子块小，也不把它当全空间驻值。

之后才与达到原门的共同/分parityEOL点比较；更深4d单孔或明确第二核孔类保持独立完整分支。
当前4d/双核孔关联、同点Breit/QED/反冲仍缺测，旧异epoch分量绝不拼到本31谱。
六条准备工作流仍并行维护，heavy数值按root精确caps准入；本页零NumPy/NPZ/Atomx/API/新科学。
