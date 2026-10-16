# 固定第208步：完整5f双jj角通道的有限RCI判别准备

这是未执行的原生 Atomx 判别，不是新MCDHF驻值结果。预声明种子是
[run005 literal208](../correlated29_optimization_run005_preparation/completion.json)，
不使用history10/40分支的最低gap点。原29轨道、七个核no-pair P、核模型、
原531参考tracker及五UID等权0.2保持；原3451逆度量只作身份与历史残差来源。
新增31轨道的RCI空间没有轨道优化、新逆度量或旧metric resume声明。

[合同](contract.json)和[91个完整NR配置](models.json)先固定模型、种子和资源。
[整数估计](../physical_next_step208/integer_space_estimates.json)给出完整父空间
3789 CSF及五完整目标扇区1595 CSF（219/223/367/370/416）。原1046扇区全量
嵌入，移除的其它J/parity只是不进入五目标的严格不可达扇区；无配置或振幅删选。
实际公共耦合树计数、signed磁行和兼容determinant-pair数尚未执行。
各扇区磁行整数上界1872/1917/1653/1694/1324低于原2000门，但原1m兼容对门
仍须先测；1595全H或C各约19.41MiB不是完整RSS界。

两个新标签(5,+3)/(5,−4)各用原核P内与现4f波S正交的118维补空间。
种子是同未屏蔽核h的预声明最低补空间根，不以能量>0定义正电子分支。
复用整份[已资格化全谱数值helper](../screened_core_virtual_discriminator002/refine_spectrum.py)，
先fullgvd初值、再有界longdouble全谱Jacobi；全部118根通过原物理矩阵
Rayleigh/约束残差/Gram门，最低根还须原1e−8 Ha、间隙>1e−10及原根span>
0.5门。只翻全局相位，保存原始与带符号种子，不修复原29波。
[旧minimum失败](../rank_vs_virtual_discriminator/failure.json)保留；此次不使用
其gvx reported值作为精确核最低根，不放宽门或自动换根。

选择核h是为了固定一个可复演候选并避免未声明的屏蔽场producer。
Z70核场候选可能偏局域；原P正确及4f正交不保证它适合作关联径向函数。
此次保存核根能量、原4f正交及九轨道同原B-spline quadrature的均值、中位、
RMS半径和累计密度。屏蔽闭68核的旧s/d/p字段没有新f分量，若改用该构造需要
新f场和独立signed锚的另一个合同。小响应不能排除缺失f角通道或径向质量问题，
大响应也不能证明它补好约2000cm⁻¹实验gap。

驱动只重用公共enumerate_jj_csfs、ci_solve(retain_hamiltonian=True)、
CIBasisTracking及原DC核。两完整CI各保留真实owner的H，无额外archiveH重造：
基线H/全部reported根/全部signed C/真实owner逐字对照原208；新增H保存完整旧
1046 signed principal块并逐扇区检查全部ordered Ritz根。五UID与complete
exact-degenerate span按公共tracker规则匹配，不重置reference。
输出五态绝对变化、同点upper−lower四线变化、重心/分裂及RMS；不把绝对总能量
下降当跃迁修正，不移植旧Breit/QED/反冲账。原208两驻值门与1%实验门仍FAIL。

预算是CPU31/线程1/3.95GiB、内层2040秒/外层2055秒；2H/2CI及其14原生h/S，
2补空间SVD、2核fullgvd和10完整CI block evd。另计公共angular highest-weight
nullspace/SVD≤2000、tracker overlap≤3/principal SVD≤10；全谱helper最多24
完整sweep/165672 pair attempts/26全谱残差评估。零新Dirac/Fock/外部hS/梯度/
optimizer/RDM。旧七P已由run005的实际7次验证Dirac与完整closure绑定，不能把
此次零Dirac写成新的正电子谱验证。原29/1595与源均不能为预算缩小。
所有失败保留数组、preflight/counters与原因，不重试或自动增加cap。

[run.py](run.py)默认及--authorize只用stdlib读JSON/哈希，不导入NumPy/Atomx、
不读NPZ；--execute必须有root新seal及实际送达proof的message/chat/bot字段。
这份准备没有执行脚本、科学producer或数组读入。root需另审并前台执行一次：

```sh
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_preparation/run.py --authorize --binding docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_preparation/execution_binding.json
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_preparation/run.py --execute --binding docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_preparation/execution_binding.json
```

命令仅供root审核后使用，execution_binding尚不存在。独立保存数组复核若执行，
须在科学闭合后另封输入；当前没有科学通过、角/径向收敛或expandedMCDHF认证。
