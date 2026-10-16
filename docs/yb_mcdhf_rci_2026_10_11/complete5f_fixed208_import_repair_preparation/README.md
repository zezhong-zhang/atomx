# 固定208完整5f判别：仅修驱动导入的后继准备

[原唯一launch](../complete5f_fixed208_preparation/root_execution.json)以ImportError退出，
尚未读取NPZ或调用任何H/CI/Dirac/梯度等producer，全部attempted/completed为0。
原驱动、实际source、binding、失败和540项准备证据保留原字节；540检查没有验证
模块导出名，不能把其PASS写成动态导入或科学资格。该失败是驱动接口错误，不是
5f物理或数值门失败，没有自动重试。

本后继[驱动](run.py)只改变一行：
`from atomx.numerics.bspline import dirac_assembly as dirac`。
冻结f043确有dirac_assembly.py及solve_dirac；不存在dirac.py。
[合同](contract.json)保留literal208、同原29波/P/3451metric/五UID、完整1595扇区、
核h有限径向seed、全部原科学门和原资源cap，没有其它driver变化。
只增原失败证据哈希及已由旧208合同声明的screened_models直接哈希，防止遗漏
transitive JSON身份。模型字节与原准备相同。

[113项静态证明](preparation.json)核对14个Atomx导入名的冻结实际文件/AST导出、
所有消费JSON字段schema、356源字节、前后驱动逐字一行差和原失败零调用。
没有导入NumPy/SciPy/Atomx、读取数组、编译或执行科学驱动。
公共CSF/磁行/兼容overlappair等科学前置检查仍未执行；旧1m/2000门不改。

另有[可选30秒导入smoke](import_smoke.py)，需root单独精确binding才会导入。
它只检查14个name的实际module文件，禁止NPZ读取，不构造CSF/核potential/
ConvolCache或调用任何物理producer。CPU31/线程1/3.95GiB，外层45秒；不使用
browser窗口CPU30，不修改共享源或原失败。此处尚未执行smoke，静态PASS没有
动态imports或科学PASS含义。

```sh
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_import_repair_preparation/import_smoke.py --binding docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_import_repair_preparation/import_smoke_binding.json
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_import_repair_preparation/run.py --authorize --binding docs/yb_mcdhf_rci_2026_10_11/complete5f_fixed208_import_repair_preparation/execution_binding.json
```

两个binding均待root另审另建；新科学launch须另有完整root seal，不能把smoke授权
当作2H/2CI授权。科学预算仍2H/2CI/14hS、两个全118谱及原有界Jacobi，
核场5f只是预声明有限径向候选；零新增Dirac/Fock/g/optimizer。
原两驻值和实验失败保持，不能声明角/径向收敛或expandedMCDHF。
