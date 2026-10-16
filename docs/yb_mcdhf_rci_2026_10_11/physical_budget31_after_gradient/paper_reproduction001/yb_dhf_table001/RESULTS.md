# 实算与重放说明

当前采用`actual004/result.json`、`static_moments003/result.json`和
`saved_data_review002.json`。早期带PASS字样的003仅通过其当时的实时控制；后续回读
发现ns180 s通道完整物理Gram为3.66591×10⁻¹⁰，不能作为最终完整态认证。
失败的`saved_data_review.json`和原数组保持不变。

`run.py --prepare`封存第一份合同、实际359个生产源文件和压缩字节快照。
首次执行完成ns180的53次SCF迭代后，附着态输入Gram检查失败。
`diagnosis001/result.json`证实原始输入Gram和补空间Gram分别约3.37/2.54×10⁻¹⁰。
`run002.py`加入完整列的Cholesky QR，保留第一份失败和核检查点；ns220 s通道仍失败。
`run003.py`在同一源/核/模型/资源下使用可逆度量白化，完成两套模型和CI账。
但补空间内的度量通过没有保证嵌回后的完整物理度量通过，故003最终被回读限定。

`diagnosis003/result.json`和`physical_pair_diagnosis.json`保留未奏效的表示修正：
全列Gram变好，部分原始根残差仍超门。`physical_pair_diagnosis002.json`
随后验证，在完整物理度量重建时保留已经修正的首列，可以同时保留根和整列Gram。
`constrained_pair.py`的三组独立四维解析模型包含真实核/目标耦合，要求保留核约束乘子。
不得将局部驱动解释为已修改公共Atomx默认求解器。

`repair_saved004.py`消费完整已存H/S和两套核检查点，未重新SCF或构造Fock；
每个目标在原始物理H/S上作两次含占据约束的边界方程修正，
完整虚电子列重建后逐位保留该首列，再核验全部物理Gram、核正交性、原始根残差及根间隔。
之后对两个粒子扇区重新作四次原生CI装配。没有丢列、换根、缩模型或放宽门。
源/输入、数值范围和资源详见`contract004.json`及其003→002→001链。

`static_moments001.py`首次适配遗漏共享角张量到标准静态矩的相位转换，
d3/2的独立解析角因子检查失败−0.4。原共享张量约定和旧失败均未改。
后继002明确转换并与另一FAC/CRE系数及解析单d电子公式分别核对；
003在最终004的实际P/Q上重算。电子电荷负号只使用一次。
独立SciPy样条逐胞Gauss积分与原生r²径向积分最大差3.56×10⁻¹⁵ a.u.。

固定环境：现有`.venv`、CPU27、BLAS/OMP/Numba单线程、任务独立Numba缓存。
初始科学执行上限8 GiB/1200秒，004上限8 GiB/120秒；两次SCF均53次迭代。
003的11.45秒和004的465.98 MiB RSS仅是各自后继作用域，不能当作两套从头SCF的总成本。
早期最高RSS约2.4 GiB，超过最初“小于2 GiB”的预估，仍低于8 GiB硬上限。
这里没有受控性能比较或速度提升结论。

验证面板：

```text
tests/test_generalized_eigenblock.py
tests/test_dhf_nuclear_profile.py
tests/test_dhf_diis_residual.py
tests/test_dhf_seeded.py
12 passed; tests.xml / tests.log
```

独立缓存、CPU26；验证33.30秒。生产源码没有变化，所以没有声称修复此前012的两项历史回归失败。
新驱动的数值自检、保存数据回读和源码压缩包逐文件复核另计，不能把重复控制相加当作独立样本数。
本目录运行脚本使用不覆盖的`exist_ok=False`输出目录；重新执行必须使用新的明确证据目录，
保留现有成功与失败，不能直接覆盖归档后反复运行。
