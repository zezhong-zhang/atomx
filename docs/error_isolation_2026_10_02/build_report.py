"""Validate independent controls, preserve archive identities, write attribution."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = HERE.parent / "error_attribution_2026_10_02"
OLD = HERE.parent / "atomic_consistency_2026_10_02"
REF = HERE.parent / "systematic_xray_2026_10_02"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changes(a, b):
    return 100 * (np.asarray(a) / np.asarray(b) - 1)


def cell(values):
    return " / ".join(f"{v:+.6f}%" for v in values)


def main():
    prior_identity = read(PRIOR / "identity.json")
    for path, expected in prior_identity.items():
        assert sha(Path(path)) == expected, path
    old_identity = read(OLD / "result_identity.json")
    for path, expected in old_identity["files_sha256"].items():
        assert sha(OLD / path) == expected, path
    refs = read(REF / "references.json")
    rows, summary, fixed_table, field_table, fe_table, fits = [], {}, [], [], [], []
    reproduction_max = closure_max = 0.0
    for el in ("C", "Ne"):
        fixed = read(HERE / f"{el}-fixed-field.json")
        sic = read(HERE / f"{el}-sic-density.json")
        archive = read(PRIOR / f"pi-{el}.json")["variants_Mb"]["native_16000"]
        v = fixed["variants_Mb"]
        for shell in ("1s", "2s", "2p"):
            reproduction_max = max(
                reproduction_max,
                max(
                    abs(
                        np.asarray(v["scf180_rebuilt_field4000"][shell])
                        / archive[shell]
                        - 1
                    )
                ),
            )
            closure_max = max(
                closure_max,
                max(
                    abs(
                        np.asarray(v["fixed_field180_bound180"][shell])
                        / v["scf180_native_field_linear"][shell]
                        - 1
                    )
                ),
            )
            np.testing.assert_allclose(
                v["scf180_rebuilt_field4000"][shell], archive[shell], rtol=1e-11
            )
            np.testing.assert_allclose(
                v["fixed_field180_bound180"][shell],
                v["scf180_native_field_linear"][shell],
                rtol=3e-9,
            )
            fixed_table.append(
                f"| {el} {shell} | {cell(changes(v['fixed_field180_bound360'][shell], v['fixed_field180_bound180'][shell]))} | {cell(changes(v['scf360_rebuilt_field4000'][shell], v['scf180_rebuilt_field4000'][shell]))} |"
            )
            field_table.append(
                f"| {el} {shell} | {cell(changes(v['scf180_rebuilt_field128000'][shell], v['scf180_rebuilt_field4000'][shell]))} | {cell(changes(v['scf180_no_sic_field128000'][shell], v['scf180_no_sic_field4000'][shell]))} |"
            )
            np.testing.assert_allclose(
                sic["variants_Mb"]["linear_density_4000"][shell],
                v["scf180_rebuilt_field4000"][shell],
                rtol=1e-11,
            )
        for category, data in (("fixed_field", fixed), ("sic_sampling", sic)):
            for variant, values in data["variants_Mb"].items():
                for shell, points in values.items():
                    assert (
                        len(points) == 3
                        and np.isfinite(points).all()
                        and min(points) > 0
                    )
                    for photon, sigma, ref in zip(
                        refs[el]["photons_eV"],
                        points,
                        refs[el]["fac_pi"][shell],
                        strict=True,
                    ):
                        rows.append(
                            dict(
                                element=el,
                                category=category,
                                variant=variant,
                                shell=shell,
                                photon_eV=photon,
                                sigma_Mb=sigma,
                                archived_fac_Mb=ref,
                                control_over_archived_fac=sigma / ref,
                            )
                        )
        for name, parameters in sic["sic_parameters"].items():
            fits.append(dict(element=el, variant=name, **parameters))
        summary[el] = dict(
            full_scf360_high_2p_change_percent=changes(
                v["scf360_rebuilt_field4000"]["2p"], v["scf180_rebuilt_field4000"]["2p"]
            )[-1],
            fixed_bound360_high_2p_change_percent=changes(
                v["fixed_field180_bound360"]["2p"], v["fixed_field180_bound180"]["2p"]
            )[-1],
            rebuilt128000_high_2p_change_percent=changes(
                v["scf180_rebuilt_field128000"]["2p"],
                v["scf180_rebuilt_field4000"]["2p"],
            )[-1],
            no_sic128000_high_2p_change_percent=changes(
                v["scf180_no_sic_field128000"]["2p"], v["scf180_no_sic_field4000"]["2p"]
            )[-1],
            exact_density_only_high_2p_change_percent=changes(
                sic["variants_Mb"]["exact_density_128000"]["2p"],
                sic["variants_Mb"]["linear_density_128000"]["2p"],
            )[-1],
            smooth_hartree_high_2p_change_percent=changes(
                sic["variants_Mb"]["smooth_hartree_128000"]["2p"],
                sic["variants_Mb"]["linear_density_128000"]["2p"],
            )[-1],
            smooth_hartree64000_to128000_max_percent=max(
                abs(z)
                for shell in ("1s", "2s", "2p")
                for z in changes(
                    sic["variants_Mb"]["smooth_hartree_128000"][shell],
                    sic["variants_Mb"]["smooth_hartree_64000"][shell],
                )
            ),
        )
    fe = read(HERE / "Fe-fac-threshold.json")
    native = read(PRIOR / "pi-Fe.json")["variants_Mb"]["native_16000"]
    fe_common_native_ratios = {}
    for shell, data in fe["shells"].items():
        v = data["variants_Mb"]
        fe_common_native_ratios[shell] = (
            np.asarray(native[shell]) / v["native"]
        ).tolist()
        np.testing.assert_allclose(
            v["resolved"], refs["Fe"]["fac_pi"][shell], rtol=1e-10
        )
        for variant, points in v.items():
            for photon, sigma, ref in zip(
                refs["Fe"]["photons_eV"], points, v["resolved"], strict=True
            ):
                rows.append(
                    dict(
                        element="Fe",
                        category="fac_threshold_mapping",
                        variant=variant,
                        shell=shell,
                        photon_eV=photon,
                        sigma_Mb=sigma,
                        archived_fac_Mb=ref,
                        control_over_archived_fac=sigma / ref,
                    )
                )
        fe_table.append(
            f"| Fe {shell} | {cell(changes(v['centroid'], v['resolved']))} | {cell(changes(v['native'], v['resolved']))} |"
        )
    for name, table in (("pi_controls.csv", rows), ("sic_fit_parameters.csv", fits)):
        keys = list(dict.fromkeys(k for row in table for k in row))
        with (HERE / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, keys)
            writer.writeheader()
            writer.writerows(table)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = (
        """# 剩余 PI 误差隔离：固定势、SIC 重建与 Fe 阈值

这是[逐项误差归因](../error_attribution_2026_10_02/README.md)的补充。新控制确认：C/Ne 原来的径向敏感性包含 SIC 尾拟合及其输入场采样误差，不能只归给绑定 B-spline 数量。Fe 2p 的分辨阈值后处理与共同平均阈值也贡献数十 ppm。本轮只增加诊断脚本和结果，未改生产实现、FAC/XATOM 或旧归档。

所有百分比均是指定控制相对指定基线的截面变化，不是已证明的绝对误差；三点按光子能量从低到高排列。C 为 353/589/1178 eV，Ne 的实际光子点见 JSON/CSV（以下自动核验具体值）。外部能量控制只用于归因，不作为原生修正。

## 固定势把绑定表示与 SCF 场拆开

从 ns=180 收敛 Gu 密度在原 Gauss 网格重建 SCF 使用的中心场；固定这个物理场和原归档 ΔSCF 阈值，重新求 ns=180/260/360 绑定态，并使用同一连续场与同一 16000 点 shooting 网格。固定势重新求 ns=180 与原绑定态的 PI 相对闭合优于 3×10⁻⁹，验证了状态适配与组态/占据映射。

| 壳层 | 固定场，只换绑定 ns=180→360 | 完整 SCF ns=180→360，连续场按原协议重建 |
|---|---|---|
"""
        + "\n".join(fixed_table)
        + """

Ne 高能 2p 的完整变化为 +0.152331%，固定势只换绑定表示则 +0.009538%。后者不能解释前者；SCF 场与连续场重建是额外数值来源。C 高能 2s 的绑定表示本身仍有约 +0.0195% 变化，也不能直接忽略。用 PCHIP 而非线性插值表示固定物理场的独立实验保存在 JSON 中；固定势离散控制也有插值误差，不把它等同于精确连续场。

## SIC 与场采样：明确代码机制及控制

绑定态 [Gu 场](../../src/atomx/methods/dirac_fock.py:1370) 在 Gauss 网格调用 [SIC 修正](../../src/atomx/numerics/bspline/local_potential.py:24)。连续态 [static_field.py:305](../../src/atomx/potentials/static_field.py:305) 先线性插值密度与 Hartree 屏蔽电荷到默认 4000 点对数网格，再重新拟合 SIC。FAC 原 [radial.c:1957](/home/zzhang/code/fac/faclib/radial.c:1957) 与 atomx 的算法均选择 screening<N−1.55 的离散转接点，并用附近六个点估计对数斜率；移到不均匀 Gauss 网格或对已经线性插值的场重新拟合，会改变转接点与斜率。本轮没有发现这段公式的直接移植符号错误。

以下保持 ns=180 的所有绑定系数、阈值、Hartree Gauss 积分和连续 shooting 网格不变，只将场重建网格从 4000 增加到 128000。关闭 SIC 只用于隔离敏感性；其变化比较自身无 SIC 的 4000 点基线，并非建议删掉该物理模型。

| 壳层 | 原 SIC：重建网格 4000→128000 | 关闭 SIC：重建网格 4000→128000 |
|---|---|---|
"""
        + "\n".join(field_table)
        + """

Ne 高能 2p 的原 SIC 场变化 −0.116375%，关闭 SIC 后只有 +0.000594%。固定绑定态直接冻结原 Gauss SCF 场，相对原重建场又变化 +0.093922%。这些独立控制说明默认 frozen Gu 连续场未严格沿用绑定 SCF 势，并且尾拟合放大了重新采样的误差。百分比不能相加为唯一归因或“修正后精度”。

继续分解输入采样：仅将线性密度改为直接 B-spline P/Q 取值，在 128000 点的 Ne 高能 2p 只改约 +0.0093%；再把 Hartree 屏蔽电荷改为 PCHIP，累计相对原线性输入改约 +0.0866%。因此密度插值并非全部原因，Hartree 输入的局部斜率也显著影响 SIC。PCHIP 是诊断对照，不是精确势或已经验收的生产修复；它和原线性重建的细网格极限也不同。

[实际拟合参数](sic_fit_parameters.csv)通过 Python trace 直接记录运行函数的局部变量，没有另写拟合算法来推断参数。Ne 原线性输入：4000→128000 点的尾部 b 从 3.406975 到 3.481259；直接密度＋PCHIP Hartree 的 b 为 3.406667→3.426011。后者 64000→128000 的全部九点最大变化约 0.00177%，仍非完整收敛证明。C 的相同实验也显示输入采样与离散转接点敏感，详见[汇总](summary.json)、[C](C-sic-density.json)、[Ne](Ne-sic-density.json)。

## Fe：保留 FAC RR 强度，只改变阈值映射

FAC [TotalPICross]( /home/zzhang/code/fac/faclib/interpolation.c:2313) 对每条初态→终态通道从 EN 能量求阈值，按动能与 log(photon/threshold) 对归档 RR 强度插值；atomx [subshell.py](../../src/atomx/scattering/photoionization/subshell.py) 使用每轨道输入阈值，原面板对一个 n-l 空穴赋共同 ΔSCF。Fe 开壳层总 J 不能直接当作空穴 j，因此本控制整组移到简并度 centroid，没有误用闭壳层 J 映射。

使用 FAC `AdjustEnergy` 生成独立 EN 表，初态/终态各自压到简并度 centroid，或把终态 centroid 移到初态 centroid＋Gu 原生 ΔSCF。EN 原双精度能量直接读取，避免打印表的有效位舍入。FAC RR 二进制与全部径向强度保持原样，使用 FAC 自身 TotalPICross 插值；原分辨控制复现九点归档至 10⁻¹⁰。该实验量化的是阈值后处理，**不重新生成频率依赖的 RR 强度，不等同于完整同阈值重算或 CI Hamiltonian 松弛**。

| 壳层 | FAC centroid / FAC 分辨 − 1 | FAC Gu 原生共同阈值 / FAC 分辨 − 1 |
|---|---|---|
"""
        + "\n".join(fe_table)
        + """

Gu/原分辨 FAC 的 Fe 2p 比值为 0.99988317 / 0.99988403 / 0.99989144；Gu/FAC 共同原生阈值控制变为 0.99996008 / 0.99994477 / 0.99993476。这解释了原偏差的一部分，余下约 40–65 ppm 仍混合径向、核模型、有限频率强度/插值协议差异。Fe K/2s 的纯 centroid 压缩仅亚 ppm；不得把 Fe 2p 结论泛化到全部壳层。

## 修复方向与剩余边界

冻结场应保存并复用绑定 Gu SCF 实际使用的中心势及参数；普通插值不应悄悄触发第二次 SIC 拟合。若要消除 SCF 自身对网格的敏感性，还需在统一、足够光滑的场表示上控制转接点和斜率，并验证物理 SIC、尾电荷、绑定轨道与连续态共同收敛。本轮的“改密度/PCHIP”实验不能替代该一致性设计。

分辨阈值应来自原生目标态 CI Hamiltonian 与相应轨道松弛，然后逐通道求光电子动能与强度；FAC 分辨能量只能用于对照。此前 C 荧光的强度–ω³相关性、Xe/Kr jj 阈值、FAC Fe 弱线输出截断、有限核协议和 XATOM 高能积分问题见[完整前轮归因](../error_attribution_2026_10_02/README.md)，本轮未推翻这些结论。HF/Gu 模型差别、XATOM 共同场 Dirac/Schrödinger 分解、Fe Auger 缺失 CI 空间/松弛仍待独立控制；Auger 与 ωK 未重算，也没有混用新辐射宽度和旧 Auger 产额。

## 数据、复现与验证

全部新控制的逐点 CSV：[pi_controls.csv](pi_controls.csv)。原始输出：[C 固定场](C-fixed-field.json)、[Ne 固定场](Ne-fixed-field.json)、[Fe FAC 阈值](Fe-fac-threshold.json)。`raw/Fe/threshold-*` 保存 FAC 脚本、移位 EN、逐初态截面及日志；没有修改旧 FAC 文件。FAC API 要求 AI 输入，本脚本使用从有效数据库头生成的零块空 AI 表，只改变 EN；输入移位文件的 hash 也进入脚本缓存身份。

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FAC_SFAC_PATH="$HOME/code/fac/bin/sfac"
D=docs/error_isolation_2026_10_02
.venv/bin/python "$D/fixed_field_controls.py" C Ne
.venv/bin/python "$D/sic_density_controls.py" C Ne
.venv/bin/python "$D/fac_threshold_controls.py" Fe
.venv/bin/python "$D/build_report.py"
.venv/bin/ruff check "$D" --ignore E402
```

本轮检查的是诊断闭合、原生复现、完整有限/正截面、旧归档和前轮身份哈希。未修改生产实现，未另跑生产测试套件；前轮“97 项通过、2 项预期失败”不作为本轮新验证成绩。
"""
    )
    # Keep actual photon grids explicit, rather than infer from element/Z.
    grids = "; ".join(f"{el}: {refs[el]['photons_eV']} eV" for el in ("C", "Ne", "Fe"))
    report = report.replace(
        "C 为 353/589/1178 eV，Ne 的实际光子点见 JSON/CSV（以下自动核验具体值）。",
        grids + "。",
    )
    report = report.replace("]( /home", "](/home")
    (HERE / "README.md").write_text(report)
    validation = dict(
        pi_control_values=len(rows),
        prior_identity_files_verified=len(prior_identity),
        original_archive_files_verified=len(old_identity["files_sha256"]),
        native_pi_reproduction_max_fraction=reproduction_max,
        fixed_bound180_closure_max_fraction=closure_max,
        fac_resolved_reproduction_rtol=1e-10,
        fe_commonthreshold_native_ratios=fe_common_native_ratios,
        production_changed=False,
        old_archives_unchanged=True,
    )
    (HERE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    artifacts = sorted(
        p
        for p in HERE.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name not in ("identity.json", "validation.log")
    )
    source_paths = [
        ROOT / "src/atomx/potentials/static_field.py",
        ROOT / "src/atomx/numerics/bspline/local_potential.py",
        ROOT / "src/atomx/methods/dirac_fock.py",
        Path("/home/zzhang/code/fac/faclib/radial.c"),
        Path("/home/zzhang/code/fac/faclib/interpolation.c"),
        Path("/home/zzhang/code/fac/faclib/dbase.c"),
        Path("/home/zzhang/code/fac/bin/sfac").resolve(),
        PRIOR / "identity.json",
        OLD / "result_identity.json",
    ]
    source_paths += [
        REF / "raw/Fe" / f"fac-pi-{shell}" / file
        for shell in ("1s", "2s", "2p")
        for file in ("levels.b", "pi.b")
    ]
    source_paths += [
        ROOT / file
        for file in (
            "src/atomx/scattering/radiative_recombination/__init__.py",
            "src/atomx/scattering/photoionization/subshell.py",
            "src/atomx/scattering/models.py",
            "src/atomx/constants.py",
            "src/atomx/numerics/bspline/evaluate.py",
            "src/atomx/numerics/yk.py",
            "src/atomx/physics/local_exchange.py",
            "benchmarks/fac_runner.py",
            "benchmarks/bench_systematic_xray.py",
        )
    ]
    identity = {str(p): sha(p) for p in artifacts + source_paths}
    (HERE / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
