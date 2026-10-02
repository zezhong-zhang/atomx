"""Join diagnostics without modifying the archived benchmark or references."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / "atomic_consistency_2026_10_02"
REF = HERE.parent / "systematic_xray_2026_10_02"
ELEMENTS = ("C", "Ne", "Ar", "Fe", "Kr", "Xe")


def read(name):
    return list(csv.DictReader((HERE / name).open()))


def keyed(name):
    return {(r["element"], r["shell"], float(r["photon_eV"])): r for r in read(name)}


def save(name, rows):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ratio_range(values):
    return f"{min(values):.6f}–{max(values):.6f}"


def main():
    inventory = read("pi_inventory.csv")
    focused = keyed("fac_pi_grid.csv")
    point = keyed("fac_pi_point.csv")
    radial = keyed("fac_pi_radial6400.csv")
    xhalf = keyed("xatom_pi_grid.csv")
    xquarter = keyed("xatom_pi_grid_quarter.csv")
    pi_rows = []
    pi_table = []
    x_table = []
    reasons = {
        ("C", "1s"): "余差与径向数值敏感性同量级；多重态阈值需保留，未唯一归因。",
        ("C", "2s"): "绑定态/势场离散化与 FAC 径向积分敏感；共用阈值差只影响数 ppm。",
        ("C", "2p"): "双侧径向网格变化超过原偏差；不能把接近 1 当作已收敛精度。",
        ("Ne", "1s"): "双侧径向网格敏感；有限核和阈值贡献很小。",
        ("Ne", "2s"): "主要为径向数值敏感性；细化绑定网格会改符号。",
        ("Ne", "2p"): "主要为径向数值敏感性；jj 阈值控制的变化仅数十 ppm。",
        ("Ar", "1s"): "点核/有限核及径向数值均有 ppm 级贡献；无唯一主因。",
        ("Ar", "2s"): "ppm 级径向/能量/核模型余差；尚未完全隔离。",
        ("Ar", "2p"): "jj 阈值贡献很小；剩余数十 ppm 仍含 FAC 径向与频率插值。",
        ("Fe", "1s"): "有限核参考与点核原生的差异解释主要偏差。",
        ("Fe", "2s"): "点核/有限核贡献与原偏差同量级，另有 ppm 级径向余差。",
        (
            "Fe",
            "2p",
        ): "开 3d 壳层的分辨终态/强度与单一阈值不同；径向/阈值平均贡献尚未完全隔离。",
        ("Kr", "1s"): "有限核参考与点核原生的差异解释主要偏差。",
        ("Kr", "2s"): "点核/有限核贡献解释主要偏差，高能点仍有小余差。",
        ("Kr", "2p"): "两个 jj 空穴共用阈值，分辨阈值控制显著闭合。",
        ("Xe", "1s"): "有限核参考与点核原生的差异解释主要偏差。",
        ("Xe", "2s"): "有限核参考及 FAC 径向数值共同贡献。",
        ("Xe", "2p"): "共用 ΔSCF 阈值遗漏 324 eV jj 分裂，是最大误差主因。",
    }
    refs = json.loads((REF / "references.json").read_text())
    archived = list(csv.DictReader((OLD / "pi.csv").open()))
    for el in ELEMENTS:
        record = json.loads((HERE / f"pi-{el}.json").read_text())
        assert record["neutral_minus_archived_Ha"] == 0
        v = record["variants_Mb"]
        for shell in ("1s", "2s", "2p"):
            group = []
            for i, photon in enumerate(refs[el]["photons_eV"]):
                key = (el, shell, photon)
                original = next(
                    r
                    for r in archived
                    if (
                        r["element"],
                        r["shell"],
                        r["exchange"],
                        r["field"],
                        r["energy_mode"],
                        float(r["photon_eV"]),
                    )
                    == (el, shell, "gu", "frozen", "relaxed", photon)
                )
                assert np.isclose(
                    v["native_16000"][shell][i],
                    float(original["atomx_Mb"]),
                    rtol=1e-11,
                    atol=0,
                )
                fac = refs[el]["fac_pi"][shell][i]
                xr = refs[el]["xatom_pi"][str(int(photon))][shell]
                row = dict(
                    element=el,
                    shell=shell,
                    photon_eV=photon,
                    native_over_fac=float(original["ratio"]),
                    native_over_xatom=float(original["ratio_xatom"]),
                    fac_jj_threshold_control_over_fac=v["fac_jj_threshold_control"][
                        shell
                    ][i]
                    / fac,
                    fac_centroid_control_over_fac=v["fac_centroid_control"][shell][i]
                    / fac,
                    resolved_jj_control_valid=record["resolved_jj_control_valid"],
                    native_over_fac_point=v["native_16000"][shell][i]
                    / float(point[key]["fac_focused_Mb"]),
                    continuum_32000_over_16000=v["native_32000"][shell][i]
                    / v["native_16000"][shell][i],
                    fac_focused_over_original=float(
                        focused[key]["focused_over_original"]
                    ),
                    fac_radial6400_over_focused=float(radial[key]["fac_focused_Mb"])
                    / float(focused[key]["fac_focused_Mb"]),
                    ns260_over_180=None,
                    ns360_over_180=None,
                    static_over_native=v["long_wavelength_control"][shell][i]
                    / v["native_16000"][shell][i],
                    xatom_threshold_static_control_over_xatom=v[
                        "xatom_threshold_static_control"
                    ][shell][i]
                    / xr,
                    xatom_drhalf_over_original=float(xhalf[key]["fine_over_original"]),
                    xatom_drquarter_over_half=float(xquarter[key]["xatom_fine_Mb"])
                    / float(xhalf[key]["xatom_fine_Mb"]),
                    diagnosis=reasons[el, shell],
                )
                for ns in (260, 360):
                    path = HERE / f"pi-{el}-ns{ns}.json"
                    if path.exists():
                        fine = json.loads(path.read_text())
                        row[f"ns{ns}_over_180"] = (
                            fine["variants_Mb"]["native_16000"][shell][i]
                            / v["native_16000"][shell][i]
                        )
                group.append(row)
                pi_rows.append(row)
            pi_table.append(
                f"| {el} {shell} | {ratio_range([r['native_over_fac'] for r in group])} | {reasons[el, shell]} |"
            )
            x_table.append(
                f"| {el} {shell} | {ratio_range([r['native_over_xatom'] for r in group])} | {ratio_range([r['xatom_threshold_static_control_over_xatom'] for r in group])} |"
            )
    save("pi_attribution.csv", pi_rows)

    channels = read("fluorescence_channels.csv")
    cuts = [r for r in read("fac_transition_cuts.csv") if r["cut_mode"] == "all_full"]
    totals = read("fluorescence_totals.csv")
    fluorescence_rows = []
    fluorescence_table = []
    hf_table = []
    hf_pi_table = []
    for el in ELEMENTS:
        for shell in ("1s", "2s", "2p"):
            group = [r for r in inventory if r["element"] == el and r["shell"] == shell]
            ratios = [float(r["hf_frozen_relaxed_over_fac"]) for r in group]
            delta = float(group[0]["hf_threshold_minus_fac_eV"])
            hf_pi_table.append(
                f"| {el} {shell} | {ratio_range(ratios)} | {delta:+.6f} |"
            )
    x_rad_table = []
    for row in totals:
        el, model, hole = row["element"], row["exchange"], row["hole"]
        source_rows = [
            r
            for r in channels
            if (r["element"], r["exchange"], r["hole"]) == (el, model, hole)
        ]
        resolved, collapsed = 0.0, 0.0
        for ch in source_rows:
            if el == "Fe":
                full = next(
                    r for r in cuts if (r["hole"], r["source"]) == (hole, ch["source"])
                )
                resolved += float(full["resolved_per_s"])
                gu_ch = next(
                    r
                    for r in channels
                    if (r["element"], r["exchange"], r["hole"], r["source"])
                    == ("Fe", "gu", hole, ch["source"])
                )
                collapsed += (
                    float(full["same_native_photon_per_s"])
                    * (float(ch["native_photon_eV"]) / float(gu_ch["native_photon_eV"]))
                    ** 3
                )
            else:
                resolved += float(ch["fac_resolved_per_s"])
                collapsed += float(ch["fac_same_native_photon_per_s"])
        out = dict(row)
        out.update(
            fac_complete_per_s=resolved,
            fac_complete_same_native_photon_per_s=collapsed,
            native_over_fac_complete=float(row["native_per_s"]) / resolved
            if resolved
            else None,
            native_over_fac_complete_same_photon=float(row["native_per_s"]) / collapsed
            if collapsed
            else None,
        )
        if resolved:
            assert np.isclose(
                out["native_over_fac_complete"],
                out["native_over_fac_complete_same_photon"] * collapsed / resolved,
                rtol=1e-13,
            )
        fluorescence_rows.append(out)
        if model == "gu":
            if not resolved:
                fluorescence_table.append(
                    f"| {el} {hole} | 0/0 | 0/0 | 无允许的向下 E1 通道 |"
                )
                continue
            if el == "C":
                reason = "分辨 LS 能级的强度–频率相关性；径向/角系数控制闭合。"
            elif el == "Fe":
                reason = "FAC 输出截断漏弱线；去截断后还须保留能级/ω³相关性。"
            elif hole == "2p":
                reason = "jj 分裂与单一平均频率；小径向余差见逐源控制。"
            else:
                reason = "能级频率影响小；剩余主要为小的径向/核模型数值差。"
            fluorescence_table.append(
                f"| {el} {hole} | {float(row['native_over_fac']):.6f} | {out['native_over_fac_complete_same_photon']:.6f} | {reason} |"
            )
            x_rad_table.append(
                f"| {el} {hole} | {float(row['native_over_xatom']):.6f} | {float(row['native_over_xatom_same_photon']):.6f} |"
            )
        elif resolved:
            hf_table.append(
                f"| {el} {hole} | {float(row['native_over_fac']):.6f} | {out['native_over_fac_complete_same_photon']:.6f} |"
            )
    save("fluorescence_attribution.csv", fluorescence_rows)
    max_shoot = max(abs(r["continuum_32000_over_16000"] - 1) for r in pi_rows)
    max_fac_e = max(abs(r["fac_focused_over_original"] - 1) for r in pi_rows)
    max_x_q = max(abs(r["xatom_drquarter_over_half"] - 1) for r in pi_rows)
    text = f"""# 六元素误差逐项归因：atomx / FAC / XATOM

本报告回查 [原始六元素面板](../atomic_consistency_2026_10_02/README.md) 的全部 K/L 子壳层 PI 与荧光误差，包含 HF、Gu 与两套外部参考。原始 432/108 行、外部参考与源码归档保持不变。本轮只增加诊断运行，没有修改物理实现或用外部能量校准原生结果。

结论：C 的两项荧光异常主要来自分辨能级；Xe 2p PI 的最大偏差主要来自共用阈值遗漏 jj 分裂；Fe 荧光小余差主要来自 FAC 默认弱线输出截断；重元素 K/2s PI 的小余差很大部分来自点核与有限核协议不同。C/Ne PI 仍有径向网格敏感性，不能把原比值的小数位当成收敛精度。没有证据说明这些问题都已通过 Hamiltonian CI 松弛解决。

## 1. Gu 光电离对 FAC：18 个格逐项检查

| 元素/壳层 | 原生 Gu/FAC 三能量点范围 | 代码与控制支持的归因 |
|---|---:|---|
{chr(10).join(pi_table)}

全部 54 点及每项控制见 [pi_attribution.csv](pi_attribution.csv)；原生 HF/Gu 场与阈值变体、原生阈值和 FAC 分辨范围见 [pi_inventory.csv](pi_inventory.csv)。

**阈值遗漏 jj 分裂已确认。** [benchmark:82](../../benchmarks/bench_atomic_consistency.py:82) 对同一个 n-l 空穴只求一个统计组态能量，并把同一个 ΔSCF 阈值分配给两个 jj 轨道。[subshell.py:73](../../src/atomx/scattering/photoionization/subshell.py:73) 又把该阈值直接用于连续态动能。Xe FAC 的 2p₃/₂、2p₁/₂ 阈值为 4795.77196、5120.00421 eV，分裂 324.23225 eV；atomx 原面板共用一个阈值。保留 Gu 轨道和连续场，只换 FAC 分辨阈值时，39747 eV 点的比值从 0.9973629 到 {next(r["fac_jj_threshold_control_over_fac"] for r in pi_rows if r["element"] == "Xe" and r["shell"] == "2p" and r["photon_eV"] == 39747):.7f}。Kr 同类控制也显著闭合。该映射仅对闭壳层 J=0 初态成立；C/Fe 开壳层控制只换 FAC 简并度平均阈值，不能冒充逐 jj 分辨处理。

**核协议不同已确认。** atomx 驱动显式 `nuclear='point'`；原 FAC 脚本只调用 `SetAtom('Xe')`。[sfac.c:1955](/home/zzhang/code/fac/sfac/sfac.c:1955) 默认 `rn=-1`，[nucleus.c:462](/home/zzhang/code/fac/faclib/nucleus.c:462) 取有限 RMS 核半径。只改 FAC 为 `SetAtom('Xe',-1,-1,0)` 的点核控制，Xe 高能 K 截面增加约 0.0393%，几乎解释原 Gu/FAC +0.0394%。Fe、Kr 的 K/2s 也有同类贡献。点核与径向网格控制是独立实验，不能把两个百分比直接相加当作联合计算。

**径向数值敏感性已确认，但 C/Ne 的具体误差项还未唯一拆开。** 同一 Gu 初态、同一原生阈值，连续 shooting 网格 16000→32000 的全部 54 点最大变化只有 {max_shoot * 100:.6f}%。FAC 60 点电子能量网格集中到目标能区，最大变化 {max_fac_e * 100:.6f}%，不足以解释大多数余差。绑定 B-spline ns=180→260→360 与 FAC 径向 3200→6400→12800 的 C/Ne 结果却在 10⁻⁴–10⁻³ 量级变化；Ne 高能 2p 的 ns=360 截面比原生 ns=180 增加约 0.1523%。这说明 SCF 的 `tol=1e-9` 不能代替截面网格收敛证明。

具体候选代码是 [Gu 势构造](../../src/atomx/methods/dirac_fock.py:1335)、[离散 SIC 尾修正](../../src/atomx/numerics/bspline/local_potential.py:24) 和 [冻结连续场采样](../../src/atomx/potentials/static_field.py:305)：SIC 在不同网格上选择转接点并用邻点拟合尾部。当前 ns 控制同时改变绑定表示与势离散，尚未用固定外部势把二者拆开，故不把 SIC 单独定为已证实根因。Kr 中性 ns=260、300 次迭代没有满足原来的严格收敛条件，见 [失败日志](pi_ns260.log)，本报告不使用该次未收敛结果；Xe 中性 ns=260 没有随该中断继续运行。

双方 PI 都用有限频率相对论 E1：atomx [radial_eL 调用](../../src/atomx/scattering/radiative_recombination/__init__.py:333) 及 [Bessel 算符](../../src/atomx/properties/transitions/multipole.py:71)，FAC [RRRadialQkTable](/home/zzhang/code/fac/faclib/recombination.c:905) / [MultipoleRadialFRGrid](/home/zzhang/code/fac/faclib/radial.c:7273)。余差不能统一归给长波近似或遗漏 E2。FAC [TotalPICross](/home/zzhang/code/fac/faclib/interpolation.c:2415) 使用分辨终态能量与 log 强度插值；atomx 直接求指定动能连续波。它们仍有不同的径向与频率插值，Ar/Fe 2p 小余差保留为未完全隔离项。

## 2. Gu 荧光对 FAC：18 个空穴逐项检查

诊断统一 FAC 分辨线的光子能量到各个 atomx 原生 source→hole ΔSCF 能量，保留 FAC 的简并度权重和 A/E³；Fe 同时使用取消输出截断的完整 FAC 线表。下列“同能量控制”是归因实验，不是替代原生分辨谱或修正参数。

| 元素/空穴 | 原生 Gu/原 FAC | 同能量、完整线表 Gu/FAC | 归因 |
|---|---:|---:|---|
{chr(10).join(fluorescence_table)}

[全部 HF/Gu 36 行](fluorescence_attribution.csv)、[每个 source 的能量与强度](fluorescence_channels.csv)、[逐源 FAC 轨道积分控制](fluorescence_radial_ns180.csv)、[ns=260 径向敏感性](fluorescence_radial_ns260.csv)。后一个控制只重算初态轨道，保持原来的 ΔSCF 光子能量，所以不能当作完整 ΔSCF 能量的收敛重算。

**C 的能级/强度相关性已确认。** 2s→2p 的原生光子是 8.0537967 eV，FAC 简并度平均是 8.0550819 eV，两者几乎相同；但 FAC 分辨线范围 3.825202–13.865520 eV，按线强度加权的立方有效能量为 12.4666465 eV。FAC 分辨宽度比使用其自身组态 centroid 的宽度大 3.70714 倍。atomx 用一个能量乘总偶极强度，故即使 centroid 正确也得不到分辨宽度。C K 的同类因子约 1.015105，解释 1.51% 的 K 偏差。代码差异是 [fluorescence.py:55](../../src/atomx/properties/transitions/fluorescence.py:55) 每个 n-l source 一个能量，对比 FAC [TRMultipole 与分辨线写出](/home/zzhang/code/fac/faclib/transition.c:1324)。

**Fe 默认 FAC 线表漏弱线已确认。** [consts.h:316](/home/zzhang/code/fac/faclib/consts.h:316) 默认 `TRCUT=1e-4`；[transition.c:1358](/home/zzhang/code/fac/faclib/transition.c:1358) 按同一上态的总速率删弱线。Fe 2s 空穴的 2p 源累计强度缺失 3.45%，2p 空穴的 4s 源缺失 7.36%；这些源在总宽度中占比小，故总误差只有约 0.1%。分别取消角系数求和截断与输出截断后，只有取消输出截断恢复强度，见 [fac_transition_cuts.csv](fac_transition_cuts.csv)。完整 FAC 总宽度相对原 FAC：K +0.136588%，2s +0.178213%，2p +0.075866%。Fe K 原生比值由 1.001361 到 0.999995；Fe 2p 原生比值反而由 0.991447 到 0.990695，说明旧的两种误差曾部分抵消。

FAC 命令封装还存在一个可直接核对的编码问题：[sfac.c:3076](/home/zzhang/code/fac/sfac/sfac.c:3076) 把 `SetTransitionCut` 参数声明为 `int`，却用 `atof` 读取，再传给接受 double 的库函数；`1e-4` 因而被截成 0。本轮用 -1 恢复默认、0 关闭相应截断，绕过这个封装问题；没有修改 FAC 源码。

**角系数与占据归一化不是剩余主因。** 用 FAC 自身 P/Q、atomx 的角系数和 per-vacancy 占据规则重算各 source，在取消 Fe 弱线输出截断后，恢复 FAC 的 A/E³ 总强度。C、Ne、Ar、Kr、Xe 的逐源闭合约 10⁻⁶，Fe 同样闭合。这排除了剩余全局 2J、占据或空穴简并度因子错误。对照 atomx [occupation/degeneracy:67](../../src/atomx/properties/transitions/fluorescence.py:67) 与 FAC [分辨态角收缩](/home/zzhang/code/fac/faclib/transition.c:649)。

双方本面板荧光都是长波 E1 长度规范：原 FAC `tr.txt` 明确 `MULTIP=-1, GAUGE=2, MODE=1`，[transition.c:649](/home/zzhang/code/fac/faclib/transition.c:649) 调 `MultipoleRadialNR`，其 [radial.c:7091](/home/zzhang/code/fac/faclib/radial.c:7091) 是径向 r 矩。不能把这些荧光差异解释成 FAC 多极/有限波长修正。

## 3. HF 与 Gu：模型差异和实现错误分开

HF PI 也逐壳层列出。它同时更换了绑定轨道、冻结 Gu 场所用的密度和原生阈值，因此下表不能单独量化非局域交换的贡献；数值、核模型及分辨阈值问题仍与前述控制同类。

| 元素/壳层 | 原生 HF/FAC 三点范围 | HF ΔSCF − FAC 阈值 centroid (eV) |
|---|---:|---:|
{chr(10).join(hf_pi_table)}

| 元素/空穴 | HF/原 FAC | 同原生 HF 能量、完整线表 HF/FAC |
|---|---:|---:|
{chr(10).join(hf_table)}

在相同 HF 原生光子能量下，HF K 的强度差仍是 C +55.6%、Ne +32.2%、Ar +13.9% 等，不能归咎于 ω³ 或统计占据漏乘。HF 的 [非局域 Fock 交换](../../src/atomx/methods/dirac_fock.py:915) 与 Gu/FAC 的 [局域、Z/N 依赖交换](../../src/atomx/methods/dirac_fock.py:1370) 会产生不同径向轨道。PI 中 HF 绑定轨道再配由 HF 密度构造的冻结 Gu 连续场是明确的混合模型；换为 Gu 绑定态及其密度后 C 2p 的大偏差明显缩小。但当前控制同时改变密度与能量，不是只换绑定矩阵元，也不是完整的非局域 HF 连续态计算，不能据此认定 HF 物理错误。

这里的 HF 是统计组态平均，而不是按实际 CI Hamiltonian 能量泛函迭代松弛的轨道。你指出的 Hamiltonian average 要求适用于后续 CI 面板；现有 [CSFOrbitalFunctional](../../src/atomx/methods/csf/variational.py) 已含相干 CI 系数/密度矩阵的变分目标，但这个六元素驱动调用的是 `DFSolver.from_configuration(... averaging='statistical')`。要复现 C/Fe 分辨谱，需要目标态 CI、相应轨道松弛与逐线能量/强度；只换平均占据或一个 average photon 不足够。

## 4. XATOM：先分离阈值、光子算符、径向模型和网格

XATOM 原脚本未开 relativistic，默认 Slater/Latter、无相关泛函。其 [functional_xc.f90:243](/home/zzhang/code/xraypac/xatom/src/functional_xc.f90:243) 是 `-3/2 (3ρ/π)^(1/3)`；Gu/FAC 是 [get_hxs](../../src/atomx/physics/local_exchange.py:89) 的 Z/N 依赖系数并有另一套 SIC 尾修正。原来两个代码中名为 alpha 的参数约定不同，不能比较数字本身。XATOM [pcs.f90:185](/home/zzhang/code/xraypac/xatom/src/pcs.f90:185) 用 -ε 的 Koopmans 阈值、Schrödinger 连续波和静态 r 矩；atomx Gu 用 Dirac 轨道/连续波与有限频率算符。

| 元素/壳层 | 原生 Gu/XATOM | 保留 Gu 径向场、仅换 XATOM 阈值与静态 E1 |
|---|---:|---:|
{chr(10).join(x_table)}

该控制刻意只替换能量与算符，留下 Dirac vs Schrödinger、Gu/SIC vs Slater/Latter 径向模型差别，故不要求比值等于 1。Xe 高能 2p 单独移除有限波长项会使截面乘约 0.9014；继续统一阈值后，仍有明显径向模型差异。尚未用同一势场分别求 Dirac/Schrödinger 来拆开这两个因素，不能把剩余百分比全部指定给相对论效应。

**XATOM 参考网格本身有高能误差。** [pcs.f90:238](/home/zzhang/code/xraypac/xatom/src/pcs.f90:238) 在均匀 dr 网格上用梯形积分。保持 HFS 模型与 GPS N=300 不变，dr=0.0025→0.00125→0.000625 a₀：Xe 132489 eV 的 2s PI 分别下降约 2.6741%、2.7527%（均相对原值）；后两者仍相差约 0.0807%。全 54 点后两步最大变化为 {max_x_q * 100:.4f}%。这只能证明原网格存在数值误差，尚非连续态与绑定 GPS 两者的完整收敛证明。见 [步长减半](xatom_pi_grid.csv) 和 [再次减半](xatom_pi_grid_quarter.csv)。

荧光能量由 XATOM [decay.f90:119](/home/zzhang/code/xraypac/xatom/src/decay.f90:119) 的轨道能量差构造，ω³ 在 [decay.f90:132](/home/zzhang/code/xraypac/xatom/src/decay.f90:132)。保持 XATOM 每源强度，换成原生 Gu source→hole 光子后如下；剩余仍是上述径向模型差别。

| 元素/空穴 | 原生 Gu/XATOM | 同 Gu 原生光子 Gu/XATOM |
|---|---:|---:|
{chr(10).join(x_rad_table)}

C K 的原始 0.660304 主要被光子能量约定拉低；同能量变为 0.955316。Ne 2s 的 2.194248 同能量变为 1.064426。Xe 的同能量 2s 仍为 0.795615，说明不能用改频率掩盖重元素径向模型差别。这里 XATOM 光子能量取打印的 0.01 eV 精度，不能据控制比值推断更高精度的归一化。

## 5. 归档 Auger、产额与预期失败的边界

引用报告明确 Auger 和 ωK 未随新 Gu 辐射场重算，因此本轮也没有把新辐射宽度塞进旧 Auger 产额。这些旧格的误差不能从本轮 PI/荧光闭合推断为已解决。较新的 Fe 五锚点见 [main 集成报告](../main_integration_2026_10_02/README.md)：其 `fac_frozen` 全 n=3 CI 仍最大偏离 11.43%，10% 门槛未过。

该 Fe 参考还混合 2s 空穴与 n=4 伙伴，atomx 面板的 CI 流形只含 n=3；参考脚本关闭 Breit，所以不能用 Breit 为余差作解释。缺失 CI 空间、单体 core dressing、近简并态配对与轨道松弛仍需控制实验，不能按现有证据给每项分配确定百分比。`active_result` 适配器只搬运轨道，不是 Hamiltonian 松弛循环；[测试归因](../../tests/test_fe_lmm_mscheme_panel.py:124) 已写明这些边界。

## 复现与验证

所有诊断都在新目录运行；`raw/` 保存 FAC 脚本、分辨线表和 XATOM argv/输出。所有六元素 ns=180 Gu 中性态与 16 个有 E1 通道的空穴原生宽度重现原归档；PI 全 54 点相对原值在 10⁻¹¹ 内一致。本报告所有误差控制与原生数据分列。

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FAC_SFAC_PATH="$HOME/code/fac/bin/sfac"
D=docs/error_attribution_2026_10_02
.venv/bin/python "$D/controls.py"
.venv/bin/python "$D/controls.py" --fac-grid
.venv/bin/python "$D/controls.py" --fac-grid --fac-variant point
.venv/bin/python "$D/controls.py" --fac-grid --fac-variant radial6400
.venv/bin/python "$D/controls.py" --fac-grid --fac-variant radial12800 --elements C Ne
.venv/bin/python "$D/pi_native_controls.py" C Ne Ar Fe Kr Xe
.venv/bin/python "$D/pi_native_controls.py" --ns 260 C Ne Ar Fe
.venv/bin/python "$D/pi_native_controls.py" --ns 360 C Ne
.venv/bin/python "$D/fluorescence_radial_controls.py"
.venv/bin/python "$D/fluorescence_radial_controls.py" --ns 260
.venv/bin/python "$D/fac_transition_cut_controls.py"
.venv/bin/python "$D/xatom_grid_controls.py"
.venv/bin/python "$D/xatom_grid_controls.py" --dr 0.000625
.venv/bin/python "$D/build_report.py"
```

XATOM 二进制及运行库路径沿用前轮本机工具链，见 `xatom_grid_controls.py` 与 `bench_systematic_xray.environment()`。FAC 的十进制截断封装问题用负值默认哨兵规避。ns=260 Kr 中性态的失败保留在日志中，复现正常面板命令因此只列成功的 C/Ne/Ar/Fe。

本轮是归因复查；没有改生产代码或刷新原 FAC/XATOM 列。诊断脚本的数据完整性与数学分解检验记录在 [validation.json](validation.json)，源码和输入/输出哈希在 [identity.json](identity.json)。
"""
    (HERE / "README.md").write_text(text)
    old_manifest = json.loads((OLD / "result_identity.json").read_text())[
        "files_sha256"
    ]
    archive_unchanged = all(
        hashlib.sha256((OLD / name).read_bytes()).hexdigest() == digest
        for name, digest in old_manifest.items()
    )
    assert archive_unchanged
    validation = dict(
        pi_points=len(pi_rows),
        fluorescence_cases=len(fluorescence_rows),
        nonzero_gu_fluorescence_cases=sum(
            r["exchange"] == "gu" and bool(r["native_over_fac_complete"])
            for r in fluorescence_rows
        ),
        native_pi_reproduction_rtol=1e-11,
        native_fluorescence_reproduction_exact=all(
            float(r["native_over_archived"]) == 1
            for r in read("fluorescence_grid_ns180.csv")
        ),
        fac_pi_focused_points=len(focused),
        fac_pi_point_points=len(point),
        fac_pi_radial6400_points=len(radial),
        xatom_grid_points=len(xhalf),
        xatom_grid_quarter_points=len(xquarter),
        continuum_grid_max_fractional_change=max_shoot,
        fac_electron_grid_max_fractional_change=max_fac_e,
        xatom_final_grid_max_fractional_change=max_x_q,
        kr_ns260_neutral_converged=False,
        production_changed=False,
        original_archive_manifest_unchanged=archive_unchanged,
    )
    assert validation["pi_points"] == 54 and validation["fluorescence_cases"] == 36
    assert validation["native_fluorescence_reproduction_exact"]
    (HERE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (HERE / "validation.log").write_text(json.dumps(validation, indent=2) + "\n")
    inputs = [OLD / "pi.csv", OLD / "fluorescence.csv", REF / "references.json"]
    sources = [
        ROOT / "src/atomx/methods/dirac_fock.py",
        ROOT / "src/atomx/scattering/photoionization/subshell.py",
        ROOT / "src/atomx/properties/transitions/fluorescence.py",
        ROOT / "src/atomx/properties/transitions/multipole.py",
        Path("/home/zzhang/code/fac/faclib/transition.c"),
        Path("/home/zzhang/code/fac/faclib/radial.c"),
        Path("/home/zzhang/code/fac/faclib/interpolation.c"),
        Path("/home/zzhang/code/fac/sfac/sfac.c"),
        Path("/home/zzhang/code/fac/faclib/nucleus.c"),
        Path("/home/zzhang/code/xraypac/xatom/src/pcs.f90"),
        Path("/home/zzhang/code/xraypac/xatom/src/decay.f90"),
        Path("/home/zzhang/code/xraypac/xatom/src/functional_xc.f90"),
    ]
    files = (
        inputs
        + sources
        + [p for p in HERE.iterdir() if p.is_file() and p.name != "identity.json"]
    )
    identity = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (HERE / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
