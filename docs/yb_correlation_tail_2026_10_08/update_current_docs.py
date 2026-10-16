"""Promote the completed tail evidence without rewriting sealed stage records."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
DOCS=ROOT.parent


def replace_once(path,old,new):
    text=path.read_text()
    if text.count(old)!=1:raise ValueError(f"Expected one current authority block in {path}")
    path.write_text(text.replace(old,new,1))


def main():
    summary=json.loads((ROOT/"report_summary.json").read_text())
    data=json.loads((ROOT/"analysis/results.json").read_text())
    c=data["primaries"]["coarse_l8_e200"];f=data["primaries"]["fine_l6_e200"]
    radial=summary["radial200_max_increment_cm1"];uv=summary["fine_last_UV_increment_cm1"]
    angular=summary["last_angular_increment_cm1"]
    p=DOCS/"README.md"
    old=next(line for line in p.read_text().splitlines() if line.startswith("- [Yb⁺ 能量自洽与相关尾部]"))
    new=f"- [Yb⁺ 相关尾部与径向表示](yb_correlation_tail_2026_10_08/README.md)：当前诊断；两组完整五态（粗基 ℓ≤8/200 Ha、细基 ℓ≤6/200 Ha）及共同空间控制，最后分波/细基正能增量 {angular:.2f}/{uv:.2f} cm⁻¹，共同 ℓ6/200 的径向响应 {radial:.2f} cm⁻¹；数值门通过，相关收敛与 5% 实验目标仍未通过。\n- [Yb⁺ 能量自洽与相关尾部](yb_energy_dependent_sigma_2026_10_08/README.md)：上一阶段封存证据；离壳自能、DF 连续分支 Dyson 根与 CI 注入，137 项测试通过，能量反馈和空间收敛分别检验。"
    replace_once(p,old,new)
    p=DOCS/"AGENT_REFERENCE.md"
    old=next(line for line in p.read_text().splitlines() if line.startswith("| 稀土自由离子谱线"))
    new=f"| 稀土自由离子谱线、寿命与实验对照 | [当前 Yb⁺ 完整尾部与径向控制](yb_correlation_tail_2026_10_08/README.md)：粗 ℓ8/200、细 ℓ6/200 两组完整五态，共同表示控制；数值/身份门通过，空间及实验目标未通过。[能量依赖阶段](yb_energy_dependent_sigma_2026_10_08/README.md)保留离壳极点、DF 分支 Dyson 及 137 项测试；[在壳与约定阶段](yb_diagnosis_repair_2026_10_08/README.md)保留独立 DF/偶极/电流/Breit 与 QED 告警；[原始基线](rare_earth_yb_spectrum_2026_10_08/README.md)保留七谱线、寿命及失败 | 新专题 |"
    replace_once(p,old,new)
    p=DOCS/"VERIFICATION.md"
    text=p.read_text();start=text.index("The [current 2026-10-08 Yb⁺ energy-dependence controls]");end=text.index("## Current live structure evidence",start)
    new=f'''The [current 2026-10-08 Yb⁺ complete tail/radial controls](yb_correlation_tail_2026_10_08/README.md)
retain full five-state coarse l8/200-Ha and refined l6/200-Ha captures,
exact shared-space subsets, complete virtual metric/count/identity checks,
DF-connected scalar Dyson roots and native CI injection. Numerical gates pass.
The last coarse l7→l8 excitation increment is {angular:.3f} cm⁻¹, the refined
100→200-Ha increment is {uv:.3f} cm⁻¹, and the shared l6/200-Ha radial response
is {radial:.3f} cm⁻¹. Finite increments are not infinite-tail error bounds.
Maximum experimental deviations are {100*c['max_experimental_relative_error']:.3f}% and
{100*f['max_experimental_relative_error']:.3f}%; the 1-cm⁻¹ spatial and four-level 5% targets
remain failed. Production source and affected test/dependency hashes match the
[preceding 137-test validated stage](yb_energy_dependent_sigma_2026_10_08/README.md);
unaffected unit tests were not repeated in this measurement-only increment.
Frozen-orbital scalar Dyson is not off-diagonal Brueckner response, self-consistent
GF2 or a correlated-rate gate. The [earlier convention controls](yb_diagnosis_repair_2026_10_08/README.md)
preserve independent DF/dipole/current/Breit checks and analytic-QED warnings.
These dated controls do not refresh the older structure panel below.

'''
    p.write_text(text[:start]+new+text[end:])
    p=DOCS/"ARCHITECTURE_CAPABILITIES.md"
    replace_once(p,"  [current Yb energy-dependence controls](yb_energy_dependent_sigma_2026_10_08/README.md)\n  and [previous on-shell/convention controls](yb_diagnosis_repair_2026_10_08/README.md).",
        "  [current Yb complete tail/radial controls](yb_correlation_tail_2026_10_08/README.md),\n  [energy-dependence evidence](yb_energy_dependent_sigma_2026_10_08/README.md)\n  and [previous on-shell/convention controls](yb_diagnosis_repair_2026_10_08/README.md).")
    p=DOCS/"TESTING.md"
    old="`slow`, `external_fac`, `snapshot` and `book` markers are defined in"
    new="The [current Yb tail/radial campaign](yb_correlation_tail_2026_10_08/README.md)\nuses complete native five-state captures, metric/coverage checks and CI injection\nwith predeclared resource caps. It adds live convergence evidence; its numerical\npasses do not replace the separate spatial or experimental gates. The production\nsource and preceding 137-test panel are unchanged, so those unaffected unit tests\nwere not repeated. Replay commands and retained negative results live in the topic.\n\n"+old
    replace_once(p,old,new)
    p=DOCS/"AGENT_INSIGHTS.md"
    old="- 证据：同一有限Fermi/frozen DF-DKB全修正H，15→10、10→5Ha的四个跃迁"
    new=f"- 补充证据：[Yb⁺ 完整相关尾部与径向表示](yb_correlation_tail_2026_10_08/README.md)：\n  扫能量窗时同时核对实际虚谱及完整上下态响应，避免把有限表示容量当作尾界。\n  本例共同 ℓ6/50 Ha 的粗/细虚轨道为 861/1616，但共同 ℓ6/200 Ha 的实际径向\n  跃迁响应最高 {radial:.3f} cm⁻¹；最后分波增量仍 {angular:.3f} cm⁻¹，细基最后正能\n  增量仍 {uv:.3f} cm⁻¹。容量增加不等于误差增加；有限局部增量也不认证无限尾部。\n  并入现有规则，不新增根规则。\n"+old
    replace_once(p,old,new)
    trace=DOCS.parent/"trace.md"
    with trace.open("a") as stream:
        stream.write(f'''\n## 2026-10-08 Yb complete tail/radial controls (EARS)

Expectation: match all five states in both references; retain all 19 core radial orbitals, both native diagrams, complete declared virtual spaces and fixed DF-connected Dyson branch. Preserve 1 cm-1 spatial / 5% experimental gates, source and prior sealed evidence.
Action: coarse ns123 l8/200 and refined ns230 l6/200 captures; exact l/energy subsets, all-state differences/splittings, full native virtual Gram/count/identity checks, matched old on-shell/Dyson replay, and both native CI injections. Production physics code unchanged.
Results: numerical/coverage/source gates pass. Last coarse angular increment {angular:.3f}, refined UV increment {uv:.3f}, shared l6/200 radial increment {radial:.3f} cm-1. Experiment maxima {100*c['max_experimental_relative_error']:.3f}%/{100*f['max_experimental_relative_error']:.3f}%; spatial/experimental goals remain failed. Earlier 93/52 sealed hashes preserved. Source/tests match preceding 137-test panel; unaffected tests not repeated. Diagnostic harness failures and candidate-estimate typo retained and corrected without model/tolerance changes.
Significance/limits: judge finite virtual capacity through complete transition response, not root counts alone; no infinite-tail, full orbital SCF, high-order or dressed-rate certificate. Current authority and existing transition-feedback insight updated; old evidence preserved, no root-rule change. Evidence: docs/yb_correlation_tail_2026_10_08/README.md.
''')
    print("CURRENT DOCS AND EARS UPDATED")


if __name__=="__main__":main()
