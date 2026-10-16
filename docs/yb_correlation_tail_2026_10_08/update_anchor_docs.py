"""Register the independent threshold anchor without changing earlier experiments."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
DOCS=ROOT.parent


def main():
    anchor=json.loads((ROOT/"absolute_anchor/results.json").read_text())
    rows=anchor["cases"]["fine_l6_e200"]
    p=DOCS/"README.md";text=p.read_text()
    old="数值门通过，相关收敛与 5% 实验目标仍未通过。"
    assert text.count(old)==1
    p.write_text(text.replace(old,"数值门通过，相关收敛与 5% 实验目标仍未通过；独立 ASD 电离阈值锚分辨基态和上态的绝对偏差。",1))
    p=DOCS/"AGENT_REFERENCE.md";text=p.read_text()
    old="粗 ℓ8/200、细 ℓ6/200 两组完整五态，共同表示控制；数值/身份门通过"
    assert text.count(old)==1
    p.write_text(text.replace(old,"粗 ℓ8/200、细 ℓ6/200 两组完整五态，共同表示控制及 ASD 绝对能量锚；数值/身份门通过",1))
    p=DOCS/"VERIFICATION.md";text=p.read_text()
    old="Frozen-orbital scalar Dyson is not off-diagonal Brueckner response, self-consistent\nGF2 or a correlated-rate gate."
    assert text.count(old)==1
    extra=f"The independently checked ASD 5.12 ionization threshold adds an absolute\nattachment comparison: refined ground residual {rows[0]['absolute_residual_cm1']:.2f} cm⁻¹,\nupper-state residuals {rows[1]['absolute_residual_cm1']:.2f}, {rows[2]['absolute_residual_cm1']:.2f},\n{rows[3]['absolute_residual_cm1']:.2f}, {rows[4]['absolute_residual_cm1']:.2f} cm⁻¹. These pure-DC versus\nphysical-experiment residuals do not isolate missing correlation or QED. Legacy\nHandbook/paper thresholds differ and are not mixed into the new absolute zero.\n"
    p.write_text(text.replace(old,extra+old,1))
    p=DOCS/"AGENT_INSIGHTS.md";text=p.read_text()
    old="- 去向：根 §3 的先核约定规则；具体系数保留在专题参考。"
    assert text.count(old)==1
    extra="- 补充证据：[Yb⁺ 独立绝对阈值锚](yb_correlation_tail_2026_10_08/README.md)：\n  定位上下态绝对偏差时另核对电离阈值、离子终态和数据版本，不能因都标作 NIST\n  就混用 Handbook、论文和当前 ASD 的零点。旧激发能及纯增量对照保持，新的\n  绝对锚单独冻结；本次区分基态与上态系统偏差，但不把模型−实验差直接解释\n  为单个遗漏物理项。并入现有约定规则，不新增根规则。\n"
    p.write_text(text.replace(old,extra+old,1))
    with (DOCS.parent/"trace.md").open("a") as stream:
        stream.write(f"\n### Yb absolute-threshold supplement (EARS, 2026-10-08)\n\nExpectation: distinguish lower/upper errors using a matched physical ionized-core zero without fitting or modifying existing transitions. Action: verify ASD 5.12 Yb II ionization 12.179185(25) eV and all four retained level values; convert with recorded Atomx units and compare both model attachments. Results: refined residuals {[round(r['absolute_residual_cm1'],3) for r in rows]} cm-1; absolute/relative closure {anchor['maximum_closure_error_cm1']:.3e} cm-1. Handbook 98207 and printed 2009 ground binding 98269 differ from current ASD {anchor['ionization_cm1']:.5f}; no old evidence rewritten. Significance: discrepancy involves ground and upper states; threshold versions cannot be mixed, and pure-DC versus measured-state errors do not isolate omitted correlation/Breit/QED. Current authority and convention-first insight updated; no new root rule.\n")
    print("ABSOLUTE ANCHOR KNOWLEDGE UPDATED")


if __name__=="__main__":main()
