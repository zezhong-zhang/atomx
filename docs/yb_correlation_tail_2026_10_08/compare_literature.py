"""Cross-model printed E2 comparison; never inject literature offsets into native energies."""
from pathlib import Path
import json
import hashlib
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import save,sha


def main():
    from atomx.constants import HARTREE_CM1
    out=ROOT/"literature_check"
    out.mkdir(exist_ok=False)
    # Factual values independently reread from the retained author PDF, Table I.
    printed={"6s_1/2":(-8288,1016),"5d_3/2":(-9003,-1027),
        "5d_5/2":(-8001,-1301),"6p_1/2":(-4344,335),"6p_3/2":(-3606,166)}
    save(out/"contract.json",dict(script_sha256=sha(__file__),
        source="Safronova and Safronova, PRA 79, 022512 (2009), Table I, printed E(2) and E(3)",
        DOI="https://doi.org/10.1103/PhysRevA.79.022512",
        author_copy="https://mariannasafronova.com/wp-content/uploads/2022/12/YbII-2009.pdf",
        locally_reread_PDF_sha256=sha(ROOT.parent/"yb_diagnosis_repair_2026_10_08/references/Safronova_2009_author_copy.pdf"),
        comparison="Same physical attachment-correlation quantity and closed-core target; radial/partial-wave settings differ and precise nuclear/constant identity is not established. Match native on-shell E2, not nonlinear Dyson shift.",
        no_accuracy_gate="Cross-model differences are diagnostics, not missing-correction or precision error bounds; no correction is added."))
    rows=[]
    for label,(E2,E3) in printed.items():
        row=dict(label=label,paper_printed_E2_cm1=E2,paper_printed_E3_cm1=E3,
            paper_printed_E3_excitation_difference_cm1=E3-printed["6s_1/2"][1])
        for case in ("coarse_l8_e200","fine_l6_e200"):
            r=json.loads((ROOT/"results"/case/label.replace("/","_")/"results.json").read_text())
            v=r["variants"][-1];on_shell=v["on_shell_Ha"]*HARTREE_CM1
            row[case]=dict(on_shell_cm1=on_shell,native_minus_paper_E2_cm1=on_shell-E2,
                scope=dict(lmax=v["lmax"],emax_Ha=v["emax_Ha"],base=r["base"]))
        rows.append(row)
    save(out/"results.json",dict(records=rows,
        paper_representation="70 order-9 B splines, radius 200 a0; E2 lmax8 with higher-l extrapolation; E3 lmax6. This is not the native ns123/230, R60, finite unextrapolated virtual space.",
        numerical_pass_claim=False,corrections_injected=False))
    print("PRINTED E2 DIAGNOSTIC RECORDED; NO FIT OR CORRECTION")


if __name__=="__main__":main()
