"""Resolve absolute attachment errors with the verified Yb II -> Yb III threshold."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import save,sha,EXPERIMENT,source_identity

IE_URL="https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=1&biblio=1&e_out=0&el_name_out=1&ion_charge_out=1&spectra=ytterbium&unc_out=1&units=1"
LEVEL_URL="https://physics.nist.gov/cgi-bin/ASD/energy1.pl?biblio=on&conf_out=on&de=0&format=0&j_out=on&lande_out=on&level_out=on&multiplet_ordered=0&output=0&page_size=15&perc_out=on&spectrum=Yb+II&submit=Retrieve+Data&temp=&term_out=on&unc_out=1&units=0"


def main():
    from atomx.constants import HARTREE_CM1,HARTREE_EV
    out=ROOT/"absolute_anchor"
    out.mkdir(exist_ok=False)
    save(out/"contract.json",dict(script_sha256=sha(__file__),date="2026-10-08",
        hypothesis="Experimental ionization threshold supplies an independent common energy zero; separate lower- and upper-state absolute errors without shifting any calculated or experimental excitation.",
        sources=dict(ionization=IE_URL,levels=LEVEL_URL,version="ASD 5.12"),
        observed_ionization_row=dict(Z=70,ion_charge=1,IE_eV=12.179185,uncertainty_eV=.000025,reference="L12907"),
        ionized_state="Yb III [Xe]4f14 1S0, same 68-electron closed-core threshold identity",
        levels_match_retained_reference=EXPERIMENT,source_sha256=source_identity(),
        scope="Pure-DC frozen Sigma2 Dyson attachment energy versus physical measured ionization/levels; absolute residuals are not an isolated correlation or QED error budget.",
        closure_gate_cm1=1e-8,fit_or_correction=False))
    factor=HARTREE_CM1/HARTREE_EV
    I=12.179185*factor;unc=.000025*factor
    analysis=json.loads((ROOT/"analysis/results.json").read_text())
    cases={}
    for case,primary in analysis["primaries"].items():
        rr=[]
        for row in primary["records"]:
            expt=EXPERIMENT[row["label"]]-I
            native=row["dyson_attachment_Ha"]*HARTREE_CM1
            rr.append(dict(label=row["label"],experimental_attachment_cm1=expt,
                computed_attachment_cm1=native,absolute_residual_cm1=native-expt,
                retained_excitation_residual_cm1=row["experiment_residual_cm1"]))
        for row in rr:
            err=(row["absolute_residual_cm1"]-rr[0]["absolute_residual_cm1"])-row["retained_excitation_residual_cm1"]
            if abs(err)>1e-8:raise ValueError("Absolute/relative energy-zero closure failed")
            row["closure_error_cm1"]=err
        cases[case]=rr
    save(out/"results.json",dict(ionization_energy_eV=12.179185,ionization_cm1=I,
        quoted_ionization_uncertainty_cm1=unc,eV_to_cm1=factor,cases=cases,
        source_zero_discrepancies=dict(handbook_limit_cm1=98207,paper_2009_NIST_ground_binding_cm1=98269,
            handbook_minus_current_ASD_cm1=98207-I,paper_minus_current_ASD_cm1=98269-I),
        maximum_closure_error_cm1=max(abs(r["closure_error_cm1"]) for rr in cases.values() for r in rr),
        corrections_added=False,reference_excitation_values_changed=False,
        warning="ASD ionization uncertainty is a common threshold uncertainty; missing level uncertainties, isotope/hyperfine/nuclear identity and non-DC operators still preclude a precision theory error budget."))
    print("ABSOLUTE ANCHOR PASS; IE cm-1",I,"uncertainty",unc)
    for row in cases["fine_l6_e200"]:print(row["label"],row["absolute_residual_cm1"])


if __name__=="__main__":main()
