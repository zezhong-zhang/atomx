"""Aggregate measured Atomx results and independent references; no spectral fitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRIMARY = "v68_box90_newton"
ORDER = ["6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"]
DISPLAY = {"6s_1/2":"6s ²S₁/₂", "5d_3/2":"5d ²D₃/₂", "5d_5/2":"5d ²D₅/₂", "6p_1/2":"6p ²P°₁/₂", "6p_3/2":"6p ²P°₃/₂"}


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def csv_file(stem, rows):
    with (ROOT / f"{stem}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model(label):
    root = ROOT / "results" / label
    return read(root/"structure.json"), read(root/"lines.json") if (root/"lines.json").exists() else []


def main():
    reference = read(ROOT/"experimental_reference.json")
    structure, lines = model(PRIMARY)
    assert read(ROOT/"results"/PRIMARY/"status.json")["passed"]
    levels = {x["label"]:x for x in structure["levels"]}
    by_index = {x["index"]:x["label"] for x in levels.values()}
    refs = {x["label"]:x for x in reference["levels"]}
    level_rows = []
    for label in ORDER:
        a,b = levels[label]["excitation_cm1"], refs[label]["excitation_cm1"]
        assert (levels[label]["two_J"],levels[label]["parity"]) == (refs[label]["two_J"],refs[label]["parity"])
        level_rows.append({"state":label,"calc_excitation_cm1":a,"experimental_excitation_cm1":b,
            "calc_minus_experiment_cm1":a-b,"relative_error_percent":100*(a-b)/b if b else None,
            "experimental_uncertainty_cm1":None,"reference_url":refs[label]["source_url"]})
    line_map = {(by_index[x["initial_state_index"]],by_index[x["final_state_index"]],x["multipole"]):x for x in lines}
    line_rows = []
    for ref in reference["lines"]:
        key = ref["lower"],ref["upper"],ref["multipole"]
        x = line_map[key]
        observed = ref["observed_air_record"]
        calc,exp = x["wavelength_vacuum_nm"],ref["reference_Ritz_vacuum_nm"]
        line_rows.append({"lower":key[0],"upper":key[1],"multipole":key[2],
            "calc_wavenumber_cm1":x["wavenumber_cm1"],"reference_wavenumber_cm1":ref["reference_wavenumber_cm1"],
            "calc_vacuum_nm":calc,"reference_Ritz_vacuum_nm":exp,"vacuum_wavelength_error_percent":100*(calc/exp-1),
            "calc_A_length_per_s":x["A_rate_per_s"],"calc_A_velocity_per_s":x["A_velocity"],
            "calc_A_velocity_over_length":x["gauge_ratio"],"reduced_matrix_element_au":x["reduced_matrix_element_au"],
            "observed_air_nm":observed["observed_wavelength_air_nm"] if observed else None,
            "reference_type":ref["reference_type"],"reference_url":refs[key[1]]["source_url"]})
    lifetime_rows = []
    for ref in reference["lifetimes"]:
        state = ref["state"]
        outgoing = [x for x in lines if by_index[x["final_state_index"]]==state]
        rate = sum(x["A_rate_per_s"] for x in outgoing)
        unit,scale = ("ns",1e9) if "lifetime_ns" in ref else ("ms",1e3)
        actual = ref["lifetime_"+unit]
        lifetime_rows.append({"state":state,"unit":unit,"calc_lifetime":scale/rate,"experiment_lifetime":actual,
            "experiment_uncertainty":ref["uncertainty_"+unit],"relative_error_percent":100*(scale/rate/actual-1),
            "total_model_decay_rate_per_s":rate,"included_channels":";".join(by_index[x["initial_state_index"]]+":"+x["multipole"] for x in outgoing),
            "experimental_isotope":ref["isotope"],"calculation_nuclear_radius_A_parameter":175,
            "scope":"sum of included five-level E1/E2/M1 channels; not isotope-resolved or all physical channels",
            "reference_url":ref["source_url"]})
    splitting_rows=[]
    for prefix,lo,hi in [("5d","5d_3/2","5d_5/2"),("6p","6p_1/2","6p_3/2")]:
        calc=levels[hi]["excitation_cm1"]-levels[lo]["excitation_cm1"]
        exp=refs[hi]["excitation_cm1"]-refs[lo]["excitation_cm1"]
        splitting_rows.append({"term":prefix,"calc_splitting_cm1":calc,"reference_splitting_cm1":exp,"relative_error_percent":100*(calc/exp-1)})
    corrections=read(ROOT/"results/v68_default_corrections/factors.json")
    holes=read(ROOT/"results/v68_default_4f_holes/factors.json")
    qed=read(ROOT/"results/v68_default_qed_models/factors.json")
    correction_map={x["label"]:x for x in corrections["records"]}
    hole_map={x["label"]:x for x in holes["records"]}
    factor_rows=[{"state":k,"Breit_excitation_shift_cm1":correction_map[k]["excitation_shift_cm1"],
        "default_model_QED_excitation_shift_cm1":correction_map[k]["QED_excitation_shift_cm1"],
        "4f_hole_CI_excitation_shift_cm1":hole_map[k]["excitation_shift_cm1"],
        "4f_hole_baseline_overlap_squared":hole_map[k]["reference_overlap_squared"]} for k in ORDER]
    radial=[]
    for before,after,scope in [("v68_default","v68_refined_newton","grid plus native eigenpair refinement"),
                               ("v68_refined_newton",PRIMARY,"R60 to R90 at same fine grid and native refinement")]:
        a,la=model(before);b,lb=model(after)
        ea={x["label"]:x["excitation_cm1"] for x in a["levels"]}
        eb={x["label"]:x["excitation_cm1"] for x in b["levels"]}
        def rate_map(s,l):
            identity={x["index"]:x["label"] for x in s["levels"]}
            return {(identity[x["initial_state_index"]],identity[x["final_state_index"]],x["multipole"]):x["A_rate_per_s"] for x in l}
        ra,rb=rate_map(a,la),rate_map(b,lb)
        radial.append({"before":before,"after":after,"scope":scope,
            "max_excitation_drift_cm1":max(abs(eb[k]-ea[k]) for k in ORDER),
            "excitation_drift_cm1":{k:eb[k]-ea[k] for k in ORDER},
            "max_selected_A_relative_drift":max(abs(rb[(x["lower"],x["upper"],x["multipole"])]/ra[(x["lower"],x["upper"],x["multipole"])]-1) for x in reference["lines"]),
            "predeclared_excitation_gate_cm1":1.,"passed":max(abs(eb[k]-ea[k]) for k in ORDER)<1.})
    for x in lines:
        a=levels[by_index[x["initial_state_index"]]];b=levels[by_index[x["final_state_index"]]]
        assert b["energy_hartree"]>a["energy_hartree"]
        assert abs(x["wavenumber_cm1"]-(b["excitation_cm1"]-a["excitation_cm1"]))<1e-6
        assert abs(x["wavelength_vacuum_nm"]*x["wavenumber_cm1"]/1e7-1)<1e-7
        assert np.isfinite(x["A_rate_per_s"]) and x["A_rate_per_s"]>=0
        assert a["parity"]*b["parity"]==(-1 if x["multipole"]=="E1" else 1)
    assert len(line_rows)==7 and len(lifetime_rows)==2 and len(level_rows)==5
    diag=structure["solver_diagnostics"]
    assert diag["core_DF_converged"] and diag["core_DF_residual"] < 1e-8
    assert max(diag["projected_valence_residuals_Ha"].values()) < 1e-7
    assert max(abs(x) for x in diag["Hamiltonian_consistency_errors_Ha"]) < 1e-8
    assert structure["orbital_metric_max_error"] < 1e-8
    for scope in ("corrections", "4f_holes", "qed_models"):
        assert read(ROOT/"results"/("v68_default_"+scope)/"status.json")["passed"]
    source = read(ROOT/"results"/PRIMARY/"source_identity.json")
    mismatches=[k for k,v in source["source_sha256"].items() if hashlib.sha256((REPO/k).read_bytes()).hexdigest()!=v]
    assert not mismatches, mismatches
    run_checks={}
    for directory in sorted((ROOT/"results").iterdir()):
        identity=directory/"source_identity.json"
        if identity.exists():
            original=read(identity)
            run_checks[directory.name]={"same_Atomx_source_map_as_primary":original["source_map_sha256"]==source["source_map_sha256"],
                "status":read(directory/"status.json") if (directory/"status.json").exists() else "failure retained"}
    assert all(x["same_Atomx_source_map_as_primary"] for x in run_checks.values())
    verification={"date":"2026-10-08","primary":PRIMARY,"reference_level_state_match":True,
        "positive_transition_energy_and_nonnegative_finite_rates":True,"transition_energy_and_vacuum_identity":True,
        "multipole_parity":True,"table_counts":{"levels":5,"selected_lines":7,"direct_lifetimes":2,"all_native_lines":len(lines)},
        "numerical_model_gates_passed":True,"radial_controls_passed":all(x["passed"] for x in radial),
        "measured_primary_gates":{"core_DF_residual":diag["core_DF_residual"],
            "max_projected_valence_residual_Ha":max(diag["projected_valence_residuals_Ha"].values()),
            "max_Hamiltonian_consistency_error_Ha":max(abs(x) for x in diag["Hamiltonian_consistency_errors_Ha"]),
            "orbital_metric_max_error":structure["orbital_metric_max_error"]},
        "experimental_excitation_goal_percent":5.,"experimental_excitation_goal_passed":all(abs(x["relative_error_percent"])<=5 for x in level_rows if x["state"]!="6s_1/2"),
        "current_Atomx_source_hashes_match_run":True,"run_dispositions":run_checks,
        "scope":"replayed result identities, declared solver gates and radial controls; no full repository regression suite or experimental accuracy certification"}
    data={"primary":PRIMARY,"model":"Dirac-Coulomb frozen self-consistent [Xe]4f14 V68 core + one valence electron",
        "levels":level_rows,"lines":line_rows,"lifetimes":lifetime_rows,"fine_structure":splitting_rows,
        "fixed_orbital_factors":factor_rows,"QED_model_sensitivity":qed["records"],"radial_controls":radial,"verification":verification}
    save(ROOT/"comparison.json",data);save(ROOT/"verification.json",verification)
    for stem,rows in [("levels_comparison",level_rows),("lines_comparison",line_rows),("lifetimes_comparison",lifetime_rows),
                      ("fine_structure_comparison",splitting_rows),("factor_comparison",factor_rows),("qed_model_sensitivity",qed["records"])]:csv_file(stem,rows)
    assets=ROOT/"assets";assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    labels=["5d 2D3/2","5d 2D5/2","6p 2P1/2","6p 2P3/2"]
    fig,(ax,err)=plt.subplots(1,2,figsize=(10,4),gridspec_kw={"width_ratios":[1.4,1]},layout="constrained")
    xs=np.arange(4);rows=level_rows[1:]
    ax.bar(xs-.18,[x["experimental_excitation_cm1"] for x in rows],.36,color="#a5b9c7",label="NIST experimental levels")
    ax.bar(xs+.18,[x["calc_excitation_cm1"] for x in rows],.36,color="#266685",label="Atomx DC frozen-core")
    ax.set(xticks=xs,xticklabels=labels,ylabel="Excitation energy / cm^-1",ylim=(0,34500));ax.tick_params(axis="x",labelrotation=25);ax.legend(fontsize=8)
    err.barh(labels,[x["relative_error_percent"] for x in rows],color=["#bd6b40" if x["relative_error_percent"]>0 else "#266685" for x in rows]);err.axvline(0,color="#777",lw=.8);err.set_xlabel("Energy error / %")
    fig.suptitle("Yb II: numerical convergence does not remove model bias")
    fig.savefig(assets/"levels_comparison.png",dpi=180);fig.savefig(assets/"levels_comparison.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    for i,x in enumerate(line_rows):
        exp,calc=x["reference_Ritz_vacuum_nm"],x["calc_vacuum_nm"]
        ax.plot([exp,calc],[i,i],color="#becbd5",lw=2)
        ax.scatter(exp,i,color="#5b7887",s=45,marker="o",label="Reference Ritz (vacuum)" if i==0 else None)
        ax.scatter(calc,i,color="#bb6638",s=50,marker="x",label="Atomx DC (vacuum)" if i==0 else None)
    ax.set(xscale="log",xlabel="Vacuum wavelength / nm (log scale)",yticks=range(7),yticklabels=[x["lower"].replace("_"," ")+" -> "+x["upper"].replace("_"," ")+" ("+x["multipole"]+")" for x in line_rows])
    ax.invert_yaxis();ax.legend(fontsize=8);ax.set_title("Selected line positions; marker heights are labels, not intensities")
    fig.savefig(assets/"lines_comparison.png",dpi=180);fig.savefig(assets/"lines_comparison.pdf");plt.close(fig)
    scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    scripts.update({"source_snapshots/"+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"source_snapshots").glob("*.py")})
    save(ROOT/"provenance_notes.json",{"Atomx_source_map_sha256":source["source_map_sha256"],"script_hashes_at_aggregation":scripts,
        "initial_V68_driver":"source_snapshots/run_core_valence_before_newton.py",
        "refined_resumed_driver":"run_core_valence.py",
        "snapshot_field_correction":"Per-run source_identity.script_sha256 hashes the imported run_yb.py snapshot helper, not every calling script. Library hashes were complete and unchanged. This record supplies actual script identity; original per-run files are retained unchanged.",
        "initial_EOl_persistence_failure":"Initial dc_default produced a mappingproxy pickle failure before structured output; no spectrum is claimed from it. Replay and subsequent save failures retained; NPZ data preserved.",
        "timing_scope":"uncontrolled concurrent machine load; elapsed times are execution records, not performance comparisons"})
    print(json.dumps({"primary":PRIMARY,"levels":level_rows,"lifetimes":lifetime_rows,"radial":radial},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
