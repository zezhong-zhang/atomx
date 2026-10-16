"""Matched five-state angular, energy and radial controls, with native CI injection."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "yb_diagnosis_repair_2026_10_08"
DYSON = ROOT.parent / "yb_energy_dependent_sigma_2026_10_08"
sys.path.insert(0, str(PRIOR))
from common import KEYS, EXPERIMENT, load_reference, save, sha, source_identity

LABELS = ["6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"]
CASES = {"coarse_l8_e200": "v68_default", "fine_l6_e200": "v68_refined_newton"}


def read_case(case):
    from atomx.constants import HARTREE_CM1
    rows = {}
    for label in LABELS:
        folder = ROOT / "results" / case / label.replace("/", "_")
        status = json.loads((folder / "status.json").read_text())
        if not status["completed"] or not status["numerical_gates_passed"]:
            raise ValueError(f"Incomplete native capture: {case} {label}")
        row = json.loads((folder / "results.json").read_text())
        if not row["source_unchanged"]:
            raise ValueError("Capture source identity changed")
        rows[label] = row
    variants = {}
    for first in rows[LABELS[0]]["variants"]:
        key = (first["lmax"], first["emax_Ha"])
        rr = []
        for label in LABELS:
            row = rows[label]
            v = next(x for x in row["variants"] if (x["lmax"], x["emax_Ha"]) == key)
            eps = row["epsilon_DF_Ha"]
            rr.append(dict(label=label, epsilon_DF_Ha=eps,
                on_shell_shift_cm1=v["on_shell_Ha"] * HARTREE_CM1,
                on_shell_attachment_Ha=eps + v["on_shell_Ha"],
                dyson_shift_cm1=v["dyson"]["shift_Ha"] * HARTREE_CM1,
                dyson_attachment_Ha=v["dyson"]["energy_Ha"],
                residue=v["dyson"]["residue"], root_residual_Ha=v["dyson"]["residual_Ha"],
                retained_nonzero_terms=v["retained_nonzero_terms"],
                old_on_shell_error_Ha=v["matched_old_on_shell_error_Ha"]))
        ground = rr[0]
        for row in rr:
            for method in ("on_shell", "dyson"):
                row[f"excitation_{method}_cm1"] = (
                    row[f"{method}_attachment_Ha"] - ground[f"{method}_attachment_Ha"]) * HARTREE_CM1
            row["experimental_excitation_cm1"] = EXPERIMENT[row["label"]]
            row["experiment_residual_cm1"] = row["excitation_dyson_cm1"] - EXPERIMENT[row["label"]]
            row["relative_experimental_error"] = (
                abs(row["experiment_residual_cm1"]) / EXPERIMENT[row["label"]]
                if EXPERIMENT[row["label"]] else None)
        by_label = {x["label"]: x for x in rr}
        variants[key] = dict(case=case, base=CASES[case], lmax=key[0], emax_Ha=key[1], records=rr,
            D_splitting_cm1=by_label["5d_5/2"]["excitation_dyson_cm1"]-by_label["5d_3/2"]["excitation_dyson_cm1"],
            P_splitting_cm1=by_label["6p_3/2"]["excitation_dyson_cm1"]-by_label["6p_1/2"]["excitation_dyson_cm1"],
            max_experimental_relative_error=max(x["relative_experimental_error"] for x in rr[1:]))
    return rows, variants


def difference(old, new, kind):
    from atomx.constants import HARTREE_CM1
    rr = []
    for a, b in zip(old["records"], new["records"], strict=True):
        if a["label"] != b["label"]:
            raise ValueError("State identity mismatch")
        rr.append(dict(label=a["label"],
            individual_attachment_change_cm1=(b["dyson_attachment_Ha"]-a["dyson_attachment_Ha"])*HARTREE_CM1,
            correlation_shift_change_cm1=b["dyson_shift_cm1"]-a["dyson_shift_cm1"],
            DF_attachment_change_cm1=(b["epsilon_DF_Ha"]-a["epsilon_DF_Ha"])*HARTREE_CM1,
            excitation_change_cm1=b["excitation_dyson_cm1"]-a["excitation_dyson_cm1"],
            on_shell_excitation_change_cm1=b["excitation_on_shell_cm1"]-a["excitation_on_shell_cm1"]))
    return dict(kind=kind,
        old={k:old[k] for k in ("case","lmax","emax_Ha")},
        new={k:new[k] for k in ("case","lmax","emax_Ha")}, records=rr,
        max_absolute_excitation_change_cm1=max(abs(x["excitation_change_cm1"]) for x in rr[1:]),
        D_splitting_change_cm1=new["D_splitting_cm1"]-old["D_splitting_cm1"],
        P_splitting_change_cm1=new["P_splitting_cm1"]-old["P_splitting_cm1"])


def inject(case, primary):
    from atomx.methods.ci_mbpt import ci_mbpt_solve
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.constants import HARTREE_CM1
    core, r, levels, identity = load_reference(CASES[case])
    sigma1 = {KEYS[row["label"]]:row["dyson_shift_cm1"]/HARTREE_CM1 for row in primary["records"]}
    kwargs = dict(csfs=r.csfs, orbitals_dict=r.orbitals_dict, orbital_basis=sorted(r.orbitals_dict),
        bsp_data=r.bsp_data, Z=70, ns=r.bsp_data["ns"], nuclear_potential=r._nuclear_potential)
    ci = ci_mbpt_solve(**kwargs, sigma1=sigma1)
    H = build_ci_matrix(**kwargs, one_body_shifts=sigma1)
    residuals = np.linalg.norm(H @ ci.mixing_coefficients - ci.mixing_coefficients * ci.state_energies, axis=0)
    overlaps = abs(r.mixing_coefficients.T @ ci.mixing_coefficients)**2
    matches = np.argmax(overlaps, axis=1)
    weights = overlaps[np.arange(r.n_states), matches]
    expected = [lv["energy_hartree"]+sigma1[KEYS[lv["label"]]] for lv in levels]
    errors = [ci.state_energies[j]-expected[i] for i,j in enumerate(matches)]
    passed = max(residuals)<1e-8 and max(abs(np.array(errors)))<1e-8 and min(weights)>=.99 and len(set(matches))==5
    if not passed:
        raise ValueError(f"Native CI injection failed: {case}")
    return dict(case=case, passed=passed, identity=identity, residuals_Ha=residuals,
        matching_squared_overlap=weights, matched_energy_error_Ha=errors,
        corrected_energies_Ha=ci.state_energies, baseline_to_new_indices=matches,
        scope="Matched frozen five-state reference plus diagonal attachment shifts; energy bookkeeping only, not a correlated or radiatively dressed wavefunction")


def main(coarse_only=False):
    out = ROOT / ("coarse_preliminary" if coarse_only else "analysis")
    out.mkdir(exist_ok=False)
    save(out / "contract.json", dict(primary_contract_sha256=sha(ROOT/"contract.json"),
        script_sha256=sha(__file__),cap_seconds=120,coarse_only=coarse_only,
        state_order=LABELS,source_sha256=source_identity(),
        gate_policy="All five states matched within each radial/reference/angular/energy model; predeclared 1 cm-1 spatial increment and 5% experimental gates retained. No extrapolation or fitted correction."))
    started = time.monotonic()
    chosen = ["coarse_l8_e200"] if coarse_only else list(CASES)
    if not coarse_only:
        capture = json.loads((ROOT/"capture_status.json").read_text())
        if not capture["completed"] or not capture["source_unchanged"]:
            raise ValueError("Full capture gate failed")
    cases = {}
    rows = {}
    for case in chosen:
        rows[case], cases[case] = read_case(case)
    coarse = cases["coarse_l8_e200"]
    controls = []
    for e in (5.,20.,50.,100.,200.):
        for lo,hi in ((4,6),(6,7),(7,8),(6,8)):
            controls.append(difference(coarse[lo,e],coarse[hi,e],"angular"))
    for case, variants in cases.items():
        for l in sorted(set(k[0] for k in variants)):
            for lo,hi in ((5.,20.),(20.,50.),(50.,100.),(100.,200.),(50.,200.)):
                controls.append(difference(variants[l,lo],variants[l,hi],"energy_ceiling"))
    if not coarse_only:
        fine = cases["fine_l6_e200"]
        for e in (5.,20.,50.,100.,200.):
            for l in (4,6):
                controls.append(difference(coarse[l,e],fine[l,e],"radial_reference"))
            controls.append(difference(fine[4,e],fine[6,e],"angular"))
    previous = json.loads((DYSON/"replay_and_ci/subsets.json").read_text())
    matched = []
    for l in (4,6):
        for e in (5.,20.,50.):
            old = next(x for x in previous if x["lmax"]==l and x["emax_Ha"]==e)
            for row in coarse[l,e]["records"]:
                prior = next(x for x in old["records"] if x["label"]==row["label"])
                delta = row["dyson_attachment_Ha"]-prior["dyson_energy_Ha"]
                if abs(delta)>1e-10:
                    raise ValueError("Matched previous Dyson replay failed")
                matched.append(dict(label=row["label"],lmax=l,emax_Ha=e,error_Ha=delta))
    primary = {case:cases[case][(8 if case.startswith("coarse") else 6),200.] for case in chosen}
    ci = [] if coarse_only else [inject(case,primary[case]) for case in chosen]
    source_unchanged = source_identity()==json.loads((ROOT/"capture_source_sha256.json").read_text())
    if not source_unchanged:
        raise ValueError("Source changed before analysis")
    save(out/"results.json",dict(preliminary=coarse_only,primaries=primary,
        variants={case:list(v.values()) for case,v in cases.items()},controls=controls,
        matched_previous_Dyson=matched,ci_injection=ci,
        gates=dict(numerical_passed=True,source_unchanged=source_unchanged,
            max_Dyson_replay_error_Ha=max(abs(x["error_Ha"]) for x in matched),
            correlation_increment_cm1=1.,experimental_relative_error=.05,
            primaries_experimental_passed={case:r["max_experimental_relative_error"]<.05 for case,r in primary.items()}),
        measurement_scope="Pure Dirac-Coulomb frozen-orbital diagonal Sigma2 Dyson roots; finite representation controls, not infinite-tail error bars"))
    elapsed = time.monotonic()-started
    if elapsed>120:
        raise TimeoutError("Declared analysis cap exceeded")
    save(out/"status.json",dict(completed=True,elapsed_seconds=elapsed,source_unchanged=source_unchanged))
    for case, p in primary.items():
        print(case, "max_experimental_relative_error", p["max_experimental_relative_error"], flush=True)
    print("ANALYSIS PASS",elapsed,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-only",action="store_true")
    main(parser.parse_args().coarse_only)
