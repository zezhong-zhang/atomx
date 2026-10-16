"""Matched five-state arithmetic, native metric and archive/source gates."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import EXPERIMENT, KEYS, load_reference, save, source_identity
from atomx.constants import HARTREE_CM1
from atomx.numerics.radial_workspace import eval_orbital_pair, workspace_arrays
from atomx.methods.mbpt_matrix import MatrixSelfEnergy


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    status = json.loads((ROOT/"capture_status.json").read_text())
    assert status["completed"] and len(status["jobs"]) == 10
    captured = json.loads((ROOT/"capture_source_sha256.json").read_text())
    current = source_identity()
    if current != captured:
        chain = json.loads((ROOT/"final_source_chain.json").read_text())
        assert current == chain["final_production_sha256"] and captured == chain["capture_production_sha256"]
        assert chain["changed_after_capture"] == ["src/atomx/methods/mbpt_transitions.py"]
        assert chain["exact_source_diff_checked"] and chain["all_native_capture_input_conversions_are_identity"]
    contract = json.loads((ROOT/"contract.json").read_text())
    cases, rows, maxmetric, maxcore, maxresidual, maxreplay = {}, [], 0., 0., 0., 0.
    saved_residual, saved_Z_error = 0., 0.
    for case in contract["captures"]:
        core, _, _, _ = load_reference(case["reference"])
        _, w = workspace_arrays(core._bsp_data)
        state_rows = {label: json.loads((ROOT/"results"/case["name"]/label.replace("/", "_")/"results.json").read_text()) for label in KEYS}
        for label, row in state_rows.items():
            path = ROOT/"results"/case["name"]/label.replace("/", "_")/"matrix_poles.npz"
            assert sha(path) == row["matrix_poles_sha256"]
            with np.load(path) as z:
                model = MatrixSelfEnergy(z["eps_Ha"], z["poles_Ha"], z["residues_Ha2"], str(z["reference_identity"]))
            for variant in row["variants"]:
                n, root = variant["external_dimension"], variant["root"]
                energy, u = root["energy_Ha"], np.array(root["coefficients"])
                H = np.diag(model.reference_energies_Ha[:n])+model.evaluate(energy)[:n, :n]
                residual = float(np.linalg.norm(H@u-energy*u))
                Z = 1/(1-float(u@model.derivative(energy)[:n, :n]@u))
                saved_residual, saved_Z_error = max(saved_residual, residual), max(saved_Z_error, abs(Z-root["residue"]))
                assert residual <= 1e-10 and abs(Z-root["residue"]) <= 1e-10
            del model
        models = {}
        for n in (1, 2, 3):
            attached = {label: state_rows[label]["variants"][n-1]["root"]["energy_Ha"]*HARTREE_CM1 for label in KEYS}
            excitations = {label: attached[label]-attached["6s_1/2"] for label in KEYS}
            errors = {label: excitations[label]-EXPERIMENT[label] for label in KEYS if label != "6s_1/2"}
            pct = max(abs(errors[label]/EXPERIMENT[label])*100 for label in errors)
            models[str(n)] = dict(attachment_cm1=attached, excitation_cm1=excitations,
                                  experiment_residual_cm1=errors, max_abs_relative_percent=pct,
                                  experimental_5percent_passed=pct <= 5.,
                                  splitting_cm1=dict(d=excitations["5d_5/2"]-excitations["5d_3/2"],
                                                     p=excitations["6p_3/2"]-excitations["6p_1/2"]))
            for label in KEYS:
                row = state_rows[label]
                base = ROOT/"results"/case["name"]/label.replace("/", "_")
                mixed = np.load(base/f"mixed_orbital_P{n}.npz")
                p, q = eval_orbital_pair(mixed["P"], mixed["Q"], core._bsp_data)
                metric = abs(float(np.sum(w*(p*p+q*q)))-1.)
                overlap = 0.
                for pc, qc, _ in core.orbitals_raw.get(KEYS[label][1], []):
                    pc, qc = eval_orbital_pair(pc, qc, core._bsp_data)
                    overlap = max(overlap, abs(float(np.sum(w*(p*pc+q*qc)))))
                maxmetric, maxcore = max(maxmetric, metric), max(maxcore, overlap)
                root = row["variants"][n-1]["root"]
                maxresidual = max(maxresidual, root["residual_Ha"])
                maxreplay = max(maxreplay, abs(row["old_diagonal_root_replay_error_Ha"]), abs(row["old_onshell_replay_error_Ha"]))
                rows.append(dict(case=case["name"], P=n, label=label, attachment_cm1=attached[label],
                                 excitation_cm1=excitations[label], experiment_cm1=EXPERIMENT[label],
                                 experimental_error_cm1=excitations[label]-EXPERIMENT[label],
                                 orbital_df_weight=root["df_weight"], quasiparticle_residue=root["residue"],
                                 eigen_residual_Ha=root["residual_Ha"], metric_error=metric, core_overlap=overlap))
        response = {str(n): {label: models[str(n)]["excitation_cm1"][label]-models["1"]["excitation_cm1"][label] for label in KEYS} for n in (2, 3)}
        final_increment = {label: models["3"]["excitation_cm1"][label]-models["2"]["excitation_cm1"][label] for label in KEYS}
        cases[case["name"]] = dict(models=models, excitation_response_from_diagonal_cm1=response,
                                    attachment_response_from_diagonal_cm1={label: models["3"]["attachment_cm1"][label]-models["1"]["attachment_cm1"][label] for label in KEYS},
                                    P2_to_P3_excitation_cm1=final_increment,
                                    external_space_1cm1_passed=max(abs(v) for v in final_increment.values()) <= 1.,
                                    canonical_scalar_reference_change_cm1={label: row["canonical_scalar_reference_change_Ha"]*HARTREE_CM1 for label, row in state_rows.items()})
    assert maxmetric <= 1e-7 and maxcore <= 1e-7 and maxresidual <= 1e-10 and maxreplay <= 1e-10
    radial = {label: cases["fine_l6_e200"]["models"]["3"]["excitation_cm1"][label]-cases["coarse_l6_e200"]["models"]["3"]["excitation_cm1"][label] for label in KEYS}
    archives = {}
    for name in ("yb_diagnosis_repair_2026_10_08", "yb_energy_dependent_sigma_2026_10_08", "yb_correlation_tail_2026_10_08", "yb_literature_repair_strategy_2026_10_08"):
        manifest = json.loads((ROOT.parent/name/"artifact_manifest.json").read_text())
        for key, value in manifest.items():
            path = REPO/key if key.startswith("docs/") else ROOT.parent/name/key
            expected = value if isinstance(value, str) else value["sha256"]
            assert sha(path) == expected, path
        archives[name] = dict(count=len(manifest), unchanged=True)
    save(ROOT/"analysis.json", dict(cases=cases, radial_response_P3_excitation_cm1=radial,
                                    max_radial_excitation_response_cm1=max(abs(v) for v in radial.values()),
                                    maximum_matrix_eigen_residual_Ha=maxresidual,
                                    maximum_old_diagonal_replay_error_Ha=maxreplay,
                                    maximum_saved_matrix_vector_residual_Ha=saved_residual,
                                    maximum_saved_matrix_residue_replay_error=saved_Z_error,
                                    maximum_mixed_orbital_metric_error=maxmetric, maximum_core_overlap=maxcore,
                                    archive_checks=archives, source_unchanged_during_capture=True,
                                    final_source_matches_capture_exact=current == captured,
                                    final_source_preserves_native_capture_inputs=True,
                                    elapsed_capture_seconds=status["seconds"],
                                    peak_state_RSS_MiB=max(json.loads((ROOT/"results"/x["case"]/x["label"].replace("/", "_")/"results.json").read_text())["peak_process_RSS_MiB"] for x in status["jobs"])))
    with (ROOT/"comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v["models"]["3"] for k, v in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
