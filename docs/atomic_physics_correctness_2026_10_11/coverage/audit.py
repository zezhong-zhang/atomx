"""Read-only coverage audit of saved evidence; never launch atomic solvers."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FILES = {
    "production_core": "src/atomx/methods/core_sd.py",
    "production_valence": "src/atomx/methods/reduced_monovalent_sd.py",
    "native_diagrams": "src/atomx/methods/_pci_monovalent.f90",
    "finite_monovalent": "src/atomx/methods/monovalent_sd.py",
    "finite_linearized": "src/atomx/methods/linearized_sd.py",
    "transverse_bridge": "src/atomx/methods/sd_transverse.py",
    "pair_subset": "src/atomx/methods/sd_pair_diagrams.py",
    "fscc_consumer": "src/atomx/methods/fscc.py",
    "fscc_builder": "tools/build_expt_backend.py",
    "fscc_patch": "tools/expt_noncanonical_patch.py",
    "fscc_properties": "src/atomx/properties/transitions/fscc_effective.py",
    "yb_driver": "docs/atomic_pci_sd_native_2026_10_09/probe_yb_full_monovalent.py",
    "yb_cuda_driver": "docs/atomic_pci_sd_native_2026_10_09/probe_yb_monovalent_cuda.py",
    "yb_comparison_driver": "docs/atomic_pci_sd_native_2026_10_09/compare_yb_monovalent_experiment.py",
    "yb_comparison": "docs/atomic_pci_sd_native_2026_10_09/yb_monovalent_experiment003/comparison.json",
    "yb_roots": "docs/atomic_pci_sd_native_2026_10_09/yb_full_monovalent006/result.json",
    "finite_primary_sd": "docs/atomic_pci_sd_native_2026_10_09/monovalent_sd008/comparison.json",
    "finite_reduced_sd": "docs/atomic_pci_sd_native_2026_10_09/reduced_monovalent008/result.json",
    "yb_norm": "docs/atomic_pci_sd_native_2026_10_09/parallel_six_norm001/run002/result.json",
    "finite_fscc": "docs/atomic_external_benchmark_2026_10_11/fscc/run003/result.json",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if (HERE / "evidence.json").exists():
        previous = json.loads((HERE / "evidence.json").read_text())
        drift = []
        for key, record in previous["inputs"].items():
            path = record["path"]
            captured = (
                HERE / "source_epoch" / path
                if path.endswith((".py", ".f90"))
                else ROOT / path
            )
            assert sha(captured.read_bytes()) == record["sha256"], key
            if sha((ROOT / path).read_bytes()) != record["sha256"]:
                drift.append(path)
        print(
            json.dumps(
                {
                    "status": "frozen_evidence_verified",
                    "bound_files": len(previous["inputs"]),
                    "new_roots": 0,
                    "current_source_drift_from_captured_epoch": drift,
                }
            )
        )
        return
    # Freeze the consumer source epoch, since another authorized worker can
    # subsequently repair the public guard. Historical input evidence stays put.
    data = {key: (ROOT / path).read_bytes() for key, path in FILES.items()}
    bound = {
        key: {"path": path, "sha256": sha(data[key])} for key, path in FILES.items()
    }
    contract = {
        "hypothesis": "A converged finite orbital band can still have equation defects or omitted physical diagrams.",
        "fixed": "Saved full signed inputs, declared methods, states, Hamiltonians and original gates; no new root or timing run.",
        "distinguishing_outcomes": [
            "Independent retained-equation failure is an implementation defect.",
            "Passed retained equations with nonzero omitted-rank residual is a model truncation.",
            "Missing operator at the current consumer cannot be repaired by enlarging its Coulomb virtual band.",
        ],
        "acceptance": "Each coverage claim names its actual consumer and source-bound saved evidence, and respects model boundaries.",
        "stop": "Finish source tracing and arithmetic replay; do not perform larger-space scientific scans.",
        "inputs": bound,
        "scientific_solves_launched": 0,
    }
    (HERE / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    for key, path in FILES.items():
        if path.endswith((".py", ".f90")):
            snapshot = HERE / "source_epoch" / path
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data[key])
    comp = json.loads(data["yb_comparison"])
    roots = json.loads(data["yb_roots"])
    finite = json.loads(data["finite_primary_sd"])
    reduced = json.loads(data["finite_reduced_sd"])
    norm = json.loads(data["yb_norm"])
    fscc = json.loads(data["finite_fscc"])["accuracy"]
    assert len(comp["rows"]) == len(roots["roots"]) == 5
    assert comp["common_physical_and_solver_binding"]["hamiltonian"] == "Dirac-Coulomb"
    assert comp["relative_error_gate"] == 0.01
    assert finite["status"] == "pass-finite" and reduced["passed"]
    fscc_rows = []
    for case in ("two_core", "four_core"):
        for backend in ("default", "corrected"):
            v = fscc[case][backend]
            fscc_rows.append(
                {
                    "case": case,
                    "backend": backend,
                    "noccupied": v["noccupied"],
                    "nspinors": v["nspinors"],
                    "core_retained_residual_Ha": v["core_projected_rank123_max_Ha"],
                    "attachment_retained_residual_Ha": v[
                        "attachment_projected_rank012_max_Ha"
                    ],
                    "attachment_full_residual_Ha": v["full_attachment_bloch_max_Ha"],
                    "unretained_core_rank4_residual_Ha": v[
                        "core_unretained_rank4_max_Ha"
                    ],
                    "unretained_attachment_rank34_residual_Ha": v[
                        "attachment_unretained_rank34_max_Ha"
                    ],
                    "core_fci_method_error_Ha": v["core_fci_method_error_Ha"],
                    "attachment_fci_method_error_max_Ha": v[
                        "attachment_fci_method_error_max_Ha"
                    ],
                    "retained_attachment_gate": v["gates"]["attachment_projected"],
                    "fock_occupied_virtual_max_Ha": v["fock_occupied_virtual_max_Ha"],
                }
            )
    assert not fscc_rows[0]["retained_attachment_gate"]
    assert fscc_rows[1]["retained_attachment_gate"]
    assert not fscc_rows[2]["retained_attachment_gate"]
    assert fscc_rows[3]["retained_attachment_gate"]
    assert fscc_rows[3]["attachment_full_residual_Ha"] > 1e-3
    result = {
        "status": "source_and_saved_evidence_audit_complete",
        "inputs": bound,
        "current_yb_declared_omissions": comp["limitations"],
        "current_yb_state_scope": comp["scope"],
        "current_yb_raw_residual_max_Ha": max(
            x["raw_residual_Ha"] for x in roots["roots"]
        ),
        "current_yb_excitation_gate_pass": comp["excitation_gate_pass"],
        "current_yb_splitting_gate_pass": comp["splitting_gate_pass"],
        "finite_primary_equation_max_Ha": finite["maximum_errors"][
            "primary_equation_Ha"
        ],
        "finite_primary_full_H_residual_max_Ha": finite[
            "max_full_eigenstate_residual_Ha"
        ],
        "finite_reduced_cases": len(reduced["records"]),
        "finite_reduced_random_map_max_Ha": max(
            x["errors"]["random_map_Ha"] for x in reduced["records"]
        ),
        "finite_reduced_full_H_residual_max_Ha": max(
            x["full_eigenstate_residual_Ha"] for x in reduced["records"]
        ),
        "current_yb_norm_scope": norm["scope"],
        "finite_fscc_rows": fscc_rows,
        "highest_impact_demonstrated_implementation_gap": "Public 0h1p FSCC accepts the known-failing original noncanonical equation using only backend update convergence; no equation capability guard.",
        "highest_impact_current_yb_model_gap": "Linearized singles/doubles omit nonlinear products and connected triples, while one-reference core-hole treatment lacks balanced multi-configuration mixing. Which dominates the current spectrum remains unmeasured.",
        "first_fixed_space_discriminators": [
            "Reject uncertified original noncanonical 0h1p before launch; retain corrected retained-equation and omitted-rank gates separately.",
            "Same native frozen space: weak-coupling coefficient versus independently contracted two-class MBPT2 for core and each of the five attached states.",
            "Same complete finite H: linearized SD, nonlinear CCSD, corrected CCSDT and complete FCI; inspect signed residuals by rank and all roots, not only energies.",
            "Balanced 4f-hole model and complement with signed couplings/state weights before attributing near-degenerate mixing or adding a self-energy.",
        ],
        "scientific_solves_launched": 0,
        "production_modified": False,
        "cutoff_convergence_is_physical_completeness": False,
        "all_order_sd_is_all_excitations": False,
        "numerical_residual_is_spectrum_error_bound": False,
        "highest_level_reached": False,
    }
    unchanged = {
        key: sha((ROOT / path).read_bytes()) == bound[key]["sha256"]
        for key, path in FILES.items()
    }
    result["input_hashes_unchanged_during_audit"] = unchanged
    assert all(unchanged.values()), (
        "Concurrent source transition: repeat under a new source epoch"
    )
    (HERE / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "bound_files": len(bound),
                "new_roots": 0,
                "source_unchanged": all(unchanged.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
