"""Bounded, matched finite EA comparison; not a production atomic benchmark."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import tracemalloc

import numpy as np

from atomx.methods import eom_attachment


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
FILES = ["src/atomx/numerics/cluster_actions.py", "src/atomx/numerics/fermion.py",
         "src/atomx/methods/coupled_cluster.py", "src/atomx/methods/eom_attachment.py",
         "tests/test_eom_attachment.py", "tests/test_cc_state_action.py"]


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}


def compare(result, baseline):
    return {
        "core_energy_Ha": abs(result.core.total_energy_Ha - baseline.core.total_energy_Ha),
        "amplitudes": float(np.max(abs(result.core.amplitudes - baseline.core.amplitudes))),
        "projected_operator_Ha": float(np.max(abs(result.projected_operator_Ha - baseline.projected_operator_Ha))),
        "attachment_energies_Ha": float(np.max(abs(result.attachment_energies_Ha - baseline.attachment_energies_Ha))),
        "physical_projectors": max(float(np.max(abs(np.outer(a, a.conj()) - np.outer(b, b.conj()))))
                                   for a, b in zip(result.coefficients.T, baseline.coefficients.T, strict=True)),
        "full_residuals_Ha": float(np.max(abs(result.full_eigenstate_residuals_Ha - baseline.full_eigenstate_residuals_Ha))),
    }


def main():
    before = hashes()
    fixture = ROOT / "tests/test_eom_attachment.py"
    spec = importlib.util.spec_from_file_location("finite_ea_fixture", fixture)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    integrals, _, _ = mod.integrals(spinors=7)
    contract = dict(schema=1, scope="synthetic seven-spinor three-core-electron complete finite CCSDT/EA",
                    representations=["dense", "state_action"], seed=194025,
                    repetitions=5, order="alternating dense/state_action then state_action/dense",
                    acceptance=1e-10, residual_gate_Ha=1e-10,
                    omitted_rank4_defect="preserved, not required to vanish",
                    source_sha256=before, affinity=sorted(os.sched_getaffinity(0)),
                    thread_environment={k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
                    load_start=list(os.getloadavg()),
                    timing_scope="warm end-to-end finite core solve plus complete attachment solve; tracemalloc disabled",
                    memory_scope="one separate tracemalloc peak per representation; not RSS or production bound")
    (HERE / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    baseline = eom_attachment.compute(integrals)
    first_state = eom_attachment.compute(integrals, action_representation="state_action")
    controls = [compare(first_state, baseline)]
    timings = {name: [] for name in contract["representations"]}
    for repeat in range(5):
        order = contract["representations"] if repeat % 2 == 0 else list(reversed(contract["representations"]))
        for name in order:
            start = time.perf_counter()
            result = eom_attachment.compute(integrals, action_representation=name)
            timings[name].append(time.perf_counter() - start)
            controls.append(compare(result, baseline))
            if max(controls[-1].values()) > contract["acceptance"]:
                raise RuntimeError("matched complete finite physics gate failed")
    memory = {}
    for name in contract["representations"]:
        tracemalloc.start()
        result = eom_attachment.compute(integrals, action_representation=name)
        memory[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        controls.append(compare(result, baseline))
    if hashes() != before or max(max(c.values()) for c in controls) > contract["acceptance"]:
        raise RuntimeError("source or final complete finite physics gate failed")
    np.savez(HERE / "complete_results.npz", core_amplitudes=baseline.core.amplitudes,
             projected_operator_Ha=baseline.projected_operator_Ha,
             energies_Ha=baseline.attachment_energies_Ha, coefficients=baseline.coefficients,
             state_amplitudes=first_state.core.amplitudes,
             state_projected_operator_Ha=first_state.projected_operator_Ha,
             state_energies_Ha=first_state.attachment_energies_Ha, state_coefficients=first_state.coefficients,
             full_residuals_Ha=baseline.full_eigenstate_residuals_Ha,
             state_full_residuals_Ha=first_state.full_eigenstate_residuals_Ha)
    output = dict(status="passed", measured_seconds=timings,
                  medians_seconds={k: float(np.median(v)) for k, v in timings.items()},
                  tracemalloc_peak_bytes=memory, complete_comparison_errors=controls,
                  generator_bytes={"dense": baseline.provenance["generator_bytes"],
                                   "state_action": first_state.provenance["generator_bytes"]},
                  load_end=list(os.getloadavg()),
                  projected_residual_max_Ha=float(np.max(first_state.projected_residuals_Ha)),
                  full_residual_max_Ha=float(np.max(first_state.full_eigenstate_residuals_Ha)),
                  source_sha256=before)
    (HERE / "benchmark_result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("complete_comparison_errors", "source_sha256")}))


if __name__ == "__main__":
    main()
