"""One fixed-space preflight for bounded EXP-T process thread counts."""

import json
import resource
from pathlib import Path

import benchmark as bench
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    output = root / "thread_preflight001"
    output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (bench.CAP, bench.CAP))
    from atomx.methods.fscc import load_integrals

    audit = bench.load(
        bench.OLD / "parallel_six_fscc_scope001/audit.py", "thread_preflight_audit"
    )
    reader, _ = audit.helpers()
    template = root / "run001/accuracy_four_core_corrected"
    a = load_integrals(template)
    before = {str(p): bench.sha(p) for p in (Path(__file__), root / "benchmark.py")}
    contract = {
        "source_sha256": before,
        "hypothesis": "Runtime OpenMP total-thread and active-level limits bound actual process threads while preserving complete signed CCSDT tensors.",
        "parent_contract_sha256": bench.sha(root.parent / "contract.json"),
        "fixed_input_sha256": {
            name: bench.sha(template / name)
            for name in ("ATOMX_ONE", "ATOMX_TWO", "integrals.json")
        },
        "old_failed_resource_admission": "../run001/stopped_thread_budget.json",
        "policy": "1/2/4/8 requested threads, each OMP_THREAD_LIMIT=p and OMP_MAX_ACTIVE_LEVELS=1; process sampler kills owned EXP-T child if observed Threads>p.",
        "gate": "Actual Threads<=p<=8 and every coordinate of all six signed tensors and full HEFF agrees <=1e-11 Ha.",
        "stop": "One solve per p; stop remaining sweep immediately on cap/equation failure. No method, source or budget change.",
    }
    bench.dump(output / "contract.json", contract)
    baseline = bench.complete_arrays(reader, template, 12)
    rows = []
    with bench.timing_lock():
        for workers in (1, 2, 4, 8):
            directory = output / f"p{workers}"
            try:
                sample = bench.backend(
                    a, template, bench.CANDIDATE, directory, workers, "bare"
                )
                arrays = bench.complete_arrays(reader, directory, 12)
                defect = max(
                    float(np.max(abs(arrays[k] - baseline[k]))) for k in arrays
                )
                rows.append(
                    dict(
                        sample,
                        full_signed_arrays_defect_Ha=defect,
                        source_unchanged=all(
                            bench.sha(p) == s for p, s in before.items()
                        ),
                        gate_pass=(
                            defect <= 1e-11
                            and sample["actual_runtime"]["observed_backend_max_threads"]
                            <= workers
                            and sample["raw_equation_bound_Ha"] <= 1e-11
                        ),
                    )
                )
                print(
                    workers,
                    rows[-1]["actual_runtime"]["observed_backend_max_threads"],
                    rows[-1]["gate_pass"],
                    flush=True,
                )
                if not rows[-1]["gate_pass"]:
                    break
            except Exception as exc:
                rows.append(dict(workers=workers, gate_pass=False, error=repr(exc)))
                break
    bench.dump(
        output / "result.json",
        {
            "rows": rows,
            "gate_pass": len(rows) == 4 and all(r["gate_pass"] for r in rows),
            "highest_level_reached": False,
            "performance_claim": None,
        },
    )
    print(
        json.dumps({"gate_pass": len(rows) == 4 and all(r["gate_pass"] for r in rows)})
    )


if __name__ == "__main__":
    main()
