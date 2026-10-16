"""Source-bound numerical64 continuation of the complete finite CUDA gate.

Only numerical subspace capacity changes. The unchanged driver owns all
four physical variants, independent all-M equations and paired CPU/GPU roots.
Its original numeric16 description is supplemented by the explicit launch
contract; additional full vectors are guarded before build/load, not hidden
in physical-space plans.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(OWNER)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out = OWNER / args.run_name
    launch = OWNER / (args.run_name + "_execution_contract.json")
    if out.exists() or launch.exists():
        raise ValueError("new numeric64 control identity required")
    driver = OWNER / "probe_monovalent_cuda.py"
    helpers = [Path(__file__), driver, OWNER / "monovalent_davidson_control.py"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in helpers}
    launch.write_text(
        json.dumps(
            dict(
                hypothesis="A larger numerical subspace preserves complete standalone equations and all CPU/CUDA/independent gates",
                physical_and_operator_policy="Unchanged finite driver with exact term31/41/terma4, all four complete orbital variants and all ten M; no physical pruning or tolerance changes",
                base_contract_scope="The original driver's physical/core/GPU protocol is retained; its fixed numeric16 description is superseded only by this explicit numeric64 execution contract",
                numerical_subspace=64,
                full_vector_families=2,
                max_actions=160,
                coupled_stages=8,
                finite_solver_target_Ha=1e-14,
                independent_gates_Ha=1e-11,
                host_cap_bytes=256 * 1024**2,
                device_cap_bytes=2 * 1024**3,
                source_sha256=hashes,
                load=os.getloadavg(),
                stop="Any source/budget/independent/paired/root failure rejects; no silent fallback or relaxed gate",
                scope="Private finite numerical policy, no large-root performance or experimental certification",
            ),
            indent=2,
        )
        + "\n"
    )
    namespace = runpy.run_path(str(driver))["main"].__globals__
    original_solve, original_build = namespace["solve"], namespace["build_monovalent"]
    records = []

    def wide_solve(problem, key, **kwargs):
        result = original_solve(problem, key, subspace=64, max_actions=160, **kwargs)
        count = problem.plan["amplitudes"][tuple(key)]
        if result.provenance["resources"]["subspace"] != min(count, 64):
            raise AssertionError("actual numerical subspace policy differs")
        records.append(
            dict(
                key=list(key),
                actual_subspace=min(count, 64),
                matrix_vector_products=result.matrix_vector_products,
                restarts=result.provenance["restarts"],
                residual_max_Ha=result.residual_max_Ha,
            )
        )
        return result

    def guarded_build(reference, directory, *arguments, **kwargs):
        variant = Path(directory).parent.name
        baseline = json.loads((out / (variant + "_resource.json")).read_text())
        count = max(baseline["amplitude_counts"].values())
        # Conservative simultaneous numerical-vector/projection increase.
        extra = 8 * (96 * count + 8 * (65**2 - 17**2))
        bound = baseline["complete_payload_bound_bytes"] + extra
        record = dict(
            status="pass" if bound <= baseline["host_cap_bytes"] else "fail",
            whole_host_bound_bytes=bound,
            additional_numerical_bytes=extra,
            before_native_build_or_load=True,
            baseline_resource_sha256=digest(out / (variant + "_resource.json")),
        )
        (out / (variant + "_numeric64_resource.json")).write_text(
            json.dumps(record, indent=2) + "\n"
        )
        if record["status"] != "pass":
            raise MemoryError("combined numerical64 full payload exceeds original cap")
        return original_build(reference, directory, *arguments, **kwargs)

    namespace["solve"], namespace["build_monovalent"] = wide_solve, guarded_build
    try:
        namespace["main"](
            argparse.Namespace(
                run_name=args.run_name, term41_blas=True, terma4_blas=True
            )
        )
        if any(digest(ROOT / path) != sha for path, sha in hashes.items()):
            raise ValueError("actual numeric64 control source changed")
        result = json.loads((out / "result.json").read_text())
        if result["status"] != "pass" or result["independent_gates"] != 164:
            raise AssertionError("all independent finite164 gates required")
        baseline = OWNER / "monovalent_cuda002"
        baseline_result = json.loads((baseline / "result.json").read_text())
        for directory, summary in ((out, result), (baseline, baseline_result)):
            if digest(directory / "complete_arrays.npz") != summary["arrays_sha256"]:
                raise ValueError("actual complete scientific payload checksum differs")
        differences = {}
        with (
            np.load(out / "complete_arrays.npz", allow_pickle=False) as new,
            np.load(baseline / "complete_arrays.npz", allow_pickle=False) as old,
        ):
            if set(new.files) != set(old.files):
                raise ValueError("numeric16/64 scientific array coverage differs")
            for name in old.files:
                a, b = old[name], new[name]
                if a.shape != b.shape or not np.all(np.isfinite(b)):
                    raise ValueError(
                        "numeric64 scientific array shape/nonfinite failure"
                    )
                error = float(np.max(abs(a - b)))
                if error > 1e-11:
                    raise AssertionError(
                        "numeric16/64 complete scientific array differs"
                    )
                differences[name] = error
        shutil.copy2(launch, out / "numeric64_execution_contract.json")
        for path in helpers:
            target = out / "numeric64_source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (out / "numeric64_result.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    source_match=True,
                    independent_gates=164,
                    records=records,
                    numeric16_scientific_array_errors=differences,
                    baseline_result_sha256=digest(
                        OWNER / "monovalent_cuda002/result.json"
                    ),
                    base_result_sha256=digest(out / "result.json"),
                    scope="Private numerical64 finite policy only; full-Yb numeric64 roots and controlled timing unverified",
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(dict(status="pass", numerical_subspace=64, gates=164)))
    except BaseException as error:
        destination = out if out.exists() else OWNER
        (
            destination
            / (
                "numeric64_failure.json"
                if out.exists()
                else args.run_name + "_failure.json"
            )
        ).write_text(
            json.dumps(
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    traceback=traceback.format_exc(),
                ),
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    main(parser.parse_args())
