"""Explicit numerical64 execution of the unchanged complete GPU Yb driver.

This source-bound wrapper changes only numerical basis capacity, after the
finite164 numeric64 and complete42-matrix stress gates. Original five CPU
maps and actual full CPU residual of each GPU root are still mandatory.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(OWNER)]

import monovalent_davidson_control as numerical  # noqa: E402

from atomx.methods.core_sd import CoreSD  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out = OWNER / args.run_name
    launch = OWNER / (args.run_name + "_numeric64_contract.json")
    if out.exists() or launch.exists():
        raise ValueError("new complete numerical64 run identity required")
    proof = OWNER / "monovalent_cuda004"
    record = json.loads((proof / "numeric64_result.json").read_text())
    frozen = json.loads((proof / "numeric64_execution_contract.json").read_text())
    if record["status"] != "pass" or record["independent_gates"] != 164:
        raise ValueError("complete finite numeric64 independent164 proof required")
    for path, sha in frozen["source_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError("current numerical64 helper differs from finite proof")
    stress_file = OWNER / "monovalent_wide_subspace001.json"
    stress = json.loads(stress_file.read_text())
    if len(stress["rows"]) != 3 or stress["rows"][-1]["actual_dimension"] <= 16:
        raise ValueError("independent wider actual-dimension numerical gate required")
    for name, sha in stress["source_sha256"].items():
        if digest(OWNER / name) != sha:
            raise ValueError("current numerical stress source differs")
    helpers = [Path(__file__), OWNER / "probe_yb_monovalent_cuda.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in helpers}
    contract = dict(
        hypothesis="Larger numerical capacity reduces restart cost while preserving original complete Yb equations and physical space",
        physical_policy="Unchanged68core/414radialvirtual/Fermi/ns123/l<=3/5Ha/DC and all five valence keys; no fitted shift or coordinate pruning",
        base_contract_scope="Unchanged input/core/map/CPU-CUDA/root protocol from base and GPU drivers; only their fixed numeric16 description is superseded by this explicit numerical64 policy",
        numerical_capacity=64,
        full_vector_families=2,
        max_actions=160,
        eight_original_couplings=True,
        actual_raw_and_energy_gate_Ha=1e-10,
        original_signed_map_gate_Ha=1e-11,
        host_cap_bytes=32 * 1024**3,
        device_cap_bytes=2 * 1024**3,
        finite_proof_sha256=digest(proof / "numeric64_result.json"),
        actual_wide_stress_sha256=digest(stress_file),
        source_sha256=hashes,
        load=os.getloadavg(),
        stop="Any budget/source/map/branch/raw/energy failure rejects; no fallback, changed gates or reduced physics",
        timing_scope="Live-load observations only; numerical action counts separate from controlled end-to-end speedup",
        scope="Private numerical64 full-space equations; no large physical states/properties, experimental convergence or pole certificate",
    )
    launch.write_text(json.dumps(contract, indent=2) + "\n")
    original_core, original_solve = CoreSD.__init__, numerical.solve

    def guarded_core(self, *arguments, **kwargs):
        base = json.loads((out / "resource.json").read_text())
        gpu = json.loads((out / "gpu_launch_contract.json").read_text())
        count = max(base["standalone_plan"]["amplitudes"].values())
        extra = 8 * (96 * count + 8 * (65**2 - 17**2))
        bound = gpu["whole_host_bound_bytes"] + extra
        resource = dict(
            status="pass" if bound <= kwargs["memory_bytes"] else "fail",
            maximum_bound_bytes=bound,
            additional_numerical_bytes=extra,
            before_actual_native_load=True,
            base_gpu_contract_sha256=digest(out / "gpu_launch_contract.json"),
        )
        (out / "numeric64_resource.json").write_text(
            json.dumps(resource, indent=2) + "\n"
        )
        if resource["status"] != "pass":
            raise MemoryError("combined numerical64 full-space host bound exceeds cap")
        shutil.copy2(launch, out / "numeric64_execution_contract.json")
        for path in helpers:
            target = out / "numeric64_source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return original_core(self, *arguments, **kwargs)

    def wide_solve(problem, key, **kwargs):
        result = original_solve(problem, key, subspace=64, max_actions=160, **kwargs)
        if result.provenance["resources"]["subspace"] != 64:
            raise AssertionError("actual complete numerical64 policy differs")
        return replace(
            result,
            provenance=dict(
                result.provenance,
                numerical_capacity_wrapper_source_sha256=digest(__file__),
                numerical_capacity_policy="Explicit64, original full coordinates/eight-stage path/raw/energy gates",
            ),
        )

    try:
        driver = runpy.run_path(str(OWNER / "probe_yb_monovalent_cuda.py"))
        with (
            patch.object(CoreSD, "__init__", guarded_core),
            patch.object(numerical, "solve", wide_solve),
        ):
            driver["main"](
                argparse.Namespace(
                    run_name=args.run_name, cpu_reference=args.cpu_reference
                )
            )
        if any(digest(ROOT / path) != sha for path, sha in hashes.items()):
            raise ValueError("actual numerical64 execution wrapper source changed")
        (out / "numeric64_result.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    source_match=True,
                    base_gpu_result_sha256=digest(out / "gpu_result.json"),
                    resource_sha256=digest(out / "numeric64_resource.json"),
                    highest_level_reached=False,
                ),
                indent=2,
            )
            + "\n"
        )
    except BaseException as error:
        destination = out if out.exists() else OWNER
        name = (
            "numeric64_failure.json"
            if out.exists()
            else args.run_name + "_failure.json"
        )
        (destination / name).write_text(
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
    parser.add_argument("--cpu-reference", default="yb_full_monovalent004")
    main(parser.parse_args())
