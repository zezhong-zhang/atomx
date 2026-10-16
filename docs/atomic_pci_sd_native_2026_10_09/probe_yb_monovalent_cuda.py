"""Source-bound GPU execution of the existing complete Yb diagnostic driver.

The base driver owns input reconstruction and actual complete core replay.
This explicit private execution wrapper adds exact terminal GEMMs and CUDA
term2 before native load. Original complete CPU map arrays gate every column
before any root; each GPU root then passes an actual full CPU raw/energy check.
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

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(OWNER)]

import monovalent_davidson_control as numerical  # noqa: E402
from cuda_valence_bridge import BoundCUDAValenceControl  # noqa: E402
from cuda_valence_bridge import install as install_cuda  # noqa: E402
from valence_term41_blas_control import install as install_term41  # noqa: E402
from valence_terma4_blas_control import install as install_terma4  # noqa: E402
from valence_terma4_blas_control import scratch_bytes  # noqa: E402

from atomx.methods.core_sd import CoreSD  # noqa: E402
from atomx.methods.reduced_monovalent_sd import ReducedMonovalentSD  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out, cpu = OWNER / args.run_name, OWNER / args.cpu_reference
    if out.exists():
        raise ValueError("new complete GPU run directory required")
    base = runpy.run_path(str(OWNER / "probe_yb_full_monovalent.py"))
    helpers = [
        Path(__file__),
        OWNER / "probe_yb_full_monovalent.py",
        OWNER / "monovalent_term31_blas_control.py",
        OWNER / "monovalent_davidson_control.py",
        OWNER / "cuda_valence_bridge.py",
        OWNER / "cuda_full_term2_control.py",
        OWNER / "cuda_density_control.py",
        OWNER / "valence_term41_blas_control.py",
        OWNER / "valence_terma4_blas_control.py",
        OWNER / "probe_monovalent_cuda.py",
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in helpers}
    proof = OWNER / "monovalent_cuda002"
    record = json.loads((proof / "result.json").read_text())
    finite_hashes = json.loads((proof / "source_sha256.json").read_text())
    if record["status"] != "pass" or record["independent_gates"] != 164:
        raise ValueError(
            "current complete finite CUDA/GEMM independent164 gate required"
        )
    for path in helpers[2:]:
        relative = str(path.relative_to(ROOT))
        if finite_hashes[relative] != hashes[relative]:
            raise ValueError(
                "actual CUDA/GEMM/numerical helper differs from finite proof"
            )
    original_core, original_problem, original_solve = (
        CoreSD.__init__,
        ReducedMonovalentSD.__init__,
        numerical.solve,
    )
    controls, comparisons_done = {}, False

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    def certify_backend(directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, sha in manifest["files"].items():
            if Path(name).name != name or digest(directory / name) != sha:
                raise ValueError(
                    "actual source-bound native artifact differs before specialization"
                )
        return manifest

    def construct_core(self, *arguments, **kwargs):
        resource_record = json.loads((out / "resource.json").read_text())
        directory = Path(kwargs["backend_directory"])
        manifest = certify_backend(directory)
        dimensions = manifest["dimensions"]
        extra = scratch_bytes(dimensions) + (
            16 * dimensions["NXX"] ** 2 * manifest["valence_sd"]["nvh"]
        )
        bound = resource_record["maximum_bound_bytes"] + extra
        if bound > kwargs["memory_bytes"]:
            raise MemoryError(
                "combined complete GPU/GEMM host bound exceeds original cap before load"
            )
        save(
            "gpu_launch_contract.json",
            dict(
                hypothesis="Complete standalone CUDA/GEMM action and feedback roots retain original CPU full-space equations",
                base_driver_contract="contract.json records the unchanged physical reconstruction, core replay and numerical continuation; this wrapper declares the additional explicit execution policy",
                physical_space="Original68core/414radialvirtual, all direct-rho coordinates, real DC, same Fermi/ns123/l<=3/5Ha; no pruning or fitted shift",
                cpu_reference=args.cpu_reference,
                finite_proof="monovalent_cuda002:164 independent gates, allM/four orbital variants and16paired roots",
                finite_proof_sha256=digest(proof / "result.json"),
                execution_policy="Explicit CUDA term2, exact CPU term31/term41/terma4 GEMM, private directed complete-image Davidson; no automatic CPU fallback",
                signed_map_gate_Ha=1e-11,
                actual_cpu_root_raw_and_energy_gate_Ha=1e-10,
                host_cap_bytes=kwargs["memory_bytes"],
                device_cap_bytes=2 * 1024**3,
                whole_host_bound_bytes=bound,
                private_additional_host_bytes=extra,
                load=os.getloadavg(),
                timing_scope="current live-load observations only; no alternating controlled end-to-end speedup",
                stop="Any complete CPU map/root/source/cap gate failure rejects; no gate or space changes",
                scope="Private complete numerical equations and CPU/CUDA arithmetic, no public adoption, physical state/property/precision or pole certificate",
            ),
        )
        save("gpu_source_sha256.json", hashes)
        for path in helpers:
            target = out / "gpu_source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        install_term41(directory)
        certify_backend(directory)
        install_terma4(directory)
        certify_backend(directory)
        install_cuda(directory)
        certify_backend(directory)
        return original_core(self, *arguments, **kwargs)

    def construct_problem(self, *arguments, **kwargs):
        original_problem(self, *arguments, **kwargs)
        control = BoundCUDAValenceControl(
            self.valence, out / "cuda", memory_bytes=2 * 1024**3
        )
        control.select("cuda")
        controls[id(self)] = control

    def certify_complete_maps():
        nonlocal comparisons_done
        if comparisons_done:
            return
        original_hashes = json.loads((cpu / "source_sha256.json").read_text())
        if any(digest(ROOT / path) != sha for path, sha in original_hashes.items()):
            raise ValueError("complete original CPU map source identity changed")
        original_contract = json.loads((cpu / "contract.json").read_text())
        current_contract = json.loads((out / "contract.json").read_text())
        if original_contract["input_sha256"] != current_contract["input_sha256"]:
            raise ValueError("complete original CPU input identity differs")
        replay = json.loads((cpu / "core_replay.json").read_text())
        original_maps = json.loads((cpu / "map_controls.json").read_text())
        if replay["residual_max_Ha"] > 1e-10 or len(original_maps) != 5:
            raise ValueError(
                "complete original current-source core/five-map gates required"
            )
        rows = []
        for key in ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3)):
            filename = f"maps_{key[0]}_{key[1]}.npz"
            with (
                np.load(cpu / filename, allow_pickle=False) as a,
                np.load(out / filename, allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files):
                    raise ValueError(
                        "complete original/current map array coverage differs"
                    )
                comparisons = {}
                for name in a.files:
                    x, y = a[name], b[name]
                    if (
                        x.shape != y.shape
                        or not np.all(np.isfinite(x))
                        or not np.all(np.isfinite(y))
                    ):
                        raise ValueError(
                            "complete paired map array shape/nonfinite gate failed"
                        )
                    error = float(np.max(abs(x - y)))
                    if error > 1e-11:
                        raise AssertionError(
                            "complete original CPU/CUDA map differs: "
                            + str(key)
                            + "/"
                            + name
                        )
                    comparisons[name] = error
            rows.append(
                dict(
                    key=list(key),
                    arrays=comparisons,
                    original_sha256=digest(cpu / filename),
                    cuda_sha256=digest(out / filename),
                )
            )
        save(
            "complete_cpu_cuda_maps.json",
            dict(
                status="pass",
                columns=rows,
                source_match=True,
                scope="Complete original CPU versus CUDA/GEMM signed maps; no independent full-H physical oracle",
            ),
        )
        comparisons_done = True

    def solve_gpu(problem, key, **kwargs):
        certify_complete_maps()
        result = original_solve(problem, key, **kwargs)
        control = controls[id(problem)]
        control.select("cpu")
        with problem.core._lock:
            prepared = problem._prepare(key)
            energy, rhs = problem._evaluate(result.amplitudes, prepared, linear=False)
            raw = float(np.max(abs(rhs - result.denominators_Ha * result.amplitudes)))
            defect = abs(energy - result.correlation_energy_Ha)
        if not np.isfinite(max(raw, defect)) or max(raw, defect) > 1e-10:
            raise AssertionError(
                "actual complete CPU raw/energy recheck of GPU root failed"
            )
        save(
            f"cpu_root_recheck_{key[0]}_{key[1]}.json",
            dict(
                status="pass",
                raw_residual_Ha=raw,
                energy_residual_Ha=defect,
                actual_gpu_calls=control.calls,
                epochs=control.epoch,
                scope="Actual complete CPU term2 and exact GEMM map at every coordinate of the freshly solved GPU root",
            ),
        )
        control.select("cuda")
        return replace(
            result,
            provenance=dict(
                result.provenance,
                execution_backend="explicit private CUDA term2 + exact CPU term31/41/terma4 GEMM",
                execution_wrapper_source_sha256=digest(__file__),
                cpu_root_audit="separate per-column cpu_root_recheck files; actual complete1e-10Ha raw/energy required",
            ),
        )

    base_args = argparse.Namespace(
        run_name=args.run_name,
        plan_only=False,
        maps_only=False,
        term31_blas=True,
        davidson=True,
        all_roots=True,
    )
    try:
        with (
            patch.object(CoreSD, "__init__", construct_core),
            patch.object(ReducedMonovalentSD, "__init__", construct_problem),
            patch.object(numerical, "solve", solve_gpu),
        ):
            base["main"](base_args)
        changed = [path for path, sha in hashes.items() if digest(ROOT / path) != sha]
        if changed:
            raise ValueError("actual GPU execution source changed during full run")
        save(
            "gpu_source_postcheck.json",
            dict(status="pass", files=len(hashes), all_actual_cpu_root_audits=True),
        )
        save(
            "gpu_result.json",
            dict(
                status="pass",
                base_result_sha256=digest(out / "result.json"),
                complete_cpu_map_comparison_sha256=digest(
                    out / "complete_cpu_cuda_maps.json"
                ),
                five_actual_cpu_root_rechecks=True,
                source_postcheck="pass",
                highest_level_reached=False,
            ),
        )
    except BaseException as error:
        if out.exists():
            save(
                "gpu_failure.json",
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    traceback=traceback.format_exc(),
                ),
            )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--cpu-reference", default="yb_full_monovalent004")
    main(parser.parse_args())
