"""Source-bound full radial Yb linearized SD physical norm control.

Requires the complete five-root source/GPU/CPU/numerical64 gates first. It
freshly replays the complete CoreSD seed and measures bitwise seed identity,
then contracts every real core and valence radial rho channel. No full SD
eigenstate, transition strength or pole certificate follows from this norm.
"""

import hashlib
import json
import os
import resource as process_resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(OWNER)]

from monovalent_norm_control import norm_components  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def plain(value):
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    out, run = OWNER / "yb_monovalent_norm001", OWNER / "yb_full_monovalent006"
    out.mkdir(exist_ok=False)
    start = time.perf_counter()

    def save(name, value):
        (out / name).write_text(json.dumps(plain(value), indent=2) + "\n")

    def observe(**values):
        row = dict(
            values,
            elapsed_seconds=time.perf_counter() - start,
            load=os.getloadavg(),
            peak_rss_bytes=process_resource.getrusage(
                process_resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024,
        )
        with (out / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(plain(row)) + "\n")
        print(json.dumps(plain(row)), flush=True)
        if row["elapsed_seconds"] > 7200:
            raise RuntimeError("complete physical norm wall budget exhausted")

    save(
        "contract.json",
        dict(
            hypothesis="Every retained complete real radial core/valence rho contributes to source-bound (1+Tcore+Sv) physical wave norm with exact angular/fermion interference and Pauli blocking",
            computed_run=run.name,
            keys=[[6, -1], [6, 1], [6, -2], [5, 2], [5, -3]],
            original_physics="Z70/V68/Fermi/ns123/l<=3/414radial virtuals below5Ha/DC; no physical cutoff",
            core_gate_Ha=1e-10,
            orthonormal_metric_gate=1e-10,
            core_seed_gate="actual fresh CoreSD solution must equal full frozen seed bitwise; no reconstructed or inferred current amplitudes",
            norm_M_gate=1e-11,
            cap_bytes=32 * 1024**3,
            extra_norm_bytes=64 * 1024**2,
            wall_cap_seconds=7200,
            stop="Any finite prerequisites/current-source/core-seed/complete coordinate/resource/norm/completeM covariance gate failure rejects",
            scope="Private complete physical norm; no production state/property, spectral intensity, Hamiltonian eigenstate, correlation convergence, experimental precision or resonance claim",
        ),
    )
    try:
        summary = json.loads((run / "result.json").read_text())
        if len(summary["roots"]) != 5 or len(summary["controls"]) != 5:
            raise ValueError("all five complete roots/maps required")
        for name in (
            "source_postcheck.json",
            "gpu_result.json",
            "gpu_source_postcheck.json",
            "numeric64_result.json",
        ):
            if json.loads((run / name).read_text())["status"] != "pass":
                raise ValueError("complete source/execution gate required")
        if any(run.glob("*failure.json")):
            raise ValueError("failed root execution cannot bind physical norm")
        files = json.loads((run / "source_sha256.json").read_text())
        numeric_contract = json.loads(
            (run / "numeric64_execution_contract.json").read_text()
        )
        numeric_result = json.loads((run / "numeric64_result.json").read_text())
        if numeric_result["base_gpu_result_sha256"] != digest(
            run / "gpu_result.json"
        ) or numeric_result["resource_sha256"] != digest(
            run / "numeric64_resource.json"
        ):
            raise ValueError("numerical64 certified execution/resource payload differs")
        for bound in (
            json.loads((run / "gpu_source_sha256.json").read_text()),
            numeric_contract["source_sha256"],
        ):
            for name, sha in bound.items():
                if name in files and files[name] != sha:
                    raise ValueError("root source archives disagree: " + name)
                files[name] = sha
        for proof, checks in (
            ("monovalent_norm003", 672),
            ("monovalent_norm_multicore004", 168),
        ):
            finite = json.loads((OWNER / proof / "result.json").read_text())
            if (
                finite["status"] != "pass"
                or finite["checks"] != checks
                or not finite["source_match"]
            ):
                raise ValueError("independent finite norm prerequisite failed")
            bound = json.loads((OWNER / proof / "source_sha256.json").read_text())
            for name, sha in bound.items():
                if digest(ROOT / name) != sha:
                    raise ValueError(
                        "finite physical norm prerequisite source differs: " + name
                    )
                if name in files and files[name] != sha:
                    raise ValueError(
                        "root and norm prerequisite sources differ: " + name
                    )
                files[name] = sha
        files[str(Path(__file__).resolve().relative_to(ROOT))] = digest(__file__)
        for name, sha in files.items():
            if digest(ROOT / name) != sha:
                raise ValueError("current complete-root source differs: " + name)
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        save("source_sha256.json", files)
        resource = json.loads((run / "resource.json").read_text())
        occupied = tuple(map(tuple, resource["core_plan"]["occupied_keys"]))
        virtual = tuple(map(tuple, resource["core_plan"]["virtual_keys"]))
        keys = tuple(
            map(tuple, json.loads((out / "contract.json").read_text())["keys"])
        )
        raw = resource["core_plan"]["raw_amplitudes"]
        maxval = max(resource["standalone_plan"]["amplitudes"].values())
        bound = json.loads((run / "numeric64_resource.json").read_text())
        if bound["status"] != "pass" or not bound["before_actual_native_load"]:
            raise ValueError("actual complete numerical resource preflight required")
        retained_bound = bound["maximum_bound_bytes"]
        extra = 3 * 8 * raw + 2 * 8 * maxval + 64 * 1024**2
        total = retained_bound + extra
        if total > 32 * 1024**3:
            raise MemoryError(
                "complete physical norm/replay payload exceeds original32GiB"
            )
        save(
            "resource.json",
            dict(
                retained_complete_root_bound_bytes=retained_bound,
                extra_replay_and_norm_bytes=extra,
                combined_conservative_host_bound_bytes=total,
                cap_bytes=32 * 1024**3,
                norm_extra_cap_bytes=64 * 1024**2,
                scope="complete conservative allocated payload; excludes interpreter/JIT/compiler runtime RSS",
            ),
        )
        parent = OWNER / "yb_full_run008"
        contract = json.loads((run / "contract.json").read_text())
        if (
            digest(parent / "yb_full_inputs.npz") != contract["input_sha256"]
            or digest(parent / "yb_full_solution.npz") != contract["seed_sha256"]
        ):
            raise ValueError("complete core input/seed payload identity differs")
        payloads = [parent / "yb_full_inputs.npz", parent / "yb_full_solution.npz"]
        payloads.extend(
            run / name
            for name in (
                "contract.json",
                "result.json",
                "resource.json",
                "core_replay.json",
                "source_sha256.json",
                "source_postcheck.json",
                "gpu_source_sha256.json",
                "gpu_result.json",
                "gpu_source_postcheck.json",
                "numeric64_execution_contract.json",
                "numeric64_result.json",
                "numeric64_resource.json",
            )
        )
        for key in keys:
            payloads.extend(
                run / name
                for name in (
                    f"root_{key[0]}_{key[1]}.json",
                    f"root_{key[0]}_{key[1]}.npz",
                    f"cpu_root_recheck_{key[0]}_{key[1]}.json",
                )
            )
        for proof in ("monovalent_norm003", "monovalent_norm_multicore004"):
            payloads.extend(
                OWNER / proof / name for name in ("result.json", "source_sha256.json")
            )
        payload_hashes = {str(p.relative_to(ROOT)): digest(p) for p in payloads}
        save("payload_sha256.json", payload_hashes)
        with np.load(parent / "yb_full_inputs.npz", allow_pickle=False) as inputs:
            workspace = {
                k.removeprefix("workspace_"): inputs[k].copy()
                for k in inputs
                if k.startswith("workspace_")
            }
            workspace = {
                k: v.item() if v.ndim == 0 else v for k, v in workspace.items()
            }
            waves = {
                key: (
                    inputs[f"P_{key[0]}_{key[1]}"].copy(),
                    inputs[f"Q_{key[0]}_{key[1]}"].copy(),
                )
                for key in occupied + virtual
            }
        with np.load(parent / "yb_full_solution.npz", allow_pickle=False) as seed_file:
            seed = seed_file["amplitudes"].copy()
        options = dict(
            term21_threads=8,
            term21_mode="streamed",
            core_singles_blas=True,
            core_single_to_doubles_blas=True,
        )
        features = PolynomialCellCoulombFeatures(workspace)
        cp = plan_core_sd(occupied, virtual, features.identity, **options)
        vp = plan_reduced_monovalent_sd(
            cp, keys, valence_term2_blas=True, valence_term2_threads=8
        )
        if (
            plain(cp) != resource["core_plan"]
            or plain(vp) != resource["standalone_plan"]
        ):
            raise ValueError("actual complete physical coordinate plans differ")
        observe(stage="full_norm_preflight_pass", combined_bound_bytes=total)
        core = CoreSD(
            waves,
            workspace,
            70,
            workspace["ns"],
            occupied_keys=occupied,
            virtual_keys=virtual,
            nuclear_potential=make_nuclear_potential(70, "fermi"),
            backend_directory=run / "backend",
            memory_bytes=32 * 1024**3,
            radial_representation="polynomial",
            **options,
        )
        if core.provenance["metric_error"] > 1e-10:
            raise ValueError("complete orthonormal physical reference metric required")
        prior = json.loads((run / "core_replay.json").read_text())
        for field in (
            "hamiltonian",
            "binding",
            "physical_source_sha256",
            "occupied_keys",
            "virtual_keys",
            "feature_identity",
            "equation",
            "amplitude_representation",
            "backend_pin",
        ):
            if plain(core.provenance[field]) != prior["provenance"][field]:
                raise ValueError(
                    "actual complete core producer identity differs: " + field
                )
        action = core.action

        def timed_action(vector):
            before = time.perf_counter()
            observe(stage="enter_current_core_action", matvecs=core._matvecs)
            value = action(vector)
            observe(
                stage="current_core_action",
                seconds=time.perf_counter() - before,
                matvecs=core._matvecs,
            )
            return value

        core.action = timed_action
        current = core.solve(initial_amplitudes=seed)
        identical = np.array_equal(current.amplitudes, seed)
        save(
            "core_replay.json",
            dict(
                seed_bitwise_identical=identical,
                actual_amplitude_sha256=hashlib.sha256(
                    current.amplitudes.tobytes()
                ).hexdigest(),
                residual_max_Ha=current.residual_max_Ha,
                correlation_energy_Ha=current.correlation_energy_Ha,
                matvecs=current.matrix_vector_products,
                provenance=current.provenance,
            ),
        )
        if not identical or current.residual_max_Ha > 1e-10:
            raise ValueError(
                "actual current fixed core amplitudes differ from complete root seed"
            )
        del seed
        rows = []
        for key in keys:
            prefix = f"root_{key[0]}_{key[1]}"
            root = json.loads((run / (prefix + ".json")).read_text())
            audit = json.loads(
                (run / f"cpu_root_recheck_{key[0]}_{key[1]}.json").read_text()
            )
            if (
                audit["status"] != "pass"
                or max(audit["raw_residual_Ha"], audit["energy_residual_Ha"]) > 1e-10
            ):
                raise ValueError("actual complete CPU root gate required")
            if (
                root["column_key"] != list(key)
                or root["channels"] != plain(vp["channels"][key])
                or root["single_keys"] != plain(vp["single_keys"][key])
            ):
                raise ValueError(
                    "actual complete root/physical norm coordinate identity differs"
                )
            with np.load(run / (prefix + ".npz"), allow_pickle=False) as packed:
                rho = packed["amplitudes"].copy()
            begin = time.perf_counter()
            values = []
            for twoM in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
                values.append(
                    dict(
                        twoM=twoM,
                        **norm_components(
                            cp, vp, current.amplitudes, rho, key, twoM=twoM
                        ),
                    )
                )
            spread = max(x["norm_squared"] for x in values) - min(
                x["norm_squared"] for x in values
            )
            if spread > 1e-11:
                raise ValueError(
                    "complete Yb multiplet physical norm covariance failed"
                )
            row = dict(
                key=key,
                root_json_sha256=digest(run / (prefix + ".json")),
                root_npz_sha256=digest(run / (prefix + ".npz")),
                components=values,
                complete_M_spread=spread,
                seconds=time.perf_counter() - begin,
                timing_scope="single actual workload observations under live load; no controlled speedup",
            )
            rows.append(row)
            save("progress.json", dict(roots=rows))
            observe(
                stage="complete_physical_norm_pass",
                key=key,
                norm_squared=values[0]["norm_squared"],
                complete_M_spread=spread,
                seconds=row["seconds"],
            )
        mismatches = [name for name, sha in files.items() if digest(ROOT / name) != sha]
        if mismatches:
            raise ValueError("physical norm source changed during complete run")
        payload_mismatches = [
            name for name, sha in payload_hashes.items() if digest(ROOT / name) != sha
        ]
        save(
            "source_and_payload_postcheck.json",
            dict(
                status="fail" if mismatches or payload_mismatches else "pass",
                source_count=len(files),
                payload_count=len(payload_hashes),
                source_mismatches=mismatches,
                payload_mismatches=payload_mismatches,
            ),
        )
        if payload_mismatches:
            raise ValueError("bound complete norm input/root payload changed")
        save(
            "result.json",
            dict(
                status="pass",
                rows=rows,
                source_match=True,
                complete_core_seed_measured_bitwise_equal=True,
                root_run=run.name,
                peak_rss_bytes=process_resource.getrusage(
                    process_resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                elapsed_seconds=time.perf_counter() - start,
                scope="private complete real linearized SD physical norm including core spectators/interference/Pauli and all M; no wavefunction export, property, full-H stationarity, physical convergence or resonance certificate",
                highest_level_achieved=False,
            ),
        )
    except Exception as error:
        save(
            "failure.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                elapsed_seconds=time.perf_counter() - start,
            ),
        )
        raise


if __name__ == "__main__":
    main()
