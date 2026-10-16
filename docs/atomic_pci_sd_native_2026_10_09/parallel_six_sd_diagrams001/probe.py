#!/usr/bin/env python3
"""Bounded replay of actual complete core source/PP/HH SD diagrams."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import tarfile
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if hasattr(value, "items"):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def active_sources(root, owners=()):
    paths = set(map(Path, owners))
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if (
                path.suffix == ".py"
                and path.is_relative_to(root)
                and ".venv" not in path.parts
                and "numba_cuda_runtime" not in path.parts
            ):
                paths.add(path)
    return sorted(paths)


def archive_sources(root, output, sources):
    before = {str(path.relative_to(root)): sha(path) for path in sources}
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(root)))
    (output / "source_sha256.json").write_text(json.dumps(before, indent=2) + "\n")
    return before


def run(output):
    import numpy as np
    import payload
    from test_sd_pair_diagrams import independent_ladders

    from atomx.methods.sd_pair_diagrams import CorePairLadders
    from atomx.methods.sd_transverse import build_j_coupled_sd_interaction

    owner = Path(__file__).resolve().parent
    root = owner.parents[2]
    input_path = owner / "inputs.npz"
    shutil.copy2(input_path, output / "inputs.npz")
    sources = active_sources(
        root,
        (
            Path(__file__).resolve(),
            owner / "contract.json",
            owner / "payload.py",
            owner / "reference/allcore-ci.f",
            owner / "reference/identity.json",
        ),
    )
    before = archive_sources(root, output, sources)
    arrays, cases = {}, {}
    data = np.load(input_path, allow_pickle=False)
    for mode in ("dc", "static", "dynamic", "complex_phase"):
        a = payload.load_hamiltonian(data, mode)
        bridge = build_j_coupled_sd_interaction(a)
        ladders = CorePairLadders(bridge)
        oracle = independent_ladders(a, bridge, ladders)
        basis = data["amplitudes_basis"]
        pp = np.asarray([ladders.evaluate(t).particle_particle_Ha for t in basis]).T
        hh = np.asarray([ladders.evaluate(t).hole_hole_Ha for t in basis]).T
        signed = ladders.evaluate(data["amplitudes_signed"])
        gates = dict(
            scalar_metric_max=float(np.max(abs(oracle["metric"] - np.eye(10)))),
            source_full_signed_max_Ha=float(
                np.max(abs(ladders.source_Ha - oracle["source"]))
            ),
            pp_all_directions_max_Ha=float(np.max(abs(pp - oracle["pp"]))),
            hh_all_directions_max_Ha=float(np.max(abs(hh - oracle["hh"]))),
            hh_normal_ordered_reference_max_Ha=float(
                np.max(abs(oracle["normal_ordered_hh_reference"]))
            ),
            signed_pp_max_Ha=float(
                np.max(
                    abs(
                        signed.particle_particle_Ha
                        - oracle["pp"] @ data["amplitudes_signed"]
                    )
                )
            ),
            signed_hh_max_Ha=float(
                np.max(
                    abs(signed.hole_hole_Ha - oracle["hh"] @ data["amplitudes_signed"])
                )
            ),
            linked_doubles_energy_error_Ha=float(
                abs(
                    signed.doubles_linked_energy_Ha
                    - np.vdot(oracle["source"], data["amplitudes_signed"])
                )
            ),
            preserved_fock_max_Ha=float(
                np.max(abs(ladders.interaction.fock_Ha - a.fock_Ha))
            ),
            preserved_one_body_max_Ha=float(
                np.max(abs(ladders.interaction.one_body_Ha - a.one_body_Ha))
            ),
            preserved_reference_error_Ha=float(
                abs(ladders.interaction.reference_energy_Ha - a.reference_energy_Ha)
            ),
            preserved_frozen_scalar_error_Ha=float(
                abs(ladders.interaction.frozen_energy_Ha - a.frozen_energy_Ha)
            ),
        )
        if gates["scalar_metric_max"] > 1e-12 or any(
            v > 1e-11 for k, v in gates.items() if k.endswith("Ha")
        ):
            raise AssertionError("original signed SD subgraph gate failed")
        for key, value in oracle.items():
            arrays[f"{mode}_oracle_{key}"] = value
        for key, value in (("source", ladders.source_Ha), ("pp", pp), ("hh", hh)):
            arrays[f"{mode}_reduced_{key}"] = value
        for block, value in bridge.blocks_Ha.items():
            arrays[f"{mode}_j{block[0]}_parity{block[1]}"] = value
        cases[mode] = dict(
            gates=gates,
            bridge_gates=dict(bridge.gates),
            interaction_identity=clean(bridge.identity),
            full_four_electron_dimension=len(oracle["masks"]),
            scalar_doubles=10,
            noncanonical_fov_max_Ha=float(np.max(abs(a.fock_Ha[:4, 4:]))),
            complete_subgraph_plan=dict(ladders.plan),
            provenance=clean(ladders.provenance),
        )
    # Whole-payload refusal must occur before the block-copy or any solve.
    try:
        CorePairLadders(bridge, memory_bytes=1)
        raise AssertionError("memory refusal missing")
    except MemoryError:
        pass
    np.savez_compressed(output / "arrays.npz", **arrays)
    after = {str(path.relative_to(root)): sha(path) for path in sources}
    if before != after or sha(input_path) != sha(output / "inputs.npz"):
        raise RuntimeError("source or scientific input changed")
    return dict(
        status="passed",
        gates={mode: case["gates"] for mode, case in cases.items()},
        cases=cases,
        resource_refusal_passed=True,
        source_unchanged=True,
        source_sha256=before,
        source_files=list(before),
        input_files=[str(input_path.relative_to(root))],
        input_sha256=sha(input_path),
        arrays_sha256=sha(output / "arrays.npz"),
        native_scalable_sd_transverse_complete=False,
        highest_level_reached=False,
        card_summary="原生 signed J/parity SD 已实际消费完整 doubles source＋PP＋normal-ordered HH 图集；DC/static/dynamic/复相位在全部 10 doubles、495 个四电子行列式独立 Fermi action 下通过 1e-11 Ha 门。保留 noncanonical Fock/reference，native_scalable_sd_transverse_complete=false；剩余 core/Fock/singles/ring、valence、Sigma2 图项未完成。",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise SystemExit("refuse to overwrite evidence")
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    owner = Path(__file__).resolve().parent
    root = owner.parents[2]
    sys.path[:0] = [str(owner), str(root / "src"), str(root / "tests")]
    result = dict(
        status="failed",
        gates={},
        highest_level_reached=False,
        native_scalable_sd_transverse_complete=False,
        card_summary="SD source/PP/HH 未通过，不宣称完整 SD。",
    )
    started = time.monotonic()
    try:
        result.update(run(output))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result.update(
            caps=dict(address_bytes=4 * 1024**3, cpu_seconds=120, threads=1),
            observed_wall_seconds=time.monotonic() - started,
            process_peak_rss_bytes=1024
            * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            performance_claim="uncontrolled bounded observation; no large-scale all-SD, speedup or accuracy extrapolation",
        )
        (output / "result.json").write_text(
            json.dumps(clean(result), indent=2, default=repr) + "\n"
        )


if __name__ == "__main__":
    main()
