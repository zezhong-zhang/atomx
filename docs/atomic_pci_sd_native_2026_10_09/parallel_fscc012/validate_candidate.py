"""Unchanged-H candidate FS-CCSDT rerun with complete signed amplitude audit."""

import argparse
import json
import os
import resource
import tarfile
import time
from pathlib import Path

import numpy as np
import probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, default=probe.DEFAULT_CASE)
    parser.add_argument(
        "--executable", type=Path, default=Path(__file__).parent / "candidate001/expt.x"
    )
    parser.add_argument("--symmetry", choices=("C1", "axial"), default="C1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = probe.source_inputs(args.case) + [
        Path(__file__),
        args.executable,
        args.executable.parent / "atomx_backend.json",
    ]
    binding = {str(path.resolve()): probe.sha(path) for path in files}
    contract = dict(
        hypothesis="Removing only the duplicated antisymmetrization in dgd2b_2 repairs the original unchanged-H signed attachment equations.",
        fixed="Complete saved ten-spinor Hamiltonian; 2 core electrons, eight active magnetic spinors and all 120 determinant rows. Full CCSDT, no Fov deletion or fitted shift.",
        gates=dict(
            strict_backend_Ha=1e-10, complete_bloch_Ha=1e-11, core_and_norm=1e-12
        ),
        stop="One candidate solve <=120 seconds, one BLAS thread, 4 GiB address cap. Any gate failure stays visible; no production backend replacement.",
        source_and_inputs=binding,
        load=os.getloadavg(),
        available_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
    )
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    with tarfile.open(args.output / "source_inputs.tar.gz", "w:gz") as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=path.resolve().relative_to(probe.REPO))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    start = time.perf_counter()
    from atomx.methods.fscc import compute, load_integrals

    integrals = load_integrals(args.case)
    result = compute(
        integrals,
        executable=args.executable,
        directory=args.output / "fscc",
        model="CCSDT",
        sector="0h1p",
        active_indices=tuple(range(2, 10)),
        tolerance=1e-14,
        residual_bound_tolerance_Ha=1e-11,
        memory_bytes=512 * 1024**2,
        tile_size=4,
        threads=1,
        timeout_seconds=120,
        environment=os.environ.copy(),
        tensor_symmetry=args.symmetry,
    )
    metrics, arrays = probe.audit(args.output / "fscc")
    roots = np.linalg.eigvalsh(arrays["full_h3_Ha"])[:8]
    metrics.update(
        candidate_energy_root_max_Ha=float(
            np.max(abs(np.sort(result.sector_energies_Ha) - roots))
        ),
        candidate_raw_equation_bound_Ha=result.equation_residual_bound_Ha,
        original_integral_files_match={
            name: probe.sha(args.case / name) == probe.sha(args.output / "fscc" / name)
            for name in ("ATOMX_ONE", "ATOMX_TWO")
        },
        original_backend_unchanged=probe.sha(probe.BACKEND) == probe.BACKEND_SHA,
        candidate_backend_sha256=probe.sha(args.executable),
        source_and_inputs=binding,
    )
    metrics["complete_candidate_gate_pass"] = bool(
        metrics["decoder_and_core_gate_pass"]
        and metrics["strict_backend_gate_pass"]
        and metrics["exact_oracle_gate_pass"]
        and metrics["backend_wave_bloch_residual_max_Ha"] <= 1e-11
        and metrics["candidate_energy_root_max_Ha"] <= 1e-10
        and all(metrics["original_integral_files_match"].values())
    )
    metrics["status"] = (
        "finite_candidate_pass"
        if metrics["complete_candidate_gate_pass"]
        else "candidate_scientific_gate_failed"
    )
    if not all(probe.sha(Path(path)) == digest for path, digest in binding.items()):
        raise ValueError(
            "candidate source or original inputs changed during validation"
        )
    np.savez_compressed(args.output / "full_arrays.npz", **arrays)
    metrics.update(
        inputs_unchanged=True,
        arrays_sha256=probe.sha(args.output / "full_arrays.npz"),
        archive_sha256=probe.sha(args.output / "source_inputs.tar.gz"),
        observed_phase_wall_seconds=time.perf_counter() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    (args.output / "result.json").write_text(json.dumps(metrics, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k not in ("decoder_metadata", "source_and_inputs")
            }
        )
    )


if __name__ == "__main__":
    main()
