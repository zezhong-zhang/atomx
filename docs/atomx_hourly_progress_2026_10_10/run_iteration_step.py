"""One bounded, source-bound atomic calculation; never a passive test replay."""

import argparse
import hashlib
import json
import os
import resource
import sys
import tarfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OWNER = REPO / "docs/atomic_pci_sd_native_2026_10_09"
sys.path.insert(0, str(OWNER))
from probe_bloch_oracle import diagnose  # noqa: E402

from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.fscc import compute  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.physics.csf.spinor_integrals import build_spinor_integrals  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    if job["kind"] != "finite_transverse_bloch" or job["ns"] not in (42, 60, 80, 100):
        raise ValueError("unsupported immutable queue input")
    if job["mode"] not in ("DC", "static", "dynamic"):
        raise ValueError("unsupported Hamiltonian")
    backend = OWNER / "expt_compact_public_source001-build/expt.x"
    files = sorted((REPO / "src/atomx").rglob("*.py")) + [
        Path(__file__),
        OWNER / "probe_bloch_oracle.py",
        REPO / "tests/test_transverse_spinor_integrals.py",
        args.job,
    ]
    binding = {str(p.resolve()): sha(p) for p in files}
    contract = dict(
        job=job,
        hypothesis="Radial refinement of unchanged complete 1s/2s/2p magnetic model spaces separates core/Bloch reconstruction accuracy from the noncanonical FS-CCSDT effective-operator defect.",
        fixed="Z=4; 2 occupied + 8 virtual spinors; full signed 120 determinants and 8 attachment roots; ks=7; VN virtuals; declared DC/static/finite operator. Each new-H result compared to its own full FCI.",
        gates=dict(
            core_Bloch_Ha=1e-11, backend_original_Ha=1e-9, backend_strict_Ha=1e-10
        ),
        caps=dict(
            process_address_bytes=4 * 1024**3,
            backend_bytes=512 * 1024**2,
            backend_timeout_s=120,
            threads=1,
        ),
        stop="One specified space and Hamiltonian per card; preserve every target failure; no model shrinking or gate adjustment.",
        sources=binding,
        backend_sha256=sha(backend),
        load=os.getloadavg(),
    )
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    with tarfile.open(args.output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(
                path,
                arcname=str(path.relative_to(REPO))
                if path.is_relative_to(REPO)
                else path.name,
            )
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    try:
        core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=job["ns"], ks=7).solve()
        if not core.converged:
            raise RuntimeError("closed-core DF did not converge; no downstream scoring")
        orbitals = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
        for k in (-1, 1, -2):
            _, virtual = build_virtual_basis(
                core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
            )
            orbitals[2, k] = virtual[0][:2]
        options = dict(
            occupied_keys=[(1, -1)],
            virtual_keys=[(2, -1), (2, 1), (2, -2)],
            nuclear_potential=core._nuclear_potential,
        )
        dc = build_spinor_integrals(orbitals, core._bsp_data, 4, job["ns"], **options)
        mapping = {
            label[:2]: dc.fock_Ha[i, i].real for i, label in enumerate(dc.labels)
        }
        integrals = (
            dc
            if job["mode"] == "DC"
            else build_spinor_integrals(
                orbitals,
                core._bsp_data,
                4,
                job["ns"],
                include_breit=True,
                breit_orbital_energies=mapping if job["mode"] == "dynamic" else None,
                **options,
            )
        )
        solved = compute(
            integrals,
            executable=backend,
            directory=args.output / "fscc",
            model="CCSDT",
            sector="0h1p",
            active_indices=range(2, 10),
            tolerance=1e-14,
            residual_bound_tolerance_Ha=1e-11,
            memory_bytes=512 * 1024**2,
            tile_size=4,
            threads=1,
            timeout_seconds=120,
        )
        result, arrays = diagnose(integrals, solved.model_space_matrix_Ha)
        result.update(
            job=job,
            core_scf=dict(
                converged=core.converged,
                iterations=core.iterations,
                stopping_residual=core.residual,
            ),
            backend_raw_bound_Ha=solved.equation_residual_bound_Ha,
            numerical_oracle_pass=result["oracle_gate_pass"],
            original_backend_gate_pass=result["backend_heff_max_Ha"] <= 1e-9,
            source_and_inputs_unchanged=all(
                sha(Path(p)) == h for p, h in binding.items()
            )
            and sha(backend) == contract["backend_sha256"],
        )
        assert result["numerical_oracle_pass"] and result["source_and_inputs_unchanged"]
        np.savez_compressed(args.output / "full_arrays.npz", **arrays)
        result["arrays_sha256"] = sha(args.output / "full_arrays.npz")
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps(dict(error=type(error).__name__, message=str(error)))
        )
        raise


if __name__ == "__main__":
    main()
