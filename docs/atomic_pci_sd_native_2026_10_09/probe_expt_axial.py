"""Experimental full signed C1/axial CCSDT control, with native finite FCI.

Tests exact-zero symmetry partition only. All core and virtual spinors, original
tensor equations, full triples and physical matrix phases remain unchanged.
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path.insert(0, str(ROOT / "src"))

from atomx.angular.recoupling import l_of_kappa  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.fscc import (  # noqa: E402
    export_integrals,
    read_heff,
    resource_estimate,
)
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.physics.csf.sparse_ci import assemble_sparse_hamiltonian  # noqa: E402
from atomx.physics.csf.spinor_integrals import (  # noqa: E402
    SpinorIntegrals,
    build_spinor_integrals,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def irrep(m2, parity):
    # Original generate_irreps_Dinfh(62): half-integer g then u, +M then -M.
    return (0 if parity == 1 else 62) + abs(m2) - 1 + int(m2 < 0)


def main(args):
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)
    source = out / "source"
    hashes = {}
    for p in sorted((ROOT / "src/atomx").rglob("*.py")) + [
        Path(__file__).resolve(),
        OWNER / "expt_axial_patch.py",
    ]:
        q = source / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        hashes[str(p.relative_to(ROOT))] = digest(p)
    (out / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    exe = args.executable.resolve(strict=True)
    backend = json.loads((exe.parent / "atomx_backend.json").read_text())
    if (
        backend["executable_sha256"] != digest(exe)
        or b"ATOMX_AXIAL_V1" not in exe.read_bytes()
    ):
        raise ValueError("actual experimental axial executable identity missing")
    contract = dict(
        hypothesis="Exact signed M/parity blocking preserves full native CCSDT effective Hamiltonian and complete-three-electron CI spectrum",
        fixed="Be2+1s2 Z4/ns30/ks7 point nucleus, 2s/2p first VN roots, all10magnetic spinors, 2core/8virtual; full CCSDT0h0p/0h1p, triplesDIIS, no integral/amplitude truncation or fitted alignment",
        change="Optional axial/parity sidecar only, same executable/complete signed integrals and equations; second case complex within-M/parity virtual rotation seed314519",
        matrix_gate_Ha=1e-9,
        fci_gate_Ha=1e-9,
        update_gate=1e-12,
        raw_bound_gate_Ha=1e-9,
        maxiter=200,
        cap_bytes=1024**3,
        blas_inner_threads=1,
        backend=backend,
        load=os.getloadavg(),
        stop="Reject any incomplete group, actual forbidden signed integral, matrix/FCI/raw gate failure; performance and large-space claims unsupported",
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    orbitals = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    for kappa in (-1, 1, -2):
        _, states = build_virtual_basis(
            core, kappa, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        orbitals[2, kappa] = states[0][:2]
    a = build_spinor_integrals(
        orbitals,
        core._bsp_data,
        4,
        30,
        occupied_keys=[(1, -1)],
        virtual_keys=[(2, -1), (2, 1), (2, -2)],
        nuclear_potential=core._nuclear_potential,
    )
    n = len(a.labels)
    masks = [sum(1 << i for i in c) for c in itertools.combinations(range(n), 3)]
    h = assemble_sparse_hamiltonian(
        masks, a.labels, orbitals, core._bsp_data, 4, 30
    ).toarray()
    fci = np.linalg.eigvalsh(h)[:8]
    symmetry = tuple((m2, (-1) ** l_of_kappa(k)) for _, k, m2 in a.labels)
    rng = np.random.default_rng(314519)
    u = np.eye(n, dtype=complex)
    for sym in sorted(set(symmetry[2:])):
        group = [i for i in range(2, n) if symmetry[i] == sym]
        z = rng.normal(size=(len(group), len(group))) + 1j * rng.normal(
            size=(len(group), len(group))
        )
        u[np.ix_(group, group)] = np.linalg.qr(z)[0]
    dense = np.array(
        [a.eri(*i) for i in itertools.product(range(n), repeat=4)]
    ).reshape((n,) * 4)
    rotated = np.einsum(
        "pa,qb,pqrs,rc,sd->abcd", u.conj(), u.conj(), dense, u, u, optimize=True
    )
    packed = np.array(
        [
            rotated[p, q, r, s]
            for p in range(n - 1)
            for q in range(p + 1, n)
            for r in range(p, n - 1)
            for s in range(r + 1, n)
        ]
    )

    def rotate_one(matrix):
        z = u.conj().T @ matrix @ u
        return 0.5 * (z + z.conj().T)

    b = SpinorIntegrals(
        a.labels,
        2,
        rotate_one(a.one_body_Ha),
        rotate_one(a.fock_Ha),
        packed,
        a.frozen_energy_Ha,
        a.reference_energy_Ha,
        dict(
            a.identity,
            representation="explicit within-M/parity complex rotation seed314519",
        ),
    )
    np.savez_compressed(
        out / "inputs.npz",
        labels=a.labels,
        fock=a.fock_Ha,
        one_body=a.one_body_Ha,
        packed=a.packed_eri_Ha,
        unitary=u,
        rotated_fock=b.fock_Ha,
        rotated_packed=b.packed_eri_Ha,
        complete_fci_matrix=h,
        complete_fci_energies=np.linalg.eigvalsh(h),
    )
    env = dict(os.environ)
    libs = REF / "toolchain/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = f"{libs}:{libs}/openblas-serial:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    all_records = []
    matrices = {}
    for name, integrals in (("native", a), ("complex", b)):
        forbidden = 0.0
        for p, q, r, s in itertools.product(range(n), repeat=4):
            if (
                symmetry[p][0] + symmetry[q][0] != symmetry[r][0] + symmetry[s][0]
                or symmetry[p][1] * symmetry[q][1] != symmetry[r][1] * symmetry[s][1]
            ):
                forbidden = max(forbidden, abs(integrals.eri(p, q, r, s)))
        for p, q in itertools.product(range(n), repeat=2):
            if symmetry[p] != symmetry[q]:
                forbidden = max(forbidden, abs(integrals.fock_Ha[p, q]))
        if forbidden > 1e-13:
            raise AssertionError("actual integral symmetry gate failed")
        estimate = resource_estimate(
            integrals, model="CCSDT", sector="0h1p", active_count=8
        )
        if estimate["estimated_working_bytes"] > 1024**3:
            raise MemoryError("complete unchanged tensor control exceeds cap")
        for mode in ("C1", "axial"):
            directory = out / f"{name}_{mode}"
            manifest = export_integrals(integrals, directory)
            if mode == "axial":
                (directory / "ATOMX_AXIAL").write_text(
                    "ATOMX_AXIAL_001 10\n" + "".join(f"{m} {p}\n" for m, p in symmetry)
                )
            flags = " ".join(str(int(i >= 2)) for i in range(n))
            (directory / "cc.inp").write_text(
                "\n".join(
                    [
                        'title "Atomx exact axial proof"',
                        "interface pyscf",
                        "integrals ATOMX_ONE ATOMX_TWO",
                        "model ccsdt",
                        "sector 0h1p",
                        f"active {flags}",
                        "maxiter 200",
                        "conv 1e-12",
                        "diis 8",
                        "diis triples",
                        "memory 1024 mb",
                        "tilesize 4",
                        "nthreads 2",
                        "disk_usage 0",
                        "",
                    ]
                )
            )
            request = dict(
                contract=contract,
                mode=mode,
                integrals=manifest,
                input_sha256=digest(directory / "cc.inp"),
                axial_sha256=digest(directory / "ATOMX_AXIAL")
                if mode == "axial"
                else None,
            )
            (directory / "request.json").write_text(
                json.dumps(request, indent=2) + "\n"
            )
            before = time.perf_counter()
            with (directory / "cc.log").open("w") as log:
                run = subprocess.run(
                    [str(exe), "--no-clean", "cc.inp"],
                    cwd=directory,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            log = (directory / "cc.log").read_text()
            if ("ATOMX_AXIAL_V1 exact M/parity" in log) != (mode == "axial"):
                raise AssertionError("actual requested tensor symmetry was not consumed")
            states = {
                s: (int(ok), int(it), *(float(x) for x in (d1, d2, d3)))
                for s, ok, it, d1, d2, d3 in re.findall(
                    r"ATOMX_CC_STATUS (\dh\dp) (\d+) (\d+) (\S+) (\S+) (\S+)", log
                )
            }
            if (
                run.returncode
                or set(states) != {"0h0p", "0h1p"}
                or any(v[0] != 1 for v in states.values())
            ):
                raise RuntimeError(f"complete electronic equations failed: {directory}")
            update = max(max(v[2:]) for v in states.values())
            bound = 6 * float(np.max(abs(np.diag(integrals.fock_Ha)))) * update
            if update > 1e-12 or bound > 1e-9:
                raise AssertionError("raw amplitude residual bound failed")
            blocks = read_heff(directory / "scratch/HEFF")
            core_energy = blocks["0h0p", 1][0, 0]
            full = np.zeros((8, 8), dtype=complex)
            if mode == "C1":
                full[:] = blocks["0h1p", 1]
            else:
                numbers = [irrep(*symmetry[i]) for i in range(2, n)]
                expected = {("0h1p", r - min(numbers) + 1) for r in set(numbers)} | {
                    ("0h0p", 1)
                }
                if set(blocks) != expected:
                    raise AssertionError("incomplete symmetry-sector matrix blocks")
                for number in sorted(set(numbers)):
                    indices = [i for i, r in enumerate(numbers) if r == number]
                    block = blocks["0h1p", number - min(numbers) + 1]
                    if block.shape != (len(indices), len(indices)):
                        raise AssertionError("declared model-spinor row order differs")
                    full[np.ix_(indices, indices)] = block
            np.savez_compressed(
                directory / "full_signed_heff.npz",
                matrix=full,
                core_energy=core_energy,
                global_active_indices=np.arange(2, n),
                symmetry=symmetry,
            )
            energies = np.sort(np.linalg.eigvals(full).real + core_energy.real)
            err = float(np.max(abs(energies - fci)))
            if (
                abs(core_energy.imag) > 1e-10
                or np.max(abs(np.linalg.eigvals(full).imag)) > 1e-10
                or err > 1e-9
            ):
                raise AssertionError("full three-electron FCI spectrum gate failed")
            matrices[name, mode] = full
            record = dict(
                case=name,
                mode=mode,
                core_energy_Ha=float(core_energy.real),
                fci_error_Ha=err,
                forbidden_signed_integral_max_Ha=float(forbidden),
                update=update,
                residual_bound_Ha=bound,
                statuses=states,
                seconds=time.perf_counter() - before,
                log_sha256=digest(directory / "cc.log"),
            )
            (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            all_records.append(record)
            print(json.dumps(record), flush=True)
    signed = max(
        float(np.max(abs(matrices[name, "C1"] - matrices[name, "axial"])))
        for name in ("native", "complex")
    )
    phase = float(
        np.max(
            abs(
                matrices["complex", "axial"]
                - u[2:, 2:].conj().T @ matrices["native", "axial"] @ u[2:, 2:]
            )
        )
    )
    if max(signed, phase) > 1e-9:
        raise AssertionError(
            "full directed signed matrix or complex phase covariance failed"
        )
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="pass",
                signed_C1_axial_error_Ha=signed,
                full_phase_covariance_error_Ha=phase,
                records=all_records,
                production_claim=False,
                performance_claim=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        out = OWNER / args.run_name
        if out.exists() and not (out / "failure.json").exists():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
