"""Admit complete6p1/2+6p3/2 and measure their full native SDT/R3 probe."""

import argparse
import hashlib
import itertools
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import linalg

from atomx.constants import C_SQ
from atomx.methods.dirac_fock import SCFResult
from atomx.methods.tensor_coupled_cluster import (
    CCSDTTensorOperator,
    TensorCoreResult,
    resource_plan,
)
from atomx.physics.csf.spinor_integrals import SpinorIntegrals, build_spinor_integrals
from atomx.potentials.nuclear import make_nuclear_potential

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
OLD = HERE.parent / "native_tensor_4f002"
DHF = HERE.parent / "yb_dhf_table001"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(DHF))
from angular import AngularCoordinates  # noqa: E402
from constrained_pair import independent_controls, refine  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_operator(folder):
    ledger = json.loads((folder / "operator.json").read_text())
    with np.load(folder / "integrals.npz") as z:
        obj = SpinorIntegrals(
            tuple(map(tuple, ledger["labels"])),
            18,
            z["h"],
            z["fock"],
            z["g"],
            ledger["shifted_frozen_energy_Ha"],
            0.0,
            ledger["identity"],
        )
    return CCSDTTensorOperator(
        obj, memory_MiB=8192, projection_storage="canonical"
    ), ledger


def embed(vector, old_coordinates, new_coordinates):
    assert old_coordinates.occupied == new_coordinates.occupied == 18
    result = np.zeros(new_coordinates.size, dtype=np.asarray(vector).dtype)
    selected = []
    for rank, ((old_slice, *_), (new_slice, *_)) in enumerate(
        zip(old_coordinates.blocks, new_coordinates.blocks, strict=True), 1
    ):
        old_p = list(itertools.combinations(range(old_coordinates.virtual), rank))
        new_p = {
            p: i
            for i, p in enumerate(
                itertools.combinations(range(new_coordinates.virtual), rank)
            )
        }
        mapped = np.array([new_p[p] for p in old_p])
        holes = (old_slice.stop - old_slice.start) // len(old_p)
        indices = new_slice.start + np.arange(holes)[:, None] * len(new_p) + mapped
        result[indices.ravel()] = vector[old_slice]
        selected.extend(indices.ravel())
    return result, np.asarray(selected, dtype=int)


def prepare():
    inputs = [
        Path(__file__),
        HERE / "source.json",
        OLD / "angular.py",
        DHF / "constrained_pair.py",
        DHF / "actual004/blocks_ns180.npz",
        DHF / "actual004/core_ns180.h5",
        OLD / "operator.json",
        OLD / "integrals.npz",
        OLD / "actual001/core.npz",
        OLD / "actual003/S_root.npz",
        OLD / "actual003/result.json",
        OLD / "increment_receipt.json",
    ]
    save(
        HERE / "probe_contract.json",
        dict(
            hypothesis="Missing complete p virtual multiplets leave unsatisfied core/EA channels even at the converged complete4f solution. Adding6p1/2 and6p3/2 changes only virtual space, recovering every old integral/equation at exact zero new amplitudes and measuring larger contraction cost before a solve.",
            fixed=dict(
                core="18correlated/50frozen;5s2,5p1/2^2,4f14",
                virtual="previous complete6s,5d3,5d5 plus complete6p1/2,6p3/2;18magnetic virtual spinors",
                radial="same owned ns180/order9/R500/RKB and empiricalA171 core; original saved p Fock/metric constrained electron roots, exactly2 bordered corrections",
                Hamiltonian="DC only, same68e HF origin; no fitted level/borrowed correction/Breit/QED/recoil",
                equations="complete689589 SDT and127620 R3 coordinates; no amplitude screening",
                source="same362productionfiles",
                scope="One off-shell residual and one embedded oldS action; no nonlinear/root solve in this probe",
            ),
            method="Complete occupied complement of original p H/S, lower-triangle symmetric generalized eigensolve, all electron branch E>-c^2, lowest root constrained original-pencil refinement; no root/count/tolerance scan.",
            gates=dict(
                radial_dual_Ha=1e-8,
                physical_Gram=1e-10,
                core_overlap=1e-10,
                same_old_operator_Ha=1e-10,
                old_embedded_equations_Ha=1e-10,
                same_energy_Ha=1e-10,
                J2=1e-7,
            ),
            resources=dict(
                cpu=27,
                threads=1,
                AS_bytes=24 * 1024**3,
                wall_seconds=240,
                workspace_MiB=8192,
                load1_max=8,
                available_min_bytes=32 * 1024**3,
                scalar_plan=resource_plan(18, 18),
                expected_core_calls=1,
                expected_EA_calls=1,
            ),
            stopping="First failed original-physics/source/resource gate stops, preserving evidence. No automatic retry, model reduction, changed residual threshold or experimental root selection.",
            reference=dict(
                url="https://arxiv.org/html/2504.01438v1",
                scope="Paper uses40GTO/symmetry s-i, RCCSDT throughg; this newp subset is a discriminator, not the paper basis.",
            ),
            inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
            source_map_sha256=json.loads((HERE / "source.json").read_text())[
                "source_map_sha256"
            ],
        ),
    )
    print("COMPLETE_6P_PROBE_PREPARED", flush=True)


def main():
    contract = json.loads((HERE / "probe_contract.json").read_text())
    source = json.loads((HERE / "source.json").read_text())
    assert all(sha(ROOT / n) == h for n, h in source["files"].items())
    assert all(sha(ROOT / n) == h for n, h in contract["inputs"].items())
    assert os.getloadavg()[0] <= 8
    available = next(
        int(x.split()[1]) * 1024
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    )
    assert available >= 32 * 1024**3
    assert all(
        os.environ.get(x) == "1"
        for x in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    )
    os.sched_setaffinity(0, {27})
    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))

    def alarm(*_):
        raise TimeoutError("predeclared240second p-virtual probe cap")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(240)
    out = HERE / "probe001"
    out.mkdir(exist_ok=False)
    report = dict(
        status="RUNNING",
        controls=[],
        contract_sha256=sha(HERE / "probe_contract.json"),
        source_map_sha256=source["source_map_sha256"],
    )
    started = time.monotonic()

    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        report["controls"].append(
            dict(name=name, measured=measured, cap=cap, passed=bool(passed))
        )
        if not passed:
            raise AssertionError((name, measured, cap))

    try:
        report["independent_radial_correction"] = independent_controls()
        core = SCFResult.load(DHF / "actual004/core_ns180.h5")
        core._nuclear_potential = make_nuclear_potential(70, "fermi", 171)
        with np.load(DHF / "actual004/blocks_ns180.npz") as z:
            arrays = {k: z[k] for k in z.files}
        orbitals = {
            tuple(map(int, k)): (p, q)
            for k, p, q in zip(
                arrays["all_orbital_keys"],
                arrays["all_P"],
                arrays["all_Q"],
                strict=True,
            )
        }
        keep = arrays["active_coordinates"]
        saved = {}
        report["new_radial_roots"] = []
        for kappa in (1, -2):
            h, s, c = [arrays[f"{n}_k{kappa}"] for n in ("H", "S", "occupied")]
            normals = s @ c
            q, r = linalg.qr(normals, mode="full")
            gate(
                f"p{kappa}:full_constraint_rank",
                np.all(abs(np.diag(r[: c.shape[1]])) > 1e-13),
            )
            b = q[:, c.shape[1] :]
            wide = np.longdouble
            hp = np.asarray(
                b.astype(wide).T @ h.astype(wide) @ b.astype(wide), dtype=float
            )
            sp = np.asarray(
                b.astype(wide).T @ s.astype(wide) @ b.astype(wide), dtype=float
            )
            hp = np.tril(hp) + np.tril(hp, -1).T
            sp = np.tril(sp) + np.tril(sp, -1).T
            energies, coefficients = linalg.eigh(hp, sp, driver="gvd")
            electron = energies > -C_SQ
            energies, coefficients = energies[electron], coefficients[:, electron]
            gate(
                f"p{kappa}:electron_branch_and_bound_lowest",
                len(energies) > 1 and -C_SQ < energies[0] < 0,
            )
            energy, x = refine(h, s, c, float(energies[0]), b @ coefficients[:, 0])
            gap = float(np.min(abs(energies[1:] - energies[0])))
            gate(f"p{kappa}:same_lowest_root", abs(energy - energies[0]) / gap, 0.25)
            residual = np.asarray(
                b.astype(wide).T
                @ (
                    h.astype(wide) @ x.astype(wide)
                    - wide(energy) * s.astype(wide) @ x.astype(wide)
                ),
                dtype=float,
            )
            dual = np.linalg.norm(
                linalg.solve_triangular(
                    linalg.cholesky(sp, lower=True), residual, lower=True
                )
            )
            gram = abs(float(x.astype(wide) @ s.astype(wide) @ x.astype(wide)) - 1)
            overlap = float(
                np.max(abs(c.astype(wide).T @ s.astype(wide) @ x.astype(wide)))
            )
            gate(f"p{kappa}:original_projected_dual", dual, 1e-8)
            gate(f"p{kappa}:physical_Gram", gram, 1e-10)
            gate(f"p{kappa}:occupied_overlap", overlap, 1e-10)
            full = np.zeros(360)
            full[keep] = x
            orbitals[(6, kappa)] = full[:180], full[180:]
            for n, a in (
                ("H", h),
                ("S", s),
                ("occupied", c),
                ("complement", b),
                ("electron_E", energies),
                ("electron_C", coefficients),
                ("selected", full),
            ):
                saved[f"{n}_k{kappa}"] = a
            report["new_radial_roots"].append(
                dict(
                    key=[6, kappa],
                    energy_Ha=energy,
                    dual_Ha=float(dual),
                    Gram=float(gram),
                    overlap=overlap,
                    electron_columns=len(energies),
                )
            )
        np.savez(out / "virtuals.npz", **saved)
        occupied = ((5, -1), (5, 1), (4, 3), (4, -4))
        virtual = ((6, -1), (5, 2), (5, -3), (6, 1), (6, -2))
        frozen = tuple(k for k in core.orbitals_by_subshell() if k not in occupied)
        gate("18active_50frozen", sum(2 * abs(k) for n, k in frozen) == 50)
        original = build_spinor_integrals(
            orbitals,
            core._bsp_data,
            70,
            180,
            occupied_keys=occupied,
            virtual_keys=virtual,
            frozen_keys=frozen,
            nuclear_potential=core._nuclear_potential,
            max_integral_bytes=256 * 1024**2,
            metric_tolerance=1e-10,
        )
        origin = original.reference_energy_Ha
        identity = dict(original.identity)
        identity.update(
            probe="complete4f14 and complete6p1/2,6p3/2;18correlated/50frozen/18virtual",
            energy_origin_Ha=origin,
        )
        obj = SpinorIntegrals(
            original.labels,
            18,
            original.one_body_Ha,
            original.fock_Ha,
            original.packed_eri_Ha,
            original.frozen_energy_Ha - origin,
            0.0,
            identity,
        )
        save(
            HERE / "operator.json",
            dict(
                labels=obj.labels,
                identity=identity,
                absolute_energy_origin_Ha=origin,
                shifted_frozen_energy_Ha=obj.frozen_energy_Ha,
            ),
        )
        np.savez(
            HERE / "integrals.npz",
            h=obj.one_body_Ha,
            fock=obj.fock_Ha,
            g=obj.packed_eri_Ha,
        )
        # Run only the serialized producer representation used by the later solve.
        op, ledger = load_operator(HERE)
        oldop, oldledger = load_operator(OLD)
        gate("old_complete_labels_prefix", op.labels[:30] == oldop.labels)
        gate("same_origin", abs(origin - oldledger["absolute_energy_origin_Ha"]), 1e-10)
        gate(
            "same_frozen_energy",
            abs(obj.frozen_energy_Ha - oldledger["shifted_frozen_energy_Ha"]),
            1e-10,
        )
        gate("same_all_old_Fock", np.max(abs(op.f[:30, :30] - oldop.f)), 1e-10)
        gate("same_all_old_ERI", np.max(abs(op.g[:30, :30, :30, :30] - oldop.g)), 1e-10)
        with np.load(OLD / "integrals.npz") as z:
            gate(
                "same_all_old_one_body",
                np.max(abs(obj.one_body_Ha[:30, :30] - z["h"])),
                1e-10,
            )
        with np.load(OLD / "actual001/core.npz") as z:
            initial, old_indices = embed(z["amplitudes"], oldop.core, op.core)
        with np.load(OLD / "actual003/S_root.npz") as z:
            right, old_ea_indices = embed(z["vector"], oldop.attachment, op.attachment)
            old_S_energy = float(z["energy"])
        new_indices = np.setdiff1d(np.arange(op.core.size), old_indices)
        new_ea_indices = np.setdiff1d(np.arange(op.attachment.size), old_ea_indices)
        np.savez(
            HERE / "fixed_inputs.npz",
            core=initial,
            right=right,
            old_indices=old_indices,
            new_channel=new_indices,
            old_ea_indices=old_ea_indices,
            new_ea_indices=new_ea_indices,
        )
        start = time.perf_counter()
        residual = op.core_residual(initial)
        report["core_residual_seconds"] = time.perf_counter() - start
        energy = op.core_energy(initial)
        oldresult = json.loads((OLD / "actual003/result.json").read_text())
        gate(
            "embedded_old_core_energy",
            abs(energy - oldresult["core"]["energy_relative_Ha"]),
            1e-10,
        )
        gate(
            "embedded_all189834_old_equations",
            np.max(abs(residual[old_indices])),
            1e-10,
        )
        angular = AngularCoordinates(op, False)
        gate(
            "embedded_scalar_core_Jplus",
            np.linalg.norm(angular.ladder(initial, 1)),
            1e-7,
        )
        trial = TensorCoreResult(
            initial, energy, energy, float(np.max(abs(residual))), 0, op.operator_sha256
        )
        start = time.perf_counter()
        product = op.attachment_action(trial, right)
        report["EA_action_seconds"] = time.perf_counter() - start
        ea_residual = product - old_S_energy * right
        gate(
            "embedded_all34860_oldS_equations",
            np.max(abs(ea_residual[old_ea_indices])),
            1e-10,
        )
        gate("new_core_channels_nonzero", np.max(abs(residual[new_indices])) > 1e-8)
        gate(
            "newS_EA_channels_nonzero", np.max(abs(ea_residual[new_ea_indices])) > 1e-8
        )
        np.savez(
            out / "values.npz",
            core_residual=residual,
            EA_action=product,
            EA_residual=ea_residual,
        )
        report.update(
            status="PASS_COMPLETE_6P_OFFSHELL_PROBE",
            core_energy_Ha=energy,
            operator_sha256=op.operator_sha256,
            core_amplitudes=op.core.size,
            scalar_core_coordinates=len(op.scalar_core_indices()),
            attachment_coordinates=op.attachment.size,
            new_core_rank_residuals={
                str(k + 1): float(np.max(abs(residual[s])))
                for k, (s, *_) in enumerate(op.core.blocks)
            },
            newS_EA_residual_Ha=float(np.max(abs(ea_residual[new_ea_indices]))),
        )
        gate(
            "all_source_inputs_unchanged",
            all(sha(ROOT / n) == h for n, h in contract["inputs"].items())
            and all(sha(ROOT / n) == h for n, h in source["files"].items()),
        )
    except Exception:
        report.update(status="FAILED", exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_RSS_MiB"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        save(out / "result.json", report)
        print(
            json.dumps({k: v for k, v in report.items() if k != "controls"}, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else main()
