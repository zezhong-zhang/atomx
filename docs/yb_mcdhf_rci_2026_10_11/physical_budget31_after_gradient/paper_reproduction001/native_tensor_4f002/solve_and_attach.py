"""Complete4f core solve and matrix-free R3 target roots; no dense EA matrix."""

import argparse
import hashlib
import json
import os
import resource
import signal
import time
import traceback
from pathlib import Path

import numpy as np
from angular import AngularCoordinates
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

from atomx.constants import HARTREE_CM1
from atomx.methods.tensor_coupled_cluster import CCSDTTensorOperator, TensorCoreResult
from atomx.physics.csf.spinor_integrals import SpinorIntegrals

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
OLD = HERE.parent / "native_tensor_ccsdt001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_operator(folder, occupied):
    ledger = json.loads((folder / "operator.json").read_text())
    with np.load(folder / "integrals.npz") as z:
        obj = SpinorIntegrals(
            tuple(map(tuple, ledger["labels"])),
            occupied,
            z["h"],
            z["fock"],
            z["g"],
            ledger["shifted_frozen_energy_Ha"],
            0.0,
            ledger["identity"],
        )
    return CCSDTTensorOperator(
        obj, memory_MiB=4096, projection_storage="canonical"
    ), ledger


def matrixfree(op, core, angular, indices, call_limit, log):
    calls = []

    def action(vector):
        vector = np.asarray(vector).reshape(-1)
        if len(calls) >= call_limit:
            raise RuntimeError("predeclared matrix-free EA action limit")
        start = time.perf_counter()
        full = np.zeros(op.attachment.size, dtype=np.asarray(vector).dtype)
        full[indices] = vector
        product = op.attachment_action(core, full)
        outside = (angular.m2 != 1) | (angular.parity != 0)
        leakage = float(np.max(abs(product[outside])))
        if leakage > 1e-10:
            raise AssertionError(("EA sector leakage", leakage))
        calls.append(dict(seconds=time.perf_counter() - start, leakage_Ha=leakage))
        if len(calls) % 20 == 0:
            log("EA_ACTION", len(calls), calls[-1]["seconds"])
        return product[indices]

    return LinearOperator(
        (len(indices), len(indices)), matvec=action, dtype=float
    ), calls


def prepare():
    op, _ledger = load_operator(HERE, 18)
    with np.load(HERE / "fixed_inputs.npz") as z:
        initial = z["core"]
    active = op.scalar_core_indices()
    outside = np.setdiff1d(np.arange(op.core.size), active)
    assert np.all(initial[outside] == 0)
    # Metadata/admission only; no residual, nonlinear or EA root calculation.
    inputs = [
        Path(__file__),
        HERE / "angular.py",
        HERE / "operator.json",
        HERE / "integrals.npz",
        HERE / "fixed_inputs.npz",
        HERE / "probe_result.json",
        HERE / "after_source.json",
        HERE / "performance_summary.json",
        OLD / "actual001/operator.json",
        OLD / "actual001/integrals.npz",
        OLD / "actual001/CCSDT_core.npz",
    ]
    inputs.extend(
        OLD / "actual001" / f"M{m}_CCSDT_R3.npz" for m in (-5, -3, -1, 1, 3, 5)
    )
    inputs.append(OLD / "actual001/M1_FCI.npz")
    contract = dict(
        hypothesis="Complete4f14 core correlation changes both same-space attachment states. The optimized native solver must satisfy all189834 retained SDT equations and preserve angular target identities; matrix-free eigenvectors must pass original projected equations, without fitting.",
        fixed=dict(
            correlated_core_electrons=18,
            frozen_electrons=50,
            virtual_magnetic_spinors=12,
            total_spinors=30,
            nucleus="same empiricalA171",
            radial="same ns180/ks9/R500/RKB",
            Hamiltonian="same repartitioned DC operator and common68e HF origin; noBreit/QED/recoil",
            amplitudes=op.core.size,
            exact_scalar_coordinates=len(active),
            new_core_method="CCSDT",
            initial="exact1324-coordinate embedded old solved core, no new4f amplitudes",
            continuation=[1.0],
            preconditioner="Fock diagonal inverse; no clipping, shift or altered Hamiltonian",
            projection_storage="canonical; all signed permutations/equations",
            EA="completeR3, matrix-free even Mhalf sector, six SR roots, ncv36,tol1e-12,maxiter150, deterministicv0; all target partners constructed by signedJ ladders and each equation reapplied",
            scope="EA-CCSDT/R3 finite radial/virtual model; no full-H state/residual or paper/experimental certificate",
        ),
        gates=dict(
            core_residual_Ha=1e-10,
            source_and_operator="exact",
            small_directed_replay_Ha=1e-9,
            small_J2=1e-12,
            core_Jplus=1e-7,
            target_J2=1e-7,
            EA_projected_residual_Ha=1e-10,
            imaginary_target_Ha=1e-10,
            target_bare_R1_weight_min=0.5,
        ),
        resources=dict(
            cpu=27,
            threads=1,
            AS_bytes=12 * 1024**3,
            wall_seconds=1500,
            tensor_memory_MiB=4096,
            core_residual_calls_max=160,
            large_Arnoldi_EA_calls_max=300,
            direct_partner_equation_actions=12,
            core_max_iterations=40,
            load1_max=8,
            available_min_bytes=32 * 1024**3,
        ),
        stopping="First failed source/operator/equation/identity/resource gate stops, preserving core/roots already produced. No automatic retry, altered k/ncv/cut/rank/tolerance or experimental root selection.",
        source_map_sha256=json.loads((HERE / "after_source.json").read_text())[
            "source_map_sha256"
        ],
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
    )
    save(HERE / "solve_contract.json", contract)
    print(
        json.dumps(
            dict(
                status="SOLVE_PREPARED",
                scalar_coordinates=len(active),
                all_coordinates=op.core.size,
            ),
            indent=2,
        )
    )


def main():
    c = json.loads((HERE / "solve_contract.json").read_text())
    source = json.loads((HERE / "after_source.json").read_text())
    assert all(sha(ROOT / p) == h for p, h in source["files"].items())
    assert all(sha(ROOT / p) == h for p, h in c["inputs"].items())
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
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    def alarm(*_):
        raise TimeoutError("declared1500second full4f pilot limit")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1500)
    out = HERE / "actual001"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(
        status="RUNNING",
        controls=[],
        source_map_sha256=source["source_map_sha256"],
        contract_sha256=sha(HERE / "solve_contract.json"),
    )

    def log(*values):
        print(*values, flush=True)

    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        report["controls"].append(
            dict(name=name, measured=measured, cap=cap, passed=bool(passed))
        )
        if not passed:
            raise AssertionError((name, measured, cap))

    try:
        oldop, _ = load_operator(OLD / "actual001", 4)
        with np.load(OLD / "actual001/CCSDT_core.npz") as z:
            previous = z["amplitudes"]
        oldenergy = oldop.core_energy(previous)
        oldcore = TensorCoreResult(
            previous,
            oldenergy,
            oldenergy,
            float(np.max(abs(oldop.core_residual(previous)))),
            0,
            oldop.operator_sha256,
        )
        oldangular = AngularCoordinates(oldop, True)
        for m in (-5, -3, -1, 1, 3, 5):
            with np.load(OLD / "actual001" / f"M{m}_CCSDT_R3.npz") as z:
                indices = oldangular.sector(m)
                gate(
                    f"oldM{m}:complete_coordinate_indices",
                    np.array_equal(indices, z["indices"]),
                )
                pencil = np.empty_like(z["projected_H"])
                for begin in range(0, len(indices), 4):
                    chunk = indices[begin : begin + 4]
                    vectors = np.zeros((oldop.attachment.size, len(chunk)))
                    vectors[chunk, np.arange(len(chunk))] = 1
                    pencil[:, begin : begin + len(chunk)] = oldop.attachment_action(
                        oldcore, vectors
                    )[indices]
                gate(
                    f"oldM{m}:all_directed_canonical_entries",
                    np.max(abs(pencil - z["projected_H"])),
                    1e-9,
                )
        indices = oldangular.sector(1)
        rng = np.random.default_rng(62015)
        probe = np.zeros(oldop.attachment.size)
        probe[indices] = rng.normal(size=len(indices))
        with np.load(OLD / "actual001/M1_FCI.npz") as fci:
            lookup = {int(mask): i for i, mask in enumerate(fci["masks"])}
            rows = [lookup[oldangular.masks[i]] for i in indices]
            expected = oldangular.phases[indices] * (
                fci["J2"][np.ix_(rows, rows)]
                @ (oldangular.phases[indices] * probe[indices])
            )
            gate(
                "small_independent_signed_J2",
                np.max(abs(oldangular.j_squared(probe, 0.5)[indices] - expected)),
                1e-12,
            )
        small_operator, small_calls = matrixfree(
            oldop, oldcore, oldangular, indices, 200, log
        )
        v0 = rng.normal(size=len(indices))
        roots, vectors = eigs(
            small_operator, k=6, which="SR", ncv=36, tol=1e-12, maxiter=150, v0=v0
        )
        with np.load(OLD / "actual001/M1_CCSDT_R3.npz") as z:
            complete = np.linalg.eigvals(z["projected_H"])
        expected = complete[np.argsort(complete.real)[:6]]
        row, col = linear_sum_assignment(abs(roots[:, None] - expected[None, :]))
        gate(
            "small_matrixfree_six_roots", np.max(abs(roots[row] - expected[col])), 1e-9
        )
        gate(
            "small_matrixfree_all_root_equations",
            np.max(abs(small_operator @ vectors - vectors * roots)),
            1e-10,
        )
        log("SMALL_QUALIFICATION_PASS")
        op, ledger = load_operator(HERE, 18)
        with np.load(HERE / "fixed_inputs.npz") as z:
            initial = z["core"]
            old_indices = z["old_indices"]
            new_channel = z["new_channel"]
        angular_core = AngularCoordinates(op, False)
        gate(
            "initial_core_scalar_J",
            np.linalg.norm(angular_core.ladder(initial, 1)),
            1e-7,
        )
        trial = TensorCoreResult(
            initial,
            oldenergy,
            oldenergy,
            json.loads((HERE / "probe_result.json").read_text())[
                "new4f_residual_by_rank"
            ]["2"],
            0,
            op.operator_sha256,
        )
        original = op.core_residual
        residual_calls = []
        last_evaluation = {}

        def counted(*args, **kwargs):
            if len(residual_calls) >= 160:
                raise RuntimeError("predeclared160 core residual call cap")
            start = time.perf_counter()
            value = original(*args, **kwargs)
            last_evaluation.update(
                vector=np.array(args[0], copy=True), residual=value.copy()
            )
            residual_calls.append(
                dict(
                    seconds=time.perf_counter() - start,
                    maximum_Ha=float(np.max(abs(value))),
                )
            )
            if len(residual_calls) % 10 == 0:
                log("CORE_RESIDUAL_EVALUATION", len(residual_calls), residual_calls[-1])
                save(
                    out / "core_progress.json",
                    dict(
                        calls=residual_calls,
                        scope="function evaluations include numerical Jacobian directions, not accepted stationary iterates",
                    ),
                )
            return value

        op.core_residual = counted
        log("SOLVE_FULL4F_CCSDT", len(op.scalar_core_indices()), op.core.size)
        core_started = time.perf_counter()
        core = op.solve_core(
            initial=trial,
            continuation=(1.0,),
            retained_rank=3,
            scalar_sector=True,
            preconditioner="fock_diagonal",
            max_iterations=40,
            tolerance_Ha=1e-10,
        )
        core_seconds = time.perf_counter() - core_started
        op.core_residual = original
        gate(
            "all_retained_SDT_equations", np.max(abs(original(core.amplitudes))), 1e-10
        )
        gate(
            "core_scalar_J",
            np.linalg.norm(angular_core.ladder(core.amplitudes, 1)),
            1e-7,
        )
        gate(
            "new4f_connected_amplitudes_nonzero",
            np.linalg.norm(core.amplitudes[new_channel]) > 1e-8,
        )
        report["core"] = dict(
            energy_relative_Ha=core.total_energy_Ha,
            energy_absolute_Ha=ledger["absolute_energy_origin_Ha"]
            + core.total_energy_Ha,
            correlation_Ha=core.correlation_energy_Ha,
            residual_Ha=core.amplitude_residual_Ha,
            solver_evaluations=core.evaluations,
            measured_calls=len(residual_calls),
            seconds=core_seconds,
            rank_norms={
                str(k + 1): float(np.linalg.norm(core.amplitudes[s]))
                for k, (s, *_) in enumerate(op.core.blocks)
            },
            new4f_amplitude_norm=float(np.linalg.norm(core.amplitudes[new_channel])),
            change_from_frozen4f_core_Ha=core.total_energy_Ha - oldenergy,
            full_H_residual="not computed; no full18-electron Fock-sector representation",
        )
        np.savez(
            out / "core.npz",
            amplitudes=core.amplitudes,
            initial=initial,
            old_indices=old_indices,
            new_channel=new_channel,
        )
        save(out / "partial.json", report)
        log("CORE_CONVERGED", report["core"])
        angular = AngularCoordinates(op, True)
        indices = angular.sector(1)
        gate("complete_Mhalf_even_R3_size", len(indices) == 3011)
        eigen_operator, ea_calls = matrixfree(op, core, angular, indices, 300, log)
        rng = np.random.default_rng(62015)
        v0 = rng.normal(scale=0.01, size=len(indices))
        for i, index in enumerate(indices):
            if index < op.v:
                v0[i] += 1.0
        root_started = time.perf_counter()
        try:
            roots, vectors = eigs(
                eigen_operator, k=6, which="SR", ncv=36, tol=1e-12, maxiter=150, v0=v0
            )
        except ArpackNoConvergence as error:
            np.savez(
                out / "partial_Mhalf_eigenpairs.npz",
                indices=indices,
                energies=error.eigenvalues,
                vectors=error.eigenvectors,
            )
            raise
        np.savez(
            out / "Mhalf_roots.npz", indices=indices, energies=roots, vectors=vectors
        )
        root_error = np.max(abs(eigen_operator @ vectors - vectors * roots))
        gate("all_six_Mhalf_projected_equations", root_error, 1e-10)
        targets = {}
        partners = {}
        for name, key, j in [
            ("S", (6, -1), 0.5),
            ("D3", (5, 2), 1.5),
            ("D5", (5, -3), 2.5),
        ]:
            bare = [
                i
                for i, index in enumerate(indices)
                if index < op.v and tuple(op.labels[op.o + index][:2]) == key
            ]
            weights = np.sum(abs(vectors[bare]) ** 2, axis=0)
            chosen = np.flatnonzero((weights >= 0.5) & (abs(roots.imag) < 1e-10))
            gate(name + ":unique_bare_R1_root", len(chosen) == 1)
            k = int(chosen[0])
            full = np.zeros(op.attachment.size, dtype=complex)
            full[indices] = vectors[:, k]
            gate(
                name + ":J2",
                np.linalg.norm(angular.j_squared(full, 0.5) - j * (j + 1) * full),
                1e-7,
            )
            energy = float(roots[k].real)
            states = {1: full}
            for direction in (1, -1):
                state = full
                m = 0.5
                while abs(m + direction) <= j:
                    factor = np.sqrt((j - direction * m) * (j + direction * m + 1))
                    state = angular.ladder(state, direction) / factor
                    m += direction
                    states[int(round(2 * m))] = state
            gate(name + ":complete_ladder_partners", len(states) == int(2 * j + 1))
            for m2, state in states.items():
                gate(
                    name + f":M{m2}:canonical_norm",
                    abs(np.linalg.norm(state) - 1),
                    1e-8,
                )
                gate(
                    name + f":M{m2}:J2",
                    np.linalg.norm(
                        angular.j_squared(state, m2 / 2) - j * (j + 1) * state
                    ),
                    1e-7,
                )
                gate(
                    name + f":M{m2}:all_R3_equations",
                    np.max(abs(op.attachment_action(core, state) - energy * state)),
                    1e-10,
                )
                partners[name + f"_M{m2}"] = state
            targets[name] = dict(
                root=k,
                EA_Ha=energy,
                EA_cm1=energy * HARTREE_CM1,
                total69_relative_Ha=core.total_energy_Ha + energy,
                total69_absolute_Ha=ledger["absolute_energy_origin_Ha"]
                + core.total_energy_Ha
                + energy,
                bare_R1_weight=float(weights[k]),
                magnetic_partners=len(states),
                scope="normalized canonical projectedR, not exp(T) physical norm",
            )
        np.savez(out / "all_target_partners.npz", **partners)
        report["targets"] = targets
        report["lines_cm1"] = {
            name: (targets[name]["EA_Ha"] - targets["S"]["EA_Ha"]) * HARTREE_CM1
            for name in ("D3", "D5")
        }
        previous_lines = json.loads((OLD / "actual001/result.json").read_text())[
            "lines_cm1"
        ]["CCSDT_R3"]
        report["4f_opening_line_changes_cm1"] = {
            name: report["lines_cm1"][name] - previous_lines[name]
            for name in ("D3", "D5")
        }
        report["EA"] = dict(
            seconds=time.perf_counter() - root_started,
            matvec_calls=len(ea_calls),
            calls=ea_calls,
            dimension=len(indices),
            full_R3_coordinates=op.attachment.size,
            full_dense_matrix_constructed=False,
        )
        report["core_residual_calls"] = residual_calls
        gate(
            "source_unchanged",
            all(sha(ROOT / p) == h for p, h in source["files"].items()),
        )
        gate(
            "all_inputs_unchanged",
            all(sha(ROOT / p) == h for p, h in c["inputs"].items()),
        )
        report["status"] = "PASS_FULL4F_NATIVE_CCSDT_MATRIXFREE_TARGETS"
    except Exception:
        report.update(status="FAILED", exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        if "residual_calls" in locals():
            report["core_residual_calls"] = residual_calls
        if report["status"] == "FAILED" and locals().get("last_evaluation"):
            np.savez(out / "last_core_evaluation.npz", **last_evaluation)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_RSS_MiB"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        save(out / "result.json", report)
        log("END", report["status"], report["elapsed_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else main()
