"""Complete6p virtual expansion: native18e CCSDT and highest-weight R3 roots."""

import argparse
import json
import os
import resource
import signal
import time
import traceback
from pathlib import Path

import numpy as np
from diis_core import solve as solve_diis
from prepare_virtuals import HERE, OLD, ROOT, load_operator, save, sha
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

from atomx.constants import HARTREE_CM1
from atomx.methods.tensor_coupled_cluster import TensorCoreResult

TARGETS = [("S", (6, -1), 0.5), ("D3", (5, 2), 1.5), ("D5", (5, -3), 2.5)]

# prepare_virtuals installs the immutable predecessor angular-helper path.
from angular import AngularCoordinates, act  # noqa: E402


def prepare():
    result = json.loads((HERE / "probe001/result.json").read_text())
    assert result["status"] == "PASS_COMPLETE_6P_OFFSHELL_PROBE"
    inputs = [
        Path(__file__),
        HERE / "prepare_virtuals.py",
        HERE / "source.json",
        HERE / "diis_core.py",
        HERE / "qualify_diis.py",
        HERE / "diis_qualification_contract.json",
        HERE / "diis_qualification.json",
        HERE / "probe_contract.json",
        HERE / "probe001/result.json",
        HERE / "probe001/virtuals.npz",
        HERE / "operator.json",
        HERE / "integrals.npz",
        HERE / "fixed_inputs.npz",
        OLD / "angular.py",
        OLD / "targeted_davidson_J.py",
        OLD / "actual003/result.json",
        OLD / "actual003/all_target_partners.npz",
        OLD / "increment_receipt.json",
    ]
    save(
        HERE / "solve_contract.json",
        dict(
            hypothesis="The first complete p virtual pair changes core/valence correlation unequally; completeSDT and originalR3/J2 residuals must qualify the new spectrum independently of whether experimental agreement improves.",
            fixed=dict(
                nucleus="empiricalA171",
                radial="unchangedns180/order9/R500/RKB with original-constrained6p roots",
                Hamiltonian="sameDC68eorigin;18active/50frozen; noBreit/QED/recoil/fittedcorrection",
                virtual="complete6s,5d3,5d5,6p1,6p3:18magnetic spinors",
                core="all689589SDT; exactM0/even coordinates, embedded oldT seed; Fock-Jacobi/DIIS Anderson candidates with unshifted Fock inverse, last6 errors, no coefficient clipping/regularization/history-rank cutoff; max60calls,1e-10Ha",
                EA="complete127620R3; three prescribed unique bareR1>=0.5 target roots; complete highestM/even sectors, signed ker(Jplus) projector and unshifted Fock-diagonal Davidson; max48basis/target; every12magnetic partner full equation reapplied",
                norm="canonical projectedR, not physical exp(T) norm",
            ),
            gates=dict(
                core_residual_Ha=1e-10,
                EA_residual_Ha=1e-10,
                J2=1e-7,
                core_Jplus=1e-7,
                canonical_R_norm=1e-8,
                original_previous_root_Ha=1e-9,
            ),
            resources=dict(
                cpu=27,
                threads=1,
                AS_bytes=24 * 1024**3,
                workspace_MiB=8192,
                wall_seconds=2400,
                core_residual_calls_max=60,
                DIIS_history=6,
                large_EA_search_calls_max=144,
                large_direct_partner_and_target_checks=15,
                old_direct_qualification_calls=3,
                load1_max=8,
                available_min_bytes=32 * 1024**3,
                probe_core_seconds=result["core_residual_seconds"],
                probe_EA_seconds=result["EA_action_seconds"],
                estimated_total_seconds=60 * result["core_residual_seconds"]
                + 159 * result["EA_action_seconds"],
            ),
            stopping="First failed equation/identity/source/resource gate stops; keep core/target partials. No automatic cut/rank/root/tolerance/restart change or fitting. Solve call and wall caps are independent; stop at whichever comes first.",
            source_map_sha256=json.loads((HERE / "source.json").read_text())[
                "source_map_sha256"
            ],
            inputs={str(q.relative_to(ROOT)): sha(q) for q in inputs},
        ),
    )
    print("COMPLETE_6P_SOLVE_PREPARED", flush=True)


def diagonal(op):
    eps = np.diag(op.f)
    return np.concatenate(
        [
            np.sum(eps[op.o + p], axis=1) - np.sum(eps[h], axis=1)
            for _, p, h, _ in op.attachment.blocks
        ]
    )


def highest_weight_projector(angular, indices, m2, gate, name):
    higher = angular.sector(m2 + 2)
    row_of = {int(index): row for row, index in enumerate(higher)}
    rows, cols, values = [], [], []
    for col, index in enumerate(indices):
        mask = angular.masks[index]
        for p, q, factor in angular.raising:
            if not mask & (1 << q) or mask & (1 << p):
                continue
            end, phase = act(mask, ((p, True), (q, False)))
            target = angular.lookup[end]
            rows.append(row_of[target])
            cols.append(col)
            values.append(
                factor * phase * angular.phases[index] * angular.phases[target]
            )
    a = coo_matrix((values, (rows, cols)), shape=(len(higher), len(indices))).tocsr()
    gram = a @ a.T
    # In complete multiplets at positive M, J+J- is strictly positive on M+1.
    # No SVD cutoff, deleted coefficient, fitted denominator or level shift.
    rng = np.random.default_rng(62016)
    probe = rng.normal(size=len(indices))
    full = np.zeros(angular.size)
    full[indices] = probe
    gate(
        name + ":sparse_vs_signed_Jplus",
        np.max(abs(a @ probe - angular.ladder(full, 1)[higher])),
        1e-12,
    )

    def project(vector):
        rhs = a @ vector
        solution, info = cg(gram, rhs, rtol=1e-14, atol=0.0, maxiter=100)
        gate(name + ":projector_SPD_solve", info == 0)
        result = vector - a.T @ solution
        gate(
            name + ":projector_Jplus",
            np.linalg.norm(a @ result)
            / max(np.linalg.norm(vector), np.finfo(float).tiny),
            1e-12,
        )
        return result

    projected = project(probe)
    gate(
        name + ":projector_idempotence",
        np.linalg.norm(project(projected) - projected) / np.linalg.norm(probe),
        1e-12,
    )
    return project


def solve_target(op, core, angular, name, key, j, gate, progress):
    m2 = int(2 * j)
    indices = angular.sector(m2)
    bare = [
        i
        for i, index in enumerate(indices)
        if index < op.v and tuple(op.labels[op.o + index][:2]) == key
    ]
    gate(name + ":unique_bare_seed", len(bare) == 1)
    bare = bare[0]
    project = highest_weight_projector(angular, indices, m2, gate, name)
    d = diagonal(op)[indices]
    q = np.zeros(len(indices))
    q[bare] = 1.0
    basis, products = [], []
    history = []
    for iteration in range(48):
        full = np.zeros(op.attachment.size)
        full[indices] = q
        gate(
            name + ":basis_J2",
            np.linalg.norm(angular.j_squared(full, j) - j * (j + 1) * full),
            1e-7,
        )
        start = time.perf_counter()
        product = op.attachment_action(core, full)
        outside = (angular.m2 != m2) | (angular.parity != 0)
        gate(name + ":action_sector_leakage", np.max(abs(product[outside])), 1e-10)
        basis.append(q.copy())
        products.append(product[indices])
        v, av = np.column_stack(basis), np.column_stack(products)
        energies, coefficients = np.linalg.eig(v.T @ av)
        ritz = v @ coefficients
        ritz /= np.linalg.norm(ritz, axis=0)
        weights = abs(ritz[bare]) ** 2
        chosen = np.flatnonzero((weights >= 0.5) & (abs(energies.imag) < 1e-10))
        gate(name + ":unique_target_Ritz", len(chosen) == 1)
        k = int(chosen[0])
        coeff = coefficients[:, k] / np.linalg.norm(v @ coefficients[:, k])
        vector = np.real_if_close(v @ coeff, tol=1000)
        gate(name + ":real_target_representation", not np.iscomplexobj(vector))
        energy = float(energies[k].real)
        residual = np.real_if_close(av @ coeff - energy * vector, tol=1000)
        gate(name + ":real_residual_representation", not np.iscomplexobj(residual))
        maximum = float(np.max(abs(residual)))
        history.append(
            dict(
                iteration=iteration + 1,
                EA_Ha=energy,
                residual_Ha=maximum,
                bare_R1_weight=float(weights[k]),
                action_seconds=time.perf_counter() - start,
            )
        )
        progress[name] = history
        print("DAVIDSON", name, history[-1], flush=True)
        save(HERE / "actual001/EA_progress.json", progress)
        if maximum <= 1e-10:
            result = np.zeros(op.attachment.size)
            result[indices] = vector
            return energy, result, indices, history
        denominator = energy - d
        gate(
            name + ":finite_nonzero_preconditioner",
            np.all(np.isfinite(denominator)) and np.all(denominator != 0),
        )
        q = project(residual / denominator)
        for _ in range(2):
            q -= v @ (v.T @ q)
        norm = np.linalg.norm(q)
        gate(name + ":independent_correction", np.isfinite(norm) and norm > 1e-14)
        q /= norm
    raise RuntimeError("predeclared48-dimensional Davidson cap")


def main():
    contract = json.loads((HERE / "solve_contract.json").read_text())
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
        raise TimeoutError("predeclared2400second complete6p solve cap")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(2400)
    out = HERE / "actual001"
    out.mkdir(exist_ok=False)
    report = dict(
        status="RUNNING",
        controls=[],
        contract_sha256=sha(HERE / "solve_contract.json"),
        source_map_sha256=source["source_map_sha256"],
    )
    calls, last = [], {}
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
        oldop, _ = load_operator(OLD)
        oldresult = json.loads((OLD / "actual003/result.json").read_text())
        with np.load(OLD / "actual001/core.npz") as z:
            ot = z["amplitudes"]
        oe = oldop.core_energy(ot)
        oldcore = TensorCoreResult(
            ot,
            oe,
            oe,
            float(np.max(abs(oldop.core_residual(ot)))),
            0,
            oldop.operator_sha256,
        )
        for name, _key, _j in TARGETS:
            with np.load(OLD / "actual003" / (name + "_root.npz")) as z:
                gate(
                    "previous:" + name + ":original_equations",
                    np.max(
                        abs(
                            oldop.attachment_action(oldcore, z["vector"])
                            - float(z["energy"]) * z["vector"]
                        )
                    ),
                    1e-10,
                )
        op, ledger = load_operator(HERE)
        with np.load(HERE / "fixed_inputs.npz") as z:
            initial = z["core"]
            new_channel = z["new_channel"]
        residual_action = op.core_residual

        def counted(vector, **kwargs):
            if len(calls) >= 60:
                raise RuntimeError("predeclared60core residual call limit")
            start = time.perf_counter()
            result = residual_action(vector, **kwargs)
            last.update(vector=vector.copy(), residual=result.copy())
            calls.append(
                dict(
                    seconds=time.perf_counter() - start,
                    maximum_Ha=float(np.max(abs(result))),
                )
            )
            save(
                out / "core_progress.json",
                dict(
                    calls=calls,
                    scope="complete off-shell residuals at successive DIIS points; no numerical Jacobian directions",
                ),
            )
            print("CORE_RESIDUAL", len(calls), calls[-1], flush=True)
            return result

        op.core_residual = counted
        start = time.perf_counter()
        core, diis_history = solve_diis(op, initial=initial)
        save(out / "DIIS_history.json", diis_history)
        core_seconds = time.perf_counter() - start
        op.core_residual = residual_action
        gate(
            "all689589_retained_SDT_equations",
            np.max(abs(residual_action(core.amplitudes))),
            1e-10,
        )
        angular_core = AngularCoordinates(op, False)
        gate(
            "core_scalar_Jplus",
            np.linalg.norm(angular_core.ladder(core.amplitudes, 1)),
            1e-7,
        )
        gate(
            "newp_core_amplitudes_nonzero",
            np.linalg.norm(core.amplitudes[new_channel]) > 1e-8,
        )
        report["core"] = dict(
            energy_relative_Ha=core.total_energy_Ha,
            energy_absolute_Ha=ledger["absolute_energy_origin_Ha"]
            + core.total_energy_Ha,
            correlation_Ha=core.correlation_energy_Ha,
            residual_Ha=core.amplitude_residual_Ha,
            solver_evaluations=core.evaluations,
            measured_calls=len(calls),
            seconds=core_seconds,
            scalar_coordinates=len(op.scalar_core_indices()),
            all_coordinates=op.core.size,
            rank_norms={
                str(k + 1): float(np.linalg.norm(core.amplitudes[s]))
                for k, (s, *_) in enumerate(op.core.blocks)
            },
            newp_amplitude_norm=float(np.linalg.norm(core.amplitudes[new_channel])),
            change_from_no_p_Ha=core.total_energy_Ha
            - oldresult["core"]["energy_relative_Ha"],
            full_H_residual="not computed",
        )
        np.savez(
            out / "core.npz",
            amplitudes=core.amplitudes,
            initial=initial,
            new_channel=new_channel,
        )
        save(out / "partial.json", report)
        print("CORE_CONVERGED", report["core"], flush=True)
        angular = AngularCoordinates(op, True)
        progress, targets, partners = {}, {}, {}
        for name, key, j in TARGETS:
            energy, vector, indices, history = solve_target(
                op, core, angular, name, key, j, gate, progress
            )
            gate(
                name + ":all_original_R3_equations",
                np.max(abs(op.attachment_action(core, vector) - energy * vector)),
                1e-10,
            )
            states = {int(2 * j): vector}
            state, m = vector, j
            while m > -j:
                state = angular.ladder(state, -1) / np.sqrt((j + m) * (j - m + 1))
                m -= 1
                states[int(2 * m)] = state
            gate(name + ":all_magnetic_partners", len(states) == int(2 * j + 1))
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
            np.savez(
                out / (name + "_root.npz"),
                energy=energy,
                vector=vector,
                indices=indices,
            )
            targets[name] = dict(
                EA_Ha=energy,
                EA_cm1=energy * HARTREE_CM1,
                total69_relative_Ha=core.total_energy_Ha + energy,
                total69_absolute_Ha=ledger["absolute_energy_origin_Ha"]
                + core.total_energy_Ha
                + energy,
                bare_R1_weight=history[-1]["bare_R1_weight"],
                magnetic_partners=len(states),
                dimension=len(indices),
                matvec_calls=len(history),
                scope="canonical projectedR norm, not physical exp(T) norm",
            )
            save(out / "target_progress.json", targets)
        np.savez(out / "all_target_partners.npz", **partners)
        report["targets"], report["EA_history"] = targets, progress
        report["lines_cm1"] = {
            name: (targets[name]["EA_Ha"] - targets["S"]["EA_Ha"]) * HARTREE_CM1
            for name in ("D3", "D5")
        }
        report["newp_line_changes_cm1"] = {
            name: report["lines_cm1"][name] - oldresult["lines_cm1"][name]
            for name in ("D3", "D5")
        }
        report["EA_changes_cm1"] = {
            name: (targets[name]["EA_Ha"] - oldresult["targets"][name]["EA_Ha"])
            * HARTREE_CM1
            for name in targets
        }
        gate(
            "source_and_inputs_unchanged",
            all(sha(ROOT / n) == h for n, h in source["files"].items())
            and all(sha(ROOT / n) == h for n, h in contract["inputs"].items()),
        )
        report["status"] = "PASS_COMPLETE_6P_NATIVE_CCSDT_R3"
    except Exception:
        report.update(status="FAILED", exception=traceback.format_exc())
        if last:
            np.savez(out / "last_core_evaluation.npz", **last)
        raise
    finally:
        signal.alarm(0)
        report["core_residual_calls"] = calls
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_RSS_MiB"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        save(out / "result.json", report)
        print("END", report["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else main()
