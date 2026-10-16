"""New predeclared target-seeded Davidson trial after the preserved SR cap.

This is an experiment driver, not an additional production eigensolver API.
"""

import argparse
import json
import os
import resource
import signal
import time
import traceback
from pathlib import Path

import numpy as np
from angular import AngularCoordinates, act
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from solve_and_attach import HERE, OLD, ROOT, load_operator, save, sha

from atomx.constants import HARTREE_CM1
from atomx.methods.tensor_coupled_cluster import TensorCoreResult

TARGETS = [("S", (6, -1), 0.5), ("D3", (5, 2), 1.5), ("D5", (5, -3), 2.5)]


def prepare():
    inputs = [
        Path(__file__),
        HERE / "angular.py",
        HERE / "targeted_davidson.py",
        HERE / "davidson_contract.json",
        HERE / "actual002/result.json",
        HERE / "solve_and_attach.py",
        HERE / "after_source.json",
        HERE / "operator.json",
        HERE / "integrals.npz",
        HERE / "actual001/core.npz",
        HERE / "actual001/result.json",
        OLD / "actual001/operator.json",
        OLD / "actual001/integrals.npz",
        OLD / "actual001/CCSDT_core.npz",
        OLD / "actual001/result.json",
    ] + [OLD / "actual001" / f"M{m}_CCSDT_R3.npz" for m in (1, 3, 5)]
    save(
        HERE / "davidson_J_contract.json",
        dict(
            hypothesis="The unprojected correction basis amplified roundoff into otherJ at small D3 iteration7, before a large solve. An orthogonal highest-weight projector ker(Jplus) removes only otherJ irreps and stabilizes Davidson without changing the original R3 equations or target states.",
            previous="actual001 SR resource failure and actual002 small-qualification J2 failure are preserved; neither provided an accepted large spectrum",
            fixed=dict(
                source="same after_source epoch",
                core="exact saved converged18e amplitudes, independently reapply allSDT equations",
                Hamiltonian="unchanged30spinor DC / common68e origin",
                spaces="complete target-J irrep inside each highest-M/even R3 sector via P=I-A.T(AA.T)^-1 A, A=Jplus; no amplitude screening",
                targets="unique normalized bareR1 weight >=0.5 for prescribed6s/5d3/5d5; no experimental energies",
                algorithm="real nonsymmetric Davidson, one bareR1 seed per target, twice modified Gram-Schmidt, complete Fock-energy denominators, no clipping/shift, unrestarted subspace max48; signed sparse highest-weight projection with SPD CG rtol1e-14, atol0, maxiter100 and unchanged J2 gate",
                output="all12 magnetic partners by signedJ ladders; complete original equations reapplied",
            ),
            distinguishing="Small old16spinor target energies must equal the independently qualified stored dense eigenvalues. Large trial passes only if all original SDT/R3 equations and J2 gates pass; cap or root-identity failure remains failure.",
            gates=dict(
                core_residual_Ha=1e-10,
                EA_residual_Ha=1e-10,
                small_dense_energy_Ha=1e-9,
                J2=1e-7,
                canonical_R_norm=1e-8,
                target_bare_R1_weight_min=0.5,
            ),
            resources=dict(
                cpu=27,
                threads=1,
                AS_bytes=12 * 1024**3,
                wall_seconds=600,
                workspace_MiB=4096,
                subspace_max=48,
                large_solver_actions_max=144,
                direct_large_verification_actions=15,
                small_solver_actions_max=144,
                load1_max=8,
                available_min_bytes=32 * 1024**3,
            ),
            stopping="First failed identity/equation/resource gate stops. No automatic retry or altered rank, tolerance, target, ncv or denominator regularization. This new numerical contract replaces no archived failure.",
            source_map_sha256=json.loads((HERE / "after_source.json").read_text())[
                "source_map_sha256"
            ],
            inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        ),
    )
    print("TARGETED_DAVIDSON_PREPARED", flush=True)


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
        save(HERE / "actual003/progress.json", progress)
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
    contract = json.loads((HERE / "davidson_J_contract.json").read_text())
    source = json.loads((HERE / "after_source.json").read_text())
    assert all(sha(ROOT / p) == h for p, h in source["files"].items())
    assert all(sha(ROOT / p) == h for p, h in contract["inputs"].items())
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
        raise TimeoutError("declared600second targeted pilot limit")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(600)
    out = HERE / "actual003"
    out.mkdir(exist_ok=False)
    report = dict(
        status="RUNNING",
        controls=[],
        contract_sha256=sha(HERE / "davidson_J_contract.json"),
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
        op, ledger = load_operator(HERE, 18)
        with np.load(HERE / "actual001/core.npz") as z:
            amplitudes = z["amplitudes"]
        residual = op.core_residual(amplitudes)
        gate("all189834_saved_core_equations", np.max(abs(residual)), 1e-10)
        e = op.core_energy(amplitudes)
        core = TensorCoreResult(
            amplitudes, e, e, float(np.max(abs(residual))), 0, op.operator_sha256
        )
        report["core"] = json.loads((HERE / "actual001/result.json").read_text())[
            "core"
        ]
        gate("saved_core_energy", abs(e - report["core"]["energy_relative_Ha"]), 1e-12)
        oldop, _ = load_operator(OLD / "actual001", 4)
        with np.load(OLD / "actual001/CCSDT_core.npz") as z:
            oldt = z["amplitudes"]
        oe = oldop.core_energy(oldt)
        oldcore = TensorCoreResult(
            oldt,
            oe,
            oe,
            float(np.max(abs(oldop.core_residual(oldt)))),
            0,
            oldop.operator_sha256,
        )
        oldangular = AngularCoordinates(oldop, True)
        small_progress = {}
        for name, key, j in TARGETS:
            energy, vector, indices, _ = solve_target(
                oldop, oldcore, oldangular, name, key, j, gate, small_progress
            )
            with np.load(OLD / "actual001" / f"M{int(2 * j)}_CCSDT_R3.npz") as z:
                gate(
                    "small:" + name + ":same_complete_coordinates",
                    np.array_equal(indices, z["indices"]),
                )
                exact = np.linalg.eigvals(z["projected_H"])
                gate(
                    "small:" + name + ":stored_dense_root",
                    np.min(abs(exact - energy)),
                    1e-9,
                )
            gate(
                "small:" + name + ":all_original_R3_equations",
                np.max(abs(oldop.attachment_action(oldcore, vector) - energy * vector)),
                1e-10,
            )
        report["small_qualification"] = small_progress
        print("SMALL_DAVIDSON_QUALIFICATION_PASS", flush=True)
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
                total69_relative_Ha=e + energy,
                total69_absolute_Ha=ledger["absolute_energy_origin_Ha"] + e + energy,
                bare_R1_weight=history[-1]["bare_R1_weight"],
                magnetic_partners=len(states),
                dimension=len(indices),
                matvec_calls=len(history),
                scope="canonical projectedR norm only, not physical exp(T) norm",
            )
        np.savez(out / "all_target_partners.npz", **partners)
        report["targets"], report["EA_history"] = targets, progress
        report["lines_cm1"] = {
            name: (targets[name]["EA_Ha"] - targets["S"]["EA_Ha"]) * HARTREE_CM1
            for name in ("D3", "D5")
        }
        oldlines = json.loads((OLD / "actual001/result.json").read_text())["lines_cm1"][
            "CCSDT_R3"
        ]
        report["4f_opening_line_changes_cm1"] = {
            name: report["lines_cm1"][name] - oldlines[name] for name in ("D3", "D5")
        }
        gate(
            "source_unchanged",
            all(sha(ROOT / p) == h for p, h in source["files"].items()),
        )
        gate(
            "all_inputs_unchanged",
            all(sha(ROOT / p) == h for p, h in contract["inputs"].items()),
        )
        report["status"] = "PASS_FULL4F_TARGETED_DAVIDSON"
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
        print("END", report["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else main()
