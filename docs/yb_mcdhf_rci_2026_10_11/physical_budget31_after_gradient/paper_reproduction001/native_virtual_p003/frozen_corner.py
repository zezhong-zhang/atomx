"""Same-DC frozen4f companion closes the2x2 core/virtual correlation panel."""

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
from prepare_virtuals import HERE, OLD, ROOT, AngularCoordinates, save, sha
from solve import TARGETS, diagonal, highest_weight_projector

from atomx.constants import HARTREE_CM1
from atomx.methods.tensor_coupled_cluster import CCSDTTensorOperator
from atomx.physics.csf.spinor_integrals import SpinorIntegrals


def prepare():
    original = HERE.parent / "native_tensor_ccsdt001/actual001"
    inputs = [
        Path(__file__),
        HERE / "solve.py",
        HERE / "diis_core.py",
        HERE / "prepare_virtuals.py",
        HERE / "operator.json",
        HERE / "integrals.npz",
        HERE / "source.json",
        OLD / "angular.py",
        OLD / "actual003/result.json",
        original / "operator.json",
        original / "integrals.npz",
        original / "result.json",
    ]
    save(
        HERE / "corner_contract.json",
        dict(
            hypothesis="The6p virtual increment can depend on whether4f is correlated. Complete the four-model panel with the same6p bank but frozen4f; compare differential line increments only after both roots/equations qualify.",
            fixed="Same36-spinor full operator projected by exact frozen normal ordering: freeze14new4f magnetic occupieds, leaving4correlated/64frozen/18virtual/22spinors; same original68e HF origin and nativeDC radial bank. No integral, state, rank or tolerance fitting.",
            equations="CompleteSDT/R3, exact coreM/parity, unshifted DIIS history6/max60; same prescribedR1>=.5/J2/highest-weight Davidson/all12 partner gates. Start core from zero; not a controlled comparison of nonlinear solve timing.",
            gates=dict(
                operator_restriction_Ha=1e-10, core_Ha=1e-10, EA_Ha=1e-10, J2=1e-7
            ),
            resources=dict(
                cpu=26,
                threads=1,
                AS_bytes=8 * 1024**3,
                workspace_MiB=512,
                wall_seconds=180,
                load1_max=8,
                available_min_bytes=32 * 1024**3,
            ),
            stopping="First failure stops, preserves partials; no automatic retry or altered matching/tolerance/model.",
            inputs={str(q.relative_to(ROOT)): sha(q) for q in inputs},
        ),
    )
    print("FROZEN4F_6P_CORNER_PREPARED", flush=True)


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
        save(HERE / "corner001/EA_progress.json", progress)
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
    c = json.loads((HERE / "corner_contract.json").read_text())
    source = json.loads((HERE / "source.json").read_text())
    assert all(sha(ROOT / n) == h for n, h in c["inputs"].items())
    assert all(sha(ROOT / n) == h for n, h in source["files"].items())
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
    os.sched_setaffinity(0, {26})
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))

    def alarm(*_):
        raise TimeoutError("predeclared180second frozen4f6p companion cap")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(180)
    out = HERE / "corner001"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(
        status="RUNNING",
        controls=[],
        contract_sha256=sha(HERE / "corner_contract.json"),
        source_map_sha256=source["source_map_sha256"],
    )

    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        report["controls"].append(
            dict(name=name, measured=measured, cap=cap, passed=bool(passed))
        )
        if not passed:
            raise AssertionError((name, measured, cap))

    try:
        from prepare_virtuals import load_operator

        full, ledger = load_operator(HERE)
        picked = list(range(4)) + list(range(18, 36))
        frozen = range(4, 18)
        n = len(picked)
        with np.load(HERE / "integrals.npz") as z:
            h0 = z["h"]
            h = h0[np.ix_(picked, picked)].copy()
            fock = z["fock"][np.ix_(picked, picked)]
        for i, p in enumerate(picked):
            for j, q in enumerate(picked):
                h[i, j] += sum(full.g[p, a, q, a] for a in frozen)
        shifted = (
            ledger["shifted_frozen_energy_Ha"]
            + float(np.trace(h0[4:18, 4:18]).real)
            + 0.5 * sum(full.g[a, b, a, b] for a in frozen for b in frozen)
        )
        packed = np.array(
            [
                full.g[picked[p], picked[q], picked[r], picked[s]]
                for p in range(n - 1)
                for q in range(p + 1, n)
                for r in range(p, n - 1)
                for s in range(r + 1, n)
            ]
        )
        identity = dict(
            parent_operator_sha256=full.operator_sha256,
            frozen4f14=True,
            correlated=4,
            frozen_electrons=64,
            virtual_spinors=18,
            absolute_energy_origin_Ha=ledger["absolute_energy_origin_Ha"],
        )
        labels = tuple(full.labels[i] for i in picked)
        obj = SpinorIntegrals(labels, 4, h, fock, packed, shifted, 0.0, identity)
        op = CCSDTTensorOperator(obj, memory_MiB=512, projection_storage="canonical")
        original = HERE.parent / "native_tensor_ccsdt001/actual001"
        with np.load(original / "integrals.npz") as z:
            gate("same_all_old16_Fock", np.max(abs(fock[:16, :16] - z["fock"])), 1e-10)
            gate("same_all_old16_one_body", np.max(abs(h[:16, :16] - z["h"])), 1e-10)
        oldledger = json.loads((original / "operator.json").read_text())
        gate("same_old16_labels", labels[:16] == tuple(map(tuple, oldledger["labels"])))
        gate(
            "same_old_frozen_constant",
            abs(shifted - oldledger["shifted_frozen_energy_Ha"]),
            1e-10,
        )
        gate(
            "same_old_reference_origin",
            abs(
                ledger["absolute_energy_origin_Ha"]
                - oldledger["absolute_energy_origin_Ha"]
            ),
            1e-10,
        )
        gate(
            "all_new_retained_ERI",
            np.max(abs(op.g - full.g[np.ix_(picked, picked, picked, picked)])),
            1e-10,
        )
        save(
            out / "operator.json",
            dict(
                labels=labels,
                identity=identity,
                shifted_frozen_energy_Ha=shifted,
                absolute_energy_origin_Ha=ledger["absolute_energy_origin_Ha"],
            ),
        )
        np.savez(out / "integrals.npz", h=h, fock=fock, g=packed)
        core, history = solve_diis(op)
        gate(
            "all4254_core_equations",
            np.max(abs(op.core_residual(core.amplitudes))),
            1e-10,
        )
        gate(
            "core_Jplus",
            np.linalg.norm(AngularCoordinates(op, False).ladder(core.amplitudes, 1)),
            1e-7,
        )
        np.savez(out / "core.npz", amplitudes=core.amplitudes)
        report["core"] = dict(
            energy_relative_Ha=core.total_energy_Ha,
            residual_Ha=core.amplitude_residual_Ha,
            evaluations=core.evaluations,
        )
        report["DIIS_history"] = history
        angular = AngularCoordinates(op, True)
        progress, targets, partners = {}, {}, {}
        for name, key, j in TARGETS:
            energy, vector, indices, hist = solve_target(
                op, core, angular, name, key, j, gate, progress
            )
            states = {int(2 * j): vector}
            state, m = vector, j
            while m > -j:
                state = angular.ladder(state, -1) / np.sqrt((j + m) * (j - m + 1))
                m -= 1
                states[int(2 * m)] = state
            for m2, state in states.items():
                gate(name + f":M{m2}:norm", abs(np.linalg.norm(state) - 1), 1e-8)
                gate(
                    name + f":M{m2}:J2",
                    np.linalg.norm(
                        angular.j_squared(state, m2 / 2) - j * (j + 1) * state
                    ),
                    1e-7,
                )
                gate(
                    name + f":M{m2}:all5526_R3_equations",
                    np.max(abs(op.attachment_action(core, state) - energy * state)),
                    1e-10,
                )
                partners[name + f"_M{m2}"] = state
            targets[name] = dict(
                EA_Ha=energy,
                total69_relative_Ha=core.total_energy_Ha + energy,
                bare_R1_weight=hist[-1]["bare_R1_weight"],
                magnetic_partners=len(states),
                matvec_calls=len(hist),
            )
        np.savez(out / "all_target_partners.npz", **partners)
        report["targets"] = targets
        report["EA_history"] = progress
        report["lines_cm1"] = {
            name: (targets[name]["EA_Ha"] - targets["S"]["EA_Ha"]) * HARTREE_CM1
            for name in ("D3", "D5")
        }
        baseline = json.loads((original / "result.json").read_text())["lines_cm1"][
            "CCSDT_R3"
        ]
        report["newp_line_changes_cm1"] = {
            name: report["lines_cm1"][name] - baseline[name] for name in ("D3", "D5")
        }
        gate(
            "all_inputs_source_unchanged",
            all(sha(ROOT / n) == h for n, h in c["inputs"].items())
            and all(sha(ROOT / n) == h for n, h in source["files"].items()),
        )
        report["status"] = "PASS_FROZEN4F_COMPLETE_6P_CORNER"
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
        print(report["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    a = parser.parse_args()
    prepare() if a.prepare else main()
