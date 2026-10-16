"""Bounded native high-l capability qualification; no paper EA claim."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/code/atomx").resolve()
THREADS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def prospective(contract):
    """Scalar mirror of bound make_physics_knots, before numerical imports."""
    g = contract["geometry"]
    points = [g["hi"] / contract["Z"]]
    ti = points[-1]
    while True:
        tj = ti * (1 + g["he"])
        if tj - ti > g["hmax"]:
            break
        ti = tj
        if ti > g["r_max_requested_Bohr"]:
            break
        points.append(ti)
    if ti < g["r_max_requested_Bohr"]:
        while ti + g["hmax"] < g["r_max_requested_Bohr"]:
            ti += g["hmax"]
            points.append(ti)
    points.append(points[-1] + g["hmax"])
    ns = len(points) + g["ks"] - 1
    return dict(
        ns=ns,
        nsp=ns - 1,
        nsq=ns,
        intervals=len(points),
        actual_outer_Bohr=points[-1],
        full_Dirac_dimension=2 * ns,
        constrained_Dirac_dimension=2 * ns - 7,
        six_raw_hS_bytes=6 * 2 * (2 * ns) ** 2 * 8,
        complete_one_electron_magnetic_states=66,
        complete_two_electron_g_magnetic_states=math.comb(18, 2),
        one_electron_CSFS=6,
        two_electron_g_CSFS=17,
        complete_CI_matrix_bytes=(6 * 6 + 2 * 17 * 17) * 8,
        radial_workspace_bytes_upper=8 * 1024**2,
        peak_memory_bytes_upper=1024**3,
        inputs_NPZ=0,
        outputs_NPZ=4,
        native_Dirac_spectrum_calls=6,
        CI_calls=3,
        native_hS_entries_upper=26,
        orbital_optimization_calls=0,
        SCF_calls=0,
    )


def source(contract):
    for path, expected in contract["source_files"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f"source binding changed: {path}")
    metadata = contract["current31_metadata"]
    if sha(ROOT / metadata["path"]) != metadata["sha256"]:
        raise RuntimeError("current31 context metadata changed")
    if (
        sha(HERE / contract["import_preflight"]["path"])
        != contract["import_preflight"]["sha256"]
    ):
        raise RuntimeError("import metadata origin binding changed")
    if contract["orbital_keys"] != [[5, 4], [5, -5], [6, 5], [6, -6], [7, 6], [7, -7]]:
        raise RuntimeError("complete g/h/i partners changed")
    expected = prospective(contract)
    if expected != contract["prospective"]:
        raise RuntimeError("prospective dimension/work budget changed")
    return expected


def execute(contract):
    if (HERE / "actual").exists():
        raise RuntimeError("closed evidence directory already exists; no retry")
    cap = contract["resources"]
    if any(os.environ.get(key) != "1" for key in THREADS):
        raise RuntimeError("serial numerical thread environment required")
    if os.sched_getaffinity(0) != {cap["CPU"]}:
        raise RuntimeError("single declared CPU required")
    load = os.getloadavg()
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if load[0] > cap["maximum_load1"] or available < cap["minimum_available_bytes"]:
        raise RuntimeError("host admission failed")
    resource.setrlimit(resource.RLIMIT_AS, (cap["AS_bytes"], cap["AS_bytes"]))
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("inner budget"))
    )
    signal.alarm(cap["inner_seconds"])
    actual = HERE / "actual"
    actual.mkdir()
    report = dict(
        status="RUNNING",
        started_UTC=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=dict(load=load, available_bytes=available, affinity=[cap["CPU"]]),
        counts={},
        checks=[],
        scope=contract["scope"],
    )
    start = time.monotonic()
    counts = report["counts"]

    def count(name, limit):
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > limit:
            raise RuntimeError(f"producer entry cap: {name}")

    def gate(name, observed, limit=None):
        if type(observed).__module__ == "numpy":
            observed = observed.item()
        ok = bool(observed) if limit is None else bool(observed <= limit)
        report["checks"].append(
            dict(name=name, observed=observed, limit=limit, PASS=ok)
        )
        if not ok:
            raise RuntimeError(f"qualification failed: {name}")

    def checkpoint():
        report["elapsed_seconds"] = time.monotonic() - start
        (actual / "partial_ledger.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )

    try:
        import numpy as np

        sys.path.insert(0, str(ROOT / "src"))
        import atomx.angular.csf_hamiltonian as angular
        import atomx.angular.mscheme as mscheme
        import atomx.numerics.bspline.dirac_assembly as dirac
        import atomx.physics.csf.matrix as matrix
        from atomx.config.csf import (
            enumerate_jj_orbital_space,
            get_shared_orbital_basis,
        )
        from atomx.constants import C_SQ
        from atomx.numerics.bspline.basis import prepare_bspline_data
        from atomx.numerics.bspline.evaluate import eval_P, eval_Q
        from atomx.numerics.bspline.knots import make_physics_knots
        from atomx.numerics.radial_metric import RadialMetric
        from atomx.physics.csf.radial_snapshot import CIRadialSnapshot
        from atomx.physics.dirac import hydrogenic_dirac_energy
        from atomx.radial_state import RadialState

        importlib.import_module("atomx.methods.ci")
        importlib.import_module("atomx.physics.csf.hamiltonian_snapshot")
        report["loaded_source_before_producers"] = {
            str(Path(module.__file__).resolve().relative_to(ROOT)): sha(
                Path(module.__file__).resolve()
            )
            for name, module in tuple(sys.modules.items())
            if name.startswith("atomx")
            and getattr(module, "__file__", None)
            and Path(module.__file__).suffix == ".py"
        }
        gate(
            "all_loaded_Atomx_source_is_bound_before_producers",
            all(
                contract["source_files"].get(path) == digest
                for path, digest in report["loaded_source_before_producers"].items()
            ),
        )

        gate(
            "excluded_EOM_not_imported_before_first_producer",
            all(name not in sys.modules for name in contract["excluded_imports"]),
        )
        source(contract)
        checkpoint()

        original_null = mscheme.null_space

        def null_space(*args, **kwargs):
            count("angular_nullspace_entries", 128)
            return original_null(*args, **kwargs)

        mscheme.null_space = null_space
        original_build = dirac.build_dirac_matrices

        def build(*args, **kwargs):
            count("native_hS_entries", 26)
            return original_build(*args, **kwargs)

        dirac.build_dirac_matrices = matrix.build_dirac_matrices = build
        original_eval = matrix._radial_evaluator
        radial_requests = []

        def evaluator(*args, **kwargs):
            count("native_radial_evaluator_entries", 2)
            native = original_eval(*args, **kwargs)

            def evaluate(key):
                count("complete_signed_radial_key_requests", 4000)
                radial_requests.append(key)
                return native(key)

            evaluate.cache_info = native.cache_info
            evaluate.pair_cache_info = native.pair_cache_info
            return evaluate

        matrix._radial_evaluator = evaluator
        g = contract["geometry"]
        t, ns = make_physics_knots(
            ks=g["ks"],
            Z=contract["Z"],
            nuclear="point",
            r_max=g["r_max_requested_Bohr"],
            hi=g["hi"],
            he=g["he"],
            hmax=g["hmax"],
        )
        gate("prospective_ns", ns == contract["prospective"]["ns"])
        gate(
            "prospective_outer_radius",
            float(t[-1]) == contract["prospective"]["actual_outer_Bohr"],
        )
        workspace = prepare_bspline_data(t, ns, g["ks"], kinetic_balance=True)
        gate("RKB_dimensions", (workspace["nsp"], workspace["nsq"]) == (ns - 1, ns))
        od, records, raw = {}, [], {}
        keep = np.r_[np.arange(1, ns - 3), np.arange(ns + 1, 2 * ns - 2)]
        metric = RadialMetric.bspline(workspace)
        for n, kappa in contract["orbital_keys"]:
            H, S = build(ns, kappa, contract["Z"], workspace, None)
            count("native_Dirac_spectrum_entries", 6)
            result = dirac.solve_dirac(H, S, ns, kappa, nsp=ns - 1, nsq=ns)
            ell = kappa if kappa > 0 else -kappa - 1
            P, Q, energy = result.get_orbital(n - ell - 1, tolerance=1e-10)
            vec = np.r_[P, Q]
            exact = hydrogenic_dirac_energy(n, kappa, contract["Z"])
            residual = float(np.linalg.norm((H @ vec - energy * (S @ vec))[keep]))
            norm = float(vec @ S @ vec)
            quadrature = float(
                metric.integrate(
                    (eval_P(P, workspace) ** 2 + eval_Q(Q, workspace) ** 2).ravel()
                )
            )
            gate(f"{n},{kappa}:point_Dirac_energy_Ha", abs(float(energy) - exact), 1e-9)
            gate(f"{n},{kappa}:constrained_generalized_residual", residual, 1e-8)
            gate(f"{n},{kappa}:S_norm", abs(norm - 1), 1e-10)
            gate(f"{n},{kappa}:native_metric_norm", abs(quadrature - norm), 1e-10)
            gate(f"{n},{kappa}:electron_bound_branch", -C_SQ < energy < 0)
            gate(
                f"{n},{kappa}:exact_removed_boundary",
                bool(np.all(vec[np.setdiff1d(np.arange(2 * ns), keep)] == 0)),
            )
            od[n, kappa] = P, Q
            raw[f"h_{kappa}"], raw[f"S_{kappa}"] = H, S
            records.append(
                dict(
                    n=n,
                    kappa=kappa,
                    l=ell,
                    J2=2 * abs(kappa) - 1,
                    energy_Ha=float(energy),
                    analytic_Ha=exact,
                    error_Ha=float(energy) - exact,
                    residual=residual,
                    norm=norm,
                    quadrature_norm=quadrature,
                    radial_root_index=n - ell - 1,
                    complete_electron_branch_states=len(result.eigenvalues),
                )
            )
            report["orbitals"] = records
            checkpoint()
        report["orbitals"] = records
        report["splittings"] = []
        for lo, hi in zip(records[::2], records[1::2], strict=True):
            difference = hi["energy_Ha"] - lo["energy_Ha"]
            anchor = hi["analytic_Ha"] - lo["analytic_Ha"]
            gate(f"l={lo['l']}:paired_split_Ha", abs(difference - anchor), 2e-9)
            report["splittings"].append(
                dict(l=lo["l"], numerical_Ha=difference, analytic_Ha=anchor)
            )
        snapshot = CIRadialSnapshot(
            od, workspace, contract["Z"], ns, pair_potential_cache_size=128
        )
        state = RadialState(
            "bspline",
            "dirac",
            RadialMetric.bspline(snapshot._workspace),
            snapshot._workspace,
            snapshot._orbitals,
            (),
            0,
        )
        gate("RadialState_all_six_partners", set(state.orbitals) == set(od))
        gate(
            "RadialState_readonly_waves",
            all(not v.flags.writeable for pq in state.orbitals.values() for v in pq),
        )
        gate(
            "RadialState_native_metric",
            bool(np.array_equal(state.metric.weights, metric.weights)),
        )
        outputs = dict(raw)
        outputs.update(
            P=np.stack([od[tuple(k)][0] for k in contract["orbital_keys"]]),
            Q=np.stack([od[tuple(k)][1] for k in contract["orbital_keys"]]),
            orbital_keys=np.asarray(contract["orbital_keys"]),
            **{f"workspace_{key}": value for key, value in workspace.items()},
        )
        owners = []

        def ci(name, snap, keys, electrons, expected_csfs, expected_magnetic):
            csfs = enumerate_jj_orbital_space(keys, electrons, magnetic_state_cap=153)
            gate(name + ":complete_CSF_count", len(csfs) == expected_csfs)
            gate(
                name + ":complete_multiplet_dimension",
                sum(c.total_J2 + 1 for c in csfs) == expected_magnetic,
            )
            basis = get_shared_orbital_basis(csfs)
            blocks = {}
            for c in csfs:
                sig = angular.csf_signature(c)
                blocks.setdefault((c.total_J2, c.parity), []).append(sig)
            table_records = []
            for (j2, parity), signatures in blocks.items():
                patterns = tuple(dict.fromkeys(s.pattern for s in signatures))
                structure = angular.block_terms(
                    patterns, j2, 153, radial_symmetry=False
                )
                table_records.append(
                    dict(
                        J2=j2,
                        parity=parity,
                        raw_determinants=len(structure.dets),
                        radial_keys=len(structure.keys),
                        nnz=structure.table.nnz,
                        signed_pairs=len(structure.pairs),
                    )
                )
                gate(name + f":J2={j2}:raw_determinant_cap", len(structure.dets), 153)
                gate(name + f":J2={j2}:angular_nnz_cap", structure.table.nnz, 100000)
            count("complete_native_CI_entries", 3)
            result = snap.solve(
                csfs,
                basis,
                solver="dense",
                n_states=None,
                max_determinants=153,
                retain_hamiltonian=True,
            )
            H = result.hamiltonian_for(
                csfs,
                snap._orbitals,
                basis,
                snap._workspace,
                contract["Z"],
                ns,
                max_determinants=153,
                include_breit=False,
            )
            C, E = result.mixing_coefficients, result.state_energies
            gate(
                name + ":whole_spectrum_shape",
                C.shape == (expected_csfs, expected_csfs),
            )
            gate(
                name + ":whole_H_residual", float(np.max(np.abs(H @ C - C * E))), 1e-10
            )
            gate(
                name + ":whole_Gram",
                float(np.max(np.abs(C.T @ C - np.eye(expected_csfs)))),
                1e-10,
            )
            gate(name + ":complete_trace", abs(float(np.sum(E) - np.trace(H))), 1e-10)
            gate(name + ":immutable_H", not H.flags.writeable)
            gate(
                name + ":operator_binding",
                result.hamiltonian_snapshot.hamiltonian_terms == ("dirac-coulomb",),
            )
            outputs[name + "_H"], outputs[name + "_C"], outputs[name + "_E"] = H, C, E
            owners.append(
                dict(
                    name=name,
                    owner=result.hamiltonian_snapshot.identity_payload(),
                    input_sha256=result.hamiltonian_snapshot.input_sha256,
                    matrix_sha256=result.hamiltonian_snapshot.matrix_sha256,
                    ordered_CSFs=[
                        dict(signature=angular.csf_signature(c).key, parity=c.parity)
                        for c in csfs
                    ],
                    angular=table_records,
                    provenance=result.provenance,
                )
            )
            report["owners"] = owners
            np.savez_compressed(actual / "partial_complete_proof.npz", **outputs)
            checkpoint()
            return csfs, H, E

        keys = tuple(map(tuple, contract["orbital_keys"]))
        _, _, E1 = ci("one_electron_ghi", snapshot, keys, 1, 6, 66)
        gate(
            "N1_complete_sorted_analytic_energies",
            float(np.max(np.abs(E1 - np.sort([r["analytic_Ha"] for r in records])))),
            1e-9,
        )
        gkeys = ((5, 4), (5, -5))
        csfs2, H2, E2 = ci("two_electron_g", snapshot, gkeys, 2, 17, 153)
        gate(
            "g_Coulomb_nonzero_offdiagonal",
            bool(np.max(np.abs(H2 - np.diag(np.diag(H2)))) > 1e-6),
        )
        flipped = dict(od)
        flipped[5, -5] = tuple(-v for v in od[5, -5])
        second = CIRadialSnapshot(
            flipped, workspace, contract["Z"], ns, pair_potential_cache_size=128
        )
        _, Hflip, Eflip = ci("two_electron_g_phase", second, gkeys, 2, 17, 153)
        signs = np.asarray(
            [
                (-1)
                ** sum(s.occupation for s in c.subshells if (s.n, s.kappa) == (5, -5))
                for c in csfs2
            ]
        )
        gate(
            "complete_signed_orbital_phase_covariance",
            float(np.max(np.abs(Hflip - signs[:, None] * H2 * signs))),
            1e-12,
        )
        gate("complete_signed_phase_spectrum", float(np.max(np.abs(Eflip - E2))), 1e-12)
        rkeys = [key for key in radial_requests if key[0] == "r"]
        gate("g_Coulomb_rank8_is_evaluated", any(key[1] == 8 for key in rkeys))
        report.update(
            owners=owners,
            radial_cache=[snapshot.diagnostics(), second.diagnostics()],
            radial_keys=sorted(set(map(repr, radial_requests))),
            native_RadialState_scope="owned six-wave capability bank; no invented SCF result/occupied density",
        )
        np.savez_compressed(actual / "complete_proof.npz", **outputs)
        report["archive"] = dict(
            path="actual/complete_proof.npz",
            sha256=sha(actual / "complete_proof.npz"),
            arrays=len(outputs),
            decoded_bytes=sum(np.asarray(v).nbytes for v in outputs.values()),
        )
        source(contract)
        gate(
            "excluded_EOM_not_imported_after_producers",
            all(name not in sys.modules for name in contract["excluded_imports"]),
        )
        report["loaded_source_files"] = {
            str(Path(module.__file__).resolve().relative_to(ROOT)): sha(
                Path(module.__file__).resolve()
            )
            for name, module in tuple(sys.modules.items())
            if name.startswith("atomx")
            and getattr(module, "__file__", None)
            and Path(module.__file__).suffix == ".py"
        }
        gate(
            "all_loaded_Atomx_source_is_bound",
            all(
                contract["source_files"].get(path) == digest
                for path, digest in report["loaded_source_files"].items()
            ),
        )
        gate("bound_current_source_unchanged", True)
        report["status"] = "PASS_native_high_l_capability_only"
    except BaseException as error:
        report.update(
            status="FAILED_preserved_no_retry",
            failure=dict(type=type(error).__name__, message=str(error)),
        )
        raise
    finally:
        signal.alarm(0)
        report.update(
            elapsed_seconds=time.monotonic() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            interpreter=sys.executable,
            driver_sha256=sha(Path(__file__)),
            contract_sha256=sha(HERE / "contract.json"),
        )
        (actual / "result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    budget = source(contract)
    if args.execute:
        execute(contract)
    else:
        write(
            "preparation.json",
            dict(
                status="SOURCE_ready",
                source_files=len(contract["source_files"]),
                prospective=budget,
                numerical_imports=0,
                numerical_calls=0,
                NPZ_inputs=0,
                science_executed=False,
            ),
        )
