"""Private inner-order anchors and complete fixed-CI five-state correction.

Stdlib-only preparation; numerical work requires an exact one-shot root seal.
No H/CI/gradient/optimizer. Old and new radial evaluators run in fresh processes.
"""
import argparse
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write(path, payload):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(plain(payload), indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def key(value):
    return tuple(key(v) if isinstance(v, list) else v for v in value)


def admit(execution):
    contract = json.loads((HERE / "contract.json").read_text())
    expected = {"execution_authorized_by_root": True, "driver_sha256": sha(__file__),
                "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(HERE / "preparation_verification.json"),
                "ordinary_source_map_sha256": contract["ordinary"]["map"],
                "candidate_source_map_sha256": contract["candidate"]["map"],
                "CPU": 31, "wall_seconds": 300, "memory_GiB": 3.95,
                "max_H": 0, "max_CI": 0, "max_gradient": 0, "max_optimizer": 0,
                "purpose": "positive_multipole_inner_order_fixed_CI_discriminator"}
    if any(execution.get(k) != v for k, v in expected.items()):
        raise ValueError("Missing exact root seal")
    proof = json.loads((HERE / "preparation_verification.json").read_text())
    if not (proof["all_passed"] is True and proof["driver_sha256"] == expected["driver_sha256"]
            and proof["contract_sha256"] == expected["contract_sha256"]):
        raise ValueError("Preparation admission mismatch")
    for row in contract["inputs"].values():
        if sha(REPO / row["path"]) != row["sha256"]:
            raise ValueError("Changed input: " + row["path"])
    prior = json.loads((REPO / contract["inputs"]["normalized_control_result"]["path"]).read_text())
    if not (prior["all_numeric_controls_passed"] is True
            and prior["samepoint_full_normalized_gradient_control_completed"] is True
            and prior["source_map_sha256"] == contract["ordinary"]["map"]):
        raise ValueError("Incomplete original same-point signed-table qualification")
    for variant in ("ordinary", "candidate"):
        declaration = contract[variant]
        identity = json.loads((REPO / declaration["identity"]).read_text())
        if identity["source_map_sha256"] != declaration["map"]:
            raise ValueError("Source map changed")
        for name, digest in identity["all_atomx_python_sha256"].items():
            if sha(REPO / declaration["archive"] / name.removeprefix("src/")) != digest:
                raise ValueError("Frozen source changed")
    if any(os.environ.get(v) != "1" for v in contract["resources"]["thread_variables"]):
        raise ValueError("Single-thread environment required")
    if 31 not in os.sched_getaffinity(0):
        raise ValueError("CPU31 unavailable")
    return contract


def worker(phase, execution):
    contract = admit(execution)
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95 * 2**30), int(3.95 * 2**30)))
    os.sched_setaffinity(0, {31})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Shared300s budget")))
    signal.alarm(max(1, int(execution["remaining_seconds"])))
    variant = "candidate" if phase == "anchors" else phase
    source = REPO / contract[variant]["archive"]
    if any(n == "atomx" or n.startswith("atomx.") for n in sys.modules):
        raise ValueError("Physical import before source admission")
    sys.path.insert(0, str(source))
    import numba
    import numpy as np
    import scipy

    import atomx.numerics.bspline.convol as convol
    import atomx.numerics.bspline.dirac_assembly as dirac
    import atomx.physics.csf.matrix as matrix
    from atomx.numerics.bspline import basis
    from atomx.physics.csf.radial_snapshot import CIRadialSnapshot
    from atomx.potentials.nuclear import make_nuclear_potential

    output = HERE / "actual" / phase
    output.mkdir(exist_ok=False)
    started, completed, controls = {}, {}, []
    clock = time.monotonic()
    raw_hS = []
    call_limits = contract["anchor_calls"] if phase == "anchors" else contract["radial_calls_per_variant"]

    def flush(status):
        write(output / "progress.json", {"phase": status, "started": started,
              "completed": completed, "controls": controls,
              "elapsed_uncontrolled_s": time.monotonic() - clock})

    def gate(name, condition, error=None, limit=None):
        passed = bool(condition)
        row = {"name": name, "passed": passed}
        if error is not None:
            row.update(error=float(error), limit=float(limit))
        controls.append(row)
        if not passed:
            flush(name)
            raise ValueError("Failed gate: " + name)

    def close(name, a, b, atol=0., rtol=0.):
        delta, bound = abs(np.asarray(a) - np.asarray(b)), atol + rtol * abs(np.asarray(b))
        gate(name, np.all(np.isfinite(delta)) and np.all(delta <= bound),
             np.max(delta, initial=0.), np.max(bound, initial=0.))

    def wrap(name, function, cap=None, capture=False):
        def call(*args, **kwargs):
            started[name] = started.get(name, 0) + 1
            if cap is not None and started[name] > cap:
                flush("cap_" + name)
                raise RuntimeError("Call cap: " + name)
            result = function(*args, **kwargs)
            completed[name] = completed.get(name, 0) + 1
            if capture:
                raw_hS.append((int(args[1]), result[0].copy(), result[1].copy()))
            return result
        return call

    matrix.build_ci_matrix = wrap("H", matrix.build_ci_matrix, 0)
    dirac.solve_dirac = wrap("Dirac_solve", dirac.solve_dirac, 0)
    matrix.build_dirac_matrices = wrap("hS", dirac.build_dirac_matrices, 0 if phase == "anchors" else 7, True)
    # Both public evaluator aliases and independent anchor aliases forward to
    # the same unmodified operations; no numerical replacement or term filter.
    originals = {name: getattr(convol, name) for name in
                 ("_density_sym", "_convol_case1_t", "_convol_case1_semiseparable_t", "_sum_amb_banded")}
    labels = {"_density_sym": "density_band", "_convol_case1_t": "direct_full_bank",
              "_convol_case1_semiseparable_t": "direct_semiseparable", "_sum_amb_banded": "band_scalar"}
    for name, original in originals.items():
        function = wrap(labels[name], original, call_limits[labels[name]])
        setattr(convol, name, function)
        setattr(matrix, name, function)
    convol.ConvolCache.get_rkb_t = wrap("bank_get", convol.ConvolCache.get_rkb_t, call_limits["bank_get"])
    convol.ConvolCache.get_coulomb_factors = wrap("factor_get", convol.ConvolCache.get_coulomb_factors, call_limits["factor_get"])
    matrix._bspline_pair_potential = wrap("pair_potential", matrix._bspline_pair_potential, call_limits["pair_potential"])
    np.linalg.eigvalsh = wrap("Gauss_setup_eigvalsh", np.linalg.eigvalsh, 3 if phase == "anchors" else 0)
    np.linalg.eigh = wrap("numpy_eigh", np.linalg.eigh, 0)
    scipy.linalg.eigh = wrap("scipy_eigh", scipy.linalg.eigh, 0)

    def load(role):
        path = REPO / contract["inputs"][role]["path"]
        if path.suffix != ".npz":
            return json.loads(path.read_text())
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    try:
        if phase == "anchors":
            basis.prepare_bspline_data = wrap("basis_setup", basis.prepare_bspline_data, 3)
            for case, (ks, balanced, p_power, q_power, ranks) in enumerate(
                    ((4, False, 1, 1, (0, 1, 6)), (6, False, 4, 5, (6,)), (6, True, 4, 5, (6,)))):
                radius = .7
                knots = np.r_[np.zeros(ks), np.full(ks, radius)]
                data = basis.prepare_bspline_data(knots, ks, ks, n_gauss=16, kinetic_balance=balanced)
                orders = data["ksp"], data["ksq"]
                cache = convol.ConvolCache(data)
                cache.release_shared_banks()
                def polynomial(order, power, n_splines=ks, radius=radius):
                    values = np.zeros(n_splines)
                    for i in range(power, order):
                        values[i] = radius**power * math.comb(i, power) / math.comb(order-1, power)
                    return values
                p, q = polynomial(orders[0], p_power), polynomial(orders[1], q_power)
                gate(f"exact_P_unused_tail_{case}", np.all(p[data["nsp"]:] == 0.))
                waves = tuple((scale * p, scale * q) for scale in (1., -2., 3., 4.))
                np.savez_compressed(output / f"geometry_{case}.npz", **data,
                                    P=np.array([w[0] for w in waves]), Q=np.array([w[1] for w in waves]))
                target = tuple(convol._density_sym(waves[0][j], waves[1][j], ks, orders[j]) for j in range(2))
                source_density = tuple(convol._density_sym(waves[2][j], waves[3][j], ks, orders[j]) for j in range(2))
                for rank in ranks:
                    reference_parts = [np.zeros((ks, order)) for order in orders]
                    for left in range(2):
                        for right in range(2):
                            labels0 = ("p", "q")
                            component = labels0[left] + labels0[right] + labels0[left] + labels0[right]
                            bank, extent, _ = cache.get_rkb_t(rank, component)
                            factors = cache.get_coulomb_factors(rank, component)
                            potential = convol._convol_case1_t(bank, source_density[right], extent, orders[left], orders[right])
                            separated = convol._convol_case1_semiseparable_t(*factors, source_density[right], ks, orders[left], orders[right])
                            close(f"full_vs_factor_{case}_{rank}_{component}", separated, potential, 1e-12, 2e-12)
                            a, b = 2 * (p_power, q_power)[left], 2 * (p_power, q_power)[right]
                            expected = -24 * radius**(a+b+1)/(a+b+1) * (1/(a+rank+1)+1/(b+rank+1))
                            scalar = convol._sum_amb_banded(target[left], potential, ks, orders[left])
                            scalar2 = convol._sum_amb_banded(target[left], separated, ks, orders[left])
                            close(f"analytic_{case}_{rank}_{component}", scalar, expected, 0., 2e-13)
                            close(f"analytic_factor_{case}_{rank}_{component}", scalar2, expected, 0., 2e-13)
                            reference_parts[left] += potential
                    actual = matrix._bspline_pair_potential(cache, ks, orders, rank, source_density)
                    for left in range(2):
                        close(f"actual_route_{case}_{rank}_{left}", actual[left], reference_parts[left])
                        scalar = convol._sum_amb_banded(target[left], actual[left], ks, orders[left])
                        expected = sum(-24 * radius**(2*(p_power,q_power)[left]+2*power+1)/
                            (2*(p_power,q_power)[left]+2*power+1) *
                            (1/(2*(p_power,q_power)[left]+rank+1)+1/(2*power+rank+1)) for power in (p_power,q_power))
                        close(f"actual_route_analytic_{case}_{rank}_{left}", scalar, expected, 0., 2e-13)
                cache.release_shared_banks()
                flush("case_" + str(case))
            expected_calls = contract["anchor_calls"]
            arrays = None
        else:
            saved, radial, geometry = load("saved_point"), load("workspace"), load("logical_geometry")
            waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                     zip(saved["keys"], saved["P"], saved["Q"], strict=True)}
            data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith("basis_")}
            gate("actual123_29completekeys", data["ns"] == 123 and len(waves) == 29)
            gate("actualP8_Q9", (data["ksp"], data["ksq"]) == (8, 9))
            nuclear = make_nuclear_potential(70, "fermi", A=175)
            snapshot = CIRadialSnapshot(waves, data, 70, 123, nuclear_potential=nuclear)
            evaluate = snapshot.evaluator_for(waves, data, 70, 123, nuclear_potential=nuclear)
            for k, raw_h, s in raw_hS:
                close(f"same_Lh_{k}", np.tril(raw_h) + np.tril(raw_h, -1).T, geometry[f"h_{k}"])
                close(f"same_S_{k}", s, geometry[f"s_{k}"])
                np.savez_compressed(output / f"raw_hS_{k}.npz", raw_h=raw_h, S=s)
            gate("seven_exact_hS", len(raw_hS) == 7)
            tables = load("signed_tables")
            targets = {tuple(t["UID"]): t for t in tables["targets"]}
            gate("unique_five_physical_UIDs", len(targets) == 5 and
                 {t["label"] for t in targets.values()} == set(contract["fixed"]["target_labels"]))
            for table in tables["target_tables"]:
                target = targets[tuple(table["UID"])]
                gate("target_column_" + target["label"], table["column"] == target["column"])
                column = saved["mixing"][:, target["column"]]
                norm = sum((Fraction.from_float(float(x))**2 for x in column), Fraction(0))
                gate("exact_CI_norm_" + target["label"], norm == Fraction(*map(int, target["exact_CI_norm"])))
            all_keys = [key(row[0]) for row in tables["full_normalized"]]
            selected = [k for k in all_keys if k[0] == "r" and k[1] >= 2]
            gate("complete_7398_positive_affected_keys", len(selected) == 7398 and len(set(selected)) == 7398
                 and max(k[1] for k in selected) == 6)
            all_target_keys = {key(k) for t in tables["target_tables"] for k, c in t["unfiltered"]}
            gate("complete_table_union", set(all_keys) == all_target_keys and len(all_keys) == 12403)
            pairs = {tuple(sorted((k[4], k[5]))) for k in selected}
            target_pairs = {tuple(sorted((k[2], k[3]))) for k in selected}
            source_pairs = {(k[1], tuple(sorted((k[4], k[5])))) for k in selected}
            gate("complete_333_rank_source_278_density", len(source_pairs) == 333 and len(pairs | target_pairs) == 278)
            function = wrap("public_radial_evaluate", evaluate, 7398)
            values = np.array([function(k) for k in selected])
            gate("all_radial_values_finite", np.all(np.isfinite(values)))
            arrays = output / "radial_values.npz"
            with arrays.open("wb") as handle:
                np.savez_compressed(handle, values=values)
                handle.flush()
                os.fsync(handle.fileno())
            with np.load(arrays, allow_pickle=False) as data0:
                close("exact_archive_readback", data0["values"], values)
            write(output / "radial_keys.json", selected)
            write(output / "snapshot_diagnostics.json", snapshot.diagnostics())
            expected_calls = contract["radial_calls_per_variant"]
        for name, expected in expected_calls.items():
            gate("exact_started_" + name, started.get(name, 0) == expected)
            gate("exact_completed_" + name, completed.get(name, 0) == expected)
        for name, module in tuple(sys.modules.items()):
            if (name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None):
                gate("source_import_" + name, Path(module.__file__).resolve().is_relative_to(source.resolve()))
        admit(execution)
        write(output / "result.json", {"all_numeric_controls_passed": True, "started": started,
              "completed": completed, "controls": controls, "GPU": 0, "HCIg_optimizer": 0,
              "source_map_sha256": contract[variant]["map"], "arrays_sha256": None if arrays is None else sha(arrays),
              "source_cache_scope": "fresh process, isolated NumBa directory; no old/new shared bank, snapshot or potential object",
              "elapsed_uncontrolled_s": time.monotonic() - clock,
              "versions": {"Python": sys.version, "NumPy": np.__version__, "SciPy": scipy.__version__, "Numba": numba.__version__}})
        flush("closed_passed")
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
              "started": started, "completed": completed, "controls": controls})
        flush("closed_failed")
        raise


def aggregate(execution):
    contract = admit(execution)
    import numpy as np
    tables = json.loads((REPO / contract["inputs"]["signed_tables"]["path"]).read_text())
    arrays = []
    for variant in ("ordinary", "candidate"):
        result = json.loads((HERE / "actual" / variant / "result.json").read_text())
        if not (result["all_numeric_controls_passed"] is True and result["source_map_sha256"] == contract[variant]["map"]
                and result["arrays_sha256"] == sha(HERE / "actual" / variant / "radial_values.npz")):
            raise ValueError("Unqualified radial worker archive")
        with np.load(HERE / "actual" / variant / "radial_values.npz", allow_pickle=False) as data:
            arrays.append(data["values"].copy())
    keys0 = json.loads((HERE / "actual/ordinary/radial_keys.json").read_text())
    keys1 = json.loads((HERE / "actual/candidate/radial_keys.json").read_text())
    if keys0 != keys1:
        raise ValueError("Different complete positive-multipole keys")
    differences = {key(k): Fraction.from_float(float(v1)) - Fraction.from_float(float(v0))
                   for k, v0, v1 in zip(keys0, *arrays, strict=True)}
    targets = {tuple(row["UID"]): row for row in tables["targets"]}
    rows, shifts = [], {}
    for table in tables["target_tables"]:
        target = targets[tuple(table["UID"])]
        norm = Fraction(*map(int, target["exact_CI_norm"]))
        if norm <= 0:
            raise ValueError("Nonpositive frozen exact CI norm")
        terms, by_rank = [], {}
        for k0, coefficient in table["unfiltered"]:
            k = key(k0)
            value = Fraction.from_float(float(coefficient)) / norm * differences.get(k, Fraction(0))
            if (k[0] == "r" and k[1] >= 2) != (k in differences):
                raise ValueError("Unbound affected key")
            terms.append(value)
            if k[0] == "r":
                by_rank.setdefault(k[1], []).append(value)
        delta = sum(terms, Fraction(0))
        shifts[target["label"]] = delta
        rows.append({"label": target["label"], "UID": target["UID"], "column": target["column"],
                     "signed_table_entries": len(terms), "exact_CI_norm": target["exact_CI_norm"],
                     "delta_expectation_Ha": float(delta), "delta_exact_fraction": [str(delta.numerator), str(delta.denominator)],
                     "sum_abs_signed_contributions_Ha": float(sum(map(abs, terms), Fraction(0))),
                     "by_rank_delta_Ha": {rank: float(sum(values, Fraction(0))) for rank, values in by_rank.items()}})
    if set(shifts) != set(contract["fixed"]["target_labels"]):
        raise ValueError("Different five physical targets")
    constant = Fraction.from_float(contract["Hartree_cm1"])
    for row in rows:
        row["delta_transition_cm1"] = float((shifts[row["label"]] - shifts["6s_1/2"]) * constant)
    calls_started, calls_completed = {}, {}
    for phase in ("anchors", "ordinary", "candidate"):
        result = json.loads((HERE / "actual" / phase / "result.json").read_text())
        for name, count in result["started"].items():
            calls_started[name] = calls_started.get(name, 0) + count
        for name, count in result["completed"].items():
            calls_completed[name] = calls_completed.get(name, 0) + count
    eol = sum((Fraction.from_float(float(t["weight"])) * shifts[t["label"]] for t in tables["targets"]), Fraction(0))
    write(HERE / "actual/result.json", {"positive_inner_order_discriminator_completed": True,
          "all_declared_discriminator_controls_passed": True, "calls_started": calls_started, "calls_completed": calls_completed,
          "fixed_CI_expectation_only": True, "targets": rows, "equal_EO_delta_Ha": float(eol),
          "full_table_scope": 12403, "affected_radial_keys": 7398,
          "unchanged_h_k0_k1_reused_exact_zero": True, "new_H_CI_gradient_optimizer_GPU": 0,
          "original_anchor_and_derivative_failures_preserved": True, "original_stationarity_not_changed": True,
          "new_spectrum_or_experiment_qualification": False, "uniform_radial_error_bound": False,
          "interpretation":"Stored signed/angular coefficient and CI norm algebra; same oldCI expectation of changed inner-order radial operator, not reoptimized or rediagonalized spectrum. Outer/negative/radial truncation errors not certified.",
          "source_maps": {v: contract[v]["map"] for v in ("ordinary", "candidate")},
          "worker_results_sha256": {v: sha(HERE / "actual" / v / "result.json") for v in ("anchors", "ordinary", "candidate")},
          "driver_sha256": sha(__file__), "contract_sha256": sha(HERE / "contract.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True)
    parser.add_argument("--worker", choices=("anchors", "ordinary", "candidate"))
    args = parser.parse_args()
    execution_path = Path(args.execution).absolute()
    execution = json.loads(execution_path.read_text())
    if args.worker:
        worker(args.worker, execution)
        return
    contract = admit(execution)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95*2**30), int(3.95*2**30)))
    os.sched_setaffinity(0, {31})
    write(output / "host_admission.json", {
        "affinity": sorted(os.sched_getaffinity(0)),
        "load_average_observation": os.getloadavg(),
        "meminfo": Path("/proc/meminfo").read_text(),
        "threads": {v: os.environ[v] for v in contract["resources"]["thread_variables"]},
        "performance_controlled": False, "CPU31_window_authority": "root exact seal"})
    begin = time.monotonic()
    for phase in ("anchors", "ordinary", "candidate"):
        remaining = 300 - (time.monotonic() - begin)
        if remaining <= 0:
            raise TimeoutError("Shared300s exhausted")
        child = dict(execution, remaining_seconds=remaining)
        child_path = output / (phase + "_execution.json")
        write(child_path, child)
        env = os.environ.copy()
        env["NUMBA_CACHE_DIR"] = str(output / (phase + "_numba_cache"))
        command = [contract["resources"]["python"], str(HERE / "run.py"),
                   "--execution", str(child_path), "--worker", phase]
        with (output / (phase + ".log")).open("wb") as log:
            try:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=remaining+2, check=False)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = -999
        write(output / (phase + "_process.json"), {"command": command, "exit_code": returncode,
              "elapsed_total_uncontrolled_s": time.monotonic()-begin, "log_sha256": sha(output/(phase+".log"))})
        if returncode:
            failure = {"failed_worker": phase, "exit_code": returncode,
                  "no_retry": True, "remaining_phases_unexecuted": True,
                  "positive_inner_order_discriminator_completed": False,
                  "all_declared_discriminator_controls_passed": False}
            write(output / "failure.json", failure)
            write(output / "result.json", failure)
            raise RuntimeError("Closed failed phase: " + phase)
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95*2**30), int(3.95*2**30)))
    os.sched_setaffinity(0, {31})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Aggregate300s cap")))
    signal.alarm(max(1, int(300-(time.monotonic()-begin))))
    aggregate(execution)


if __name__ == "__main__":
    main()
