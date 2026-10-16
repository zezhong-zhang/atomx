"""Independent saved-only array/80-decimal audit; zero physical operators."""

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()
    if sha(HERE / "contract.json") != args.contract_sha256:
        raise ValueError("Changed prospective audit contract")
    contract = json.loads((HERE / "contract.json").read_text())
    if contract["script_sha256"] != sha(__file__):
        raise ValueError("Changed saved-only consumer")
    for path, expected in contract["inputs_sha256"].items():
        if sha(REPO / path) != expected:
            raise ValueError("Changed frozen input " + path)
    if any(
        os.environ.get(k) != "1"
        for k in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        )
    ):
        raise ValueError("Audit requires one BLAS thread")
    affinity = sorted(os.sched_getaffinity(0))
    if 31 not in affinity:
        raise ValueError("Authorized CPU31 unavailable")
    os.sched_setaffinity(0, {31})
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("60s saved-audit cap")),
    )
    signal.alarm(60)
    begin = time.monotonic()
    import numpy as np

    for name in ("eigh", "eig", "eigvalsh", "svd"):

        def reject(*_args, **_kwargs):
            raise ValueError("Saved audit forbids spectral decomposition")

        setattr(np.linalg, name, reject)
    controls = []
    counters = {
        "public_H_CI_gradient_Dirac_optimizer": 0,
        "spectral_decompositions": 0,
        "Decimal_target_quotients": 0,
        "Decimal_nonzero_H_summands": 0,
    }

    def check(name, error, limit=0.0):
        controls.append(
            {
                "name": name,
                "error": float(error),
                "limit": float(limit),
                "passed": bool(np.isfinite(error) and error <= limit),
            }
        )

    def load(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    def jload(path):
        return json.loads(path.read_text())

    initial = jload(BASE / "contract.json")
    roles = {key: REPO / value["path"] for key, value in initial["input_roles"].items()}
    progress = jload(BASE / "actual/progress.json")
    root = jload(BASE / "root_execution.json")
    failure = jload(BASE / "actual/failure.json")
    check("original_foreground_failed", root["exit_code"] != 1)
    check(
        "original_first_reported_derivative_failure",
        progress["phase"] != "direction0_reported_derivative"
        or "direction0_reported_derivative" not in failure["error"],
    )
    check(
        "original_attempts_completed_equal",
        progress["started"] != progress["completed"],
    )
    check(
        "original4H4CI1g7Dirac",
        any(
            progress["completed"][k] != v
            for k, v in {"H": 4, "CI": 4, "gradient": 1, "validation_Dirac": 7}.items()
        ),
    )
    check(
        "original280controls_one_expected_failure",
        len(progress["controls"]) != 280
        or sum(not r["passed"] for r in progress["controls"]) != 1,
    )
    check(
        "second_direction_unavailable",
        any(
            (BASE / "actual" / p).exists()
            for p in (
                "direction1_plus.npz",
                "direction1_minus.npz",
                "direction1_central.json",
            )
        ),
    )
    records = {r["name"]: r for r in progress["records"]}
    names = [
        "canonical_anchor",
        "recovered_coordinate_point",
        "direction0_plus",
        "direction0_minus",
    ]
    points = {name: load(BASE / "actual" / f"{name}.npz") for name in names}
    basis = jload(roles["screened_basis"])["balanced_virtual"]["retained_signatures"]
    sectors = [tuple(json.loads(row)[-2:]) for row in basis]
    target_ref = jload(roles["tracking_binding"])["targets"]
    expected = {
        t["label"]: (t["twoJ"], t["parity"], t["models"]["DC"]["local_root"])
        for t in target_ref
    }
    checkpoint = load(roles["checkpoint_arrays"])
    old_layout = jload(roles["original_layout"])
    layout = jload(BASE / "actual/new_coordinate_layout.json")
    metric = load(BASE / "actual/full_gradient_metric.npz")
    point_identity = jload(BASE / "actual/point_identity.json")
    physical = load(BASE / "actual/full_point_identity.npz")
    geo = load(roles["screened_geometry"])
    ci_metrics = {}
    for name, point in points.items():
        h, c, e = point["H"], point["mixing"], point["energies"]
        check(
            name + "_complete_shape",
            h.shape != (1046, 1046) or c.shape != (1046, 1046) or e.shape != (1046,),
        )
        check(name + "_finite", not all(np.isfinite(v).all() for v in (h, c, e)))
        check(name + "_symmetric", not np.array_equal(h, h.T))
        residual = float(np.max(np.linalg.norm(h @ c - c * e, axis=0)))
        gram = float(np.max(abs(c.T @ c - np.eye(1046))))
        check(name + "_full1046_residual", residual, 1e-8)
        check(name + "_full1046_Gram", gram, 1e-10)
        ci_metrics[name] = {"max1046_residual_Ha": residual, "Gram_max": gram}
        rowmap = {r["label"]: r for r in records[name]["targets"] if "label" in r}
        check(name + "_labels_complete", set(rowmap) != set(expected))
        for label, uid in expected.items():
            row = rowmap[label]
            selected = tuple(row["tracking"]["selected_levels"][0])
            col = row["column"]
            check(name + "_reference_UID_" + label, tuple(row["UID"]) != uid)
            check(name + "_selected_sector_" + label, selected[:2] != uid[:2])
            support = np.flatnonzero(c[:, col] != 0)
            check(
                name + "_selected_exact_CSF_sector_" + label,
                any(sectors[i] != uid[:2] for i in support),
            )
            check(
                name + "_signed_root_energy_" + label,
                abs(row["energy_Ha"] - float(e[col])),
            )
            check(
                name + "_majority_" + label,
                0.5 - min(row["tracking"]["principal_overlap_squared"]),
                0.0,
            )
            check(
                name + "_fixed_weight_" + label,
                abs(row["tracking"]["per_level_weight"] - 0.2),
            )
        for kappa in sorted(set(point["keys"][:, 1])):
            idx = np.flatnonzero(point["keys"][:, 1] == kappa)
            radial = np.concatenate((point["P"][idx], point["Q"][idx]), axis=1).T
            s = geo[f"s_{kappa}"]
            p = next(
                metric[f"projector_{i}"]
                for i, row in enumerate(layout)
                if row["keys"][0][1] == kappa
            )
            check(
                name + f"_radial_Gram_{kappa}",
                np.max(abs(radial.T @ s @ radial - np.eye(len(idx)))),
                1e-10,
            )
            outside = radial - p @ (p.T @ s @ radial)
            check(
                name + f"_P_leakage_{kappa}",
                np.max(np.sum(outside * (s @ outside), axis=0)),
                1e-12,
            )
    old_h = load(roles["screened_H"])["matrix_Ha"]
    old_ci = load(roles["screened_CI"])
    check(
        "canonical_original353_H_bitwise",
        not np.array_equal(old_h, points[names[0]]["H"]),
    )
    check(
        "canonical_original353_E_bitwise",
        not np.array_equal(old_ci["energies_Ha"], points[names[0]]["energies"]),
    )
    check(
        "canonical_original353_C_bitwise",
        not np.array_equal(old_ci["mixing"], points[names[0]]["mixing"]),
    )
    x = metric["x"]
    check(
        "same3451_authoritative_x",
        x.shape != (3451,) or not np.array_equal(x, physical["x"]),
    )
    keys = [tuple(map(int, k)) for k in physical["keys"]]
    radial_g = metric["radial_gradient"]
    saved_g = metric["tangent_gradient"]
    raw = np.zeros_like(x)
    digest = hashlib.sha256()
    geometry_report = []
    for i, row in enumerate(layout):
        group = list(map(tuple, row["keys"]))
        shape = tuple(row["shape"])
        where = slice(*row["slice"])
        p = metric[f"projector_{i}"]
        check(
            f"P_original_bytes_{i}",
            p.tobytes() != checkpoint[f"projector_{i}"].tobytes(),
        )
        y = x[where].reshape(shape)
        raw[where] = (p.T @ radial_g[[keys.index(k) for k in group]].T).ravel()
        spinor = p @ y
        check(
            f"direct_Px_{i}",
            not np.array_equal(
                spinor,
                np.concatenate(
                    (
                        physical["recovered_P"][[keys.index(k) for k in group]],
                        physical["recovered_Q"][[keys.index(k) for k in group]],
                    ),
                    axis=1,
                ).T,
            ),
        )
        original = old_layout[i]
        oldgroup = list(map(tuple, original["keys"]))
        oldy = checkpoint["x"][slice(*original["slice"])].reshape(
            tuple(original["shape"])
        )
        oldd = checkpoint["inverse_diagonal"][slice(*original["slice"])].reshape(
            tuple(original["shape"])
        )
        diagonal = metric["inverse_diagonal"][where].reshape(shape)
        copies = {
            tuple(map(int, k.split(","))): tuple(v)
            for k, v in initial["metric"][
                "new5_same_kappa_SPD_block_copy_sources"
            ].items()
        }
        for j, key in enumerate(group):
            source = key if key in oldgroup else copies[key]
            sourcej = oldgroup.index(source)
            check(
                f"inverse_copy_bytes_{key}",
                metric[f"inverse_blocks_{i}"][j].tobytes()
                != checkpoint[f"inverse_blocks_{i}"][sourcej].tobytes(),
            )
            check(
                f"diagonal_copy_bytes_{key}",
                diagonal[:, j].tobytes() != oldd[:, sourcej].tobytes(),
            )
            if key in oldgroup:
                check(
                    f"old24_x_column_bytes_{key}",
                    y[:, j].tobytes() != oldy[:, sourcej].tobytes(),
                )
        digest.update(repr((group, shape)).encode())
        digest.update(p.tobytes())
        s = geo[f"s_{group[0][1]}"]
        wide = np.longdouble
        geometry_report.append(
            {
                "kappa": group[0][1],
                "coordinate_Gram_max": float(
                    np.max(
                        abs(
                            y.astype(wide).T @ y.astype(wide)
                            - np.eye(shape[1], dtype=wide)
                        )
                    )
                ),
                "physical_Gram_max": float(
                    np.max(
                        abs(
                            spinor.astype(wide).T @ s.astype(wide) @ spinor.astype(wide)
                            - np.eye(shape[1], dtype=wide)
                        )
                    )
                ),
            }
        )
    digest.update(metric["inverse_diagonal"].tobytes())
    for i, _row in enumerate(layout):
        for block in metric[f"inverse_blocks_{i}"]:
            digest.update(block.tobytes())
    check(
        "new_metric_SHA_exact",
        digest.hexdigest() != point_identity["new_metric_sha256"],
    )

    def tangent(vector):
        value = vector.copy()
        for row in layout:
            where = slice(*row["slice"])
            shape = tuple(row["shape"])
            y = x[where].reshape(shape)
            v = vector[where].reshape(shape)
            value[where] = (v - y @ ((y.T @ v + v.T @ y) * 0.5)).ravel()
        return value

    g = tangent(raw)
    pre = g.copy()
    for i, row in enumerate(layout):
        where = slice(*row["slice"])
        shape = tuple(row["shape"])
        blockg = g[where].reshape(shape)
        pre[where] = np.column_stack(
            [b @ blockg[:, j] for j, b in enumerate(metric[f"inverse_blocks_{i}"])]
        ).ravel()
    pg = tangent(pre)
    check(
        "saved_radial_to_coordinate_all3451",
        np.max(abs(raw - metric["raw_coordinate_gradient"])),
        1e-9,
    )
    check("saved_raw_to_tangent_all3451", np.max(abs(g - saved_g)), 1e-9)
    check(
        "saved_inverse_to_pg_all3451",
        np.max(abs(pg - metric["preconditioned_gradient"])),
        1e-10,
    )
    direction = load(BASE / "actual/direction0_coordinates.npz")["direction"]
    plus = load(BASE / "actual/direction0_plus_coordinates.npz")["candidate_x"]
    minus = load(BASE / "actual/direction0_minus_coordinates.npz")["candidate_x"]
    wide = np.longdouble
    even = (plus.astype(wide) + minus.astype(wide)) / 2 - x.astype(wide)
    odd = (plus.astype(wide) - minus.astype(wide)) / 2
    polar_quadratic = np.zeros_like(even)
    initial_repair = np.zeros_like(even)
    for row in layout:
        where = slice(*row["slice"])
        shape = tuple(row["shape"])
        y = x[where].reshape(shape).astype(wide)
        d = direction[where].reshape(shape).astype(wide)
        polar_quadratic[where] = (-wide(0.5) * wide(1e-6) ** 2 * y @ (d.T @ d)).ravel()
        initial_repair[where] = (
            -wide(0.5) * y @ (y.T @ y - np.eye(shape[1], dtype=wide))
        ).ravel()
    motion = {
        "coordinate_Gram_and_physical_Gram": geometry_report,
        "even_norm": float(np.linalg.norm(even)),
        "odd_minus_hD_norm": float(np.linalg.norm(odd - wide(1e-6) * direction)),
        "leading_polar_quadratic_norm": float(np.linalg.norm(polar_quadratic)),
        "leading_baseline_Gram_repair_norm": float(np.linalg.norm(initial_repair)),
        "even_minus_leading_quadratic_and_repair_norm": float(
            np.linalg.norm(even - polar_quadratic - initial_repair)
        ),
        "raw_g_dot_even_Ha": float(raw.astype(wide) @ even),
        "tangent_g_dot_even_Ha": float(g.astype(wide) @ even),
        "raw_g_dot_leading_quadratic_Ha": float(raw.astype(wide) @ polar_quadratic),
        "raw_g_dot_leading_baseline_repair_Ha": float(
            raw.astype(wide) @ initial_repair
        ),
        "no_complete_energy_curvature_reconstruction": True,
    }
    np.savez_compressed(
        HERE / "saved_motion.npz",
        even=even,
        odd=odd,
        leading_polar_quadratic=polar_quadratic,
        leading_baseline_repair=initial_repair,
    )
    central = jload(BASE / "actual/direction0_central.json")
    diagnostics = jload(BASE / "actual/direction0_diagnostics.json")
    check(
        "analytic_saved_gradient_direction",
        abs(float(g @ direction) - central["analytic"]),
        1e-12,
    )
    check(
        "reported_gate_failed_preserved",
        abs(central["reported_central"] - central["analytic"]) <= central["gate"],
    )
    check(
        "Rayleigh_gate_not_executed",
        any(
            r["name"] == "direction0_Rayleigh_derivative" for r in progress["controls"]
        ),
    )
    check(
        "original_reported_and_Rayleigh_gates_preserved",
        central["gate"] != max(1e-8, 1e-5 * abs(central["analytic"])),
    )

    def quotient(point, column):
        c = point["mixing"][:, column]
        selected = np.flatnonzero(c != 0)
        h = point["H"]
        dec = Decimal.from_float
        with localcontext() as ctx:
            ctx.prec = 80
            coeff = [dec(float(c[i])) for i in selected]
            n = sum((v * v for v in coeff), Decimal(0))
            expectation = Decimal(0)
            for a, i in enumerate(selected):
                for b in range(a + 1):
                    j = selected[b]
                    if h[i, j] != 0:
                        counters["Decimal_nonzero_H_summands"] += 1
                        if counters["Decimal_nonzero_H_summands"] > 3_000_000:
                            raise ValueError("Decimal summand budget exhausted")
                        expectation += (
                            (1 if a == b else 2)
                            * coeff[a]
                            * coeff[b]
                            * dec(float(h[i, j]))
                        )
            counters["Decimal_target_quotients"] += 1
            lam = dec(float(point["energies"][column]))
            q = expectation / n
            return {
                "norm": n,
                "reported": lam,
                "Rayleigh": q,
                "q_minus_lambda": q - lam,
                "raw_norm_energy_term": lam * (n - 1),
                "raw_expectation_minus_lambda": expectation - lam,
            }

    decimal_points = {}
    decimal_rows = []
    with localcontext() as ctx:
        ctx.prec = 80
        for name in names[1:]:
            rowmap = {r["label"]: r for r in records[name]["targets"] if "label" in r}
            values = {
                label: quotient(points[name], rowmap[label]["column"])
                for label in expected
            }
            decimal_points[name] = values
            for label, v in values.items():
                decimal_rows.append(
                    {"point": name, "label": label, **{k: str(n) for k, n in v.items()}}
                )
        weight = Decimal.from_float(0.2)
        aggregates = {
            name: {
                kind: sum((weight * v[kind] for v in values.values()), Decimal(0))
                for kind in (
                    "reported",
                    "Rayleigh",
                    "q_minus_lambda",
                    "raw_norm_energy_term",
                )
            }
            for name, values in decimal_points.items()
        }
        dec_den = 2 * Decimal("0.000001")
        reported_slope = (
            aggregates["direction0_plus"]["reported"]
            - aggregates["direction0_minus"]["reported"]
        ) / dec_den
        rayleigh_slope = (
            aggregates["direction0_plus"]["Rayleigh"]
            - aggregates["direction0_minus"]["Rayleigh"]
        ) / dec_den
        residual_offset_slope = (
            -(
                aggregates["direction0_plus"]["q_minus_lambda"]
                - aggregates["direction0_minus"]["q_minus_lambda"]
            )
            / dec_den
        )
        check(
            "Decimal_reported_minus_Rayleigh_identity",
            abs(float(reported_slope - rayleigh_slope - residual_offset_slope)),
            1e-60,
        )
        for sign, diag in zip(
            ("direction0_plus", "direction0_minus"), diagnostics, strict=True
        ):
            for target in diag["targets"]:
                uid = tuple(target["reference_level"])
                label = next(label for label, value in expected.items() if value == uid)
                q0 = decimal_points["recovered_coordinate_point"][label][
                    "q_minus_lambda"
                ]
                q1 = decimal_points[sign][label]["q_minus_lambda"]
                check(
                    sign + "_Decimal_qminuslambda0_" + label,
                    abs(float(q0) - target["quotient_minus_reported"][0]),
                    2e-14,
                )
                check(
                    sign + "_Decimal_qminuslambda1_" + label,
                    abs(float(q1) - target["quotient_minus_reported"][1]),
                    2e-14,
                )
            check(
                sign + "_shared_radial_cancels_reported_Rayleigh",
                abs(
                    (diag["reported_change"] - diag["rayleigh_change"])
                    - (diag["reported_response"] - diag["rayleigh_response"])
                ),
                1e-22,
            )
        decomposition = {
            "Decimal_precision": 80,
            "domain": "Exact binary64 stored H/CI/lambda; no new physical matrices or radial polynomial.",
            "targets": decimal_rows,
            "weighted_points": {
                k: {n: str(v) for n, v in values.items()}
                for k, values in aggregates.items()
            },
            "Decimal_reported_root_central": str(reported_slope),
            "Decimal_normalized_Rayleigh_central": str(rayleigh_slope),
            "Decimal_endpoint_residual_offset_slope": str(residual_offset_slope),
            "saved_diagnostic_reported_minus_Rayleigh_slope": central[
                "reported_central"
            ]
            - central["Rayleigh_central"],
            "saved_diagnostic_reported_error": abs(
                central["reported_central"] - central["analytic"]
            ),
            "saved_diagnostic_Rayleigh_error": abs(
                central["Rayleigh_central"] - central["analytic"]
            ),
            "Rayleigh_original_gate_historically_unexecuted": True,
            "saved_Rayleigh_measurement_within_original_gate": abs(
                central["Rayleigh_central"] - central["analytic"]
            )
            <= central["gate"],
            "Decimal_normalized_Rayleigh_vs_analytic_error": abs(
                float(rayleigh_slope) - central["analytic"]
            ),
            "ordinary_one_ULP_per_h": central["ordinary_one_ULP_derivative_scale"],
            "common_diagnostic_explicit_radial": sum(
                d["explicit_radial"] for d in diagnostics
            )
            / 2,
            "common_diagnostic_Rayleigh_energy_change": sum(
                d["rayleigh_change"] for d in diagnostics
            )
            / 2,
            "common_shift_is_not_proven_polar_baseline_noise": True,
            "polar_motion": motion,
            "finite_CI_response_required": "Normalized Rayleigh is the physical variational expectation; along approximate CI vectors its derivative includes 2*c_prime.T*(H-qI)*c/n. A future quotient objective requires consistent baseline/trial/accepted energy semantics, never callback-only replacement.",
        }
    density = jload(BASE / "actual/signed_density.json")

    def frozen(value):
        return tuple(map(frozen, value)) if isinstance(value, list) else value

    unfiltered = {frozen(k): v for k, v in density["unfiltered"]}
    retained = {frozen(k): v for k, v in density["actual_gradient_retained"]}
    cutoff = max(
        abs(unfiltered.get(k, 0) - retained.get(k, 0))
        for k in set(unfiltered) | set(retained)
    )
    check(
        "signed_RDM_retained_gap_ledger",
        abs(cutoff - density["unfiltered_vs_retained_max_coefficient_difference"]),
        1e-30,
    )
    check("saved_norms_not_normalized_inplace", len(density["CI_column_norms"]) != 5)
    decomposition["signed_RDM"] = {
        "unfiltered_terms": len(unfiltered),
        "negative_unfiltered_terms": sum(v < 0 for v in unfiltered.values()),
        "retained_terms": len(retained),
        "maximum_coefficient_difference": cutoff,
        "not_an_independent_angular_RDM_rebuild": True,
    }
    counters["NumPy_imported"] = True
    check(
        "zero_Atomx_SciPy_imports",
        any(n == "atomx" or n.startswith(("atomx.", "scipy")) for n in sys.modules),
    )
    check(
        "all_inputs_unchanged_after_audit",
        any(sha(REPO / p) != h for p, h in contract["inputs_sha256"].items()),
    )
    result = {
        "all_saved_array_checks_passed": all(r["passed"] for r in controls),
        "checks_count": len(controls),
        "checks": controls,
        "original_probe_qualified": False,
        "full_MCDHF_optimization_completed": False,
        "experimental_certification": False,
        "scope": "Independent saved matrices, whole vector numerical metric action and Decimal algebra; no new full-gradient replay, H/CI or radial increment.",
        "whole_public_samepoint_gradient_replay_executed": False,
        "original_direction1_unavailable": True,
        "original_reported_derivative_failed": True,
        "original_Rayleigh_derivative_gate_executed": False,
        "new_metric_sha256": digest.hexdigest(),
        "saved_original_P_preserved": True,
        "reconstructed_orbital_residual": float(np.linalg.norm(pg)),
        "reconstructed_energy_indicator_Ha": 0.5 * abs(float(g @ pg)),
        "original_orbital_tolerance": 1e-7,
        "original_energy_tolerance_Ha": 1e-8,
        "saved_newmetric_stationarity_gates_passed": bool(
            np.linalg.norm(pg) < 1e-7 and 0.5 * abs(g @ pg) < 1e-8
        ),
        "full_CI_metrics": ci_metrics,
        "original_calls": progress["completed"],
        "audit_calls": counters,
        "inputs_sha256": contract["inputs_sha256"],
        "audit_contract_sha256": sha(HERE / "contract.json"),
        "audit_driver_sha256": sha(__file__),
        "elapsed_uncontrolled_s": time.monotonic() - begin,
        "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "CPU": 31,
        "available_accounting_CPUs": affinity,
        "BLAS_threads": 1,
        "address_space_GiB": 2,
        "wall_cap_s": 60,
    }
    (HERE / "decomposition.json").write_text(
        json.dumps(decomposition, indent=2, allow_nan=False) + "\n"
    )
    result["decomposition_sha256"] = sha(HERE / "decomposition.json")
    result["saved_motion_sha256"] = sha(HERE / "saved_motion.npz")
    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "checks",
                    "inputs_sha256",
                    "available_accounting_CPUs",
                    "full_CI_metrics",
                )
            }
        )
    )
    if not result["all_saved_array_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
