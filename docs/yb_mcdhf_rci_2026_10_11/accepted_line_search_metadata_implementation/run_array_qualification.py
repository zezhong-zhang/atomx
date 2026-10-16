"""One prospective bounded array-only arithmetic qualification, no Atomx."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).parent
REPO = BASE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    contract_path = BASE / "array_qualification_contract.json"
    contract = json.loads(contract_path.read_text())
    for name, digest in contract["inputs_sha256"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError("array qualification input changed: " + name)
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    os.sched_setaffinity(0, {31})
    import numpy as np
    from exact_reported_guard import (
        ExactGuardBudget,
        exact_reported_point,
        reported_acceptance_decision,
    )

    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    def independent_fraction_q(h, c):
        vector = [Fraction.from_float(float(v)) for v in c]
        numerator = sum(vector[i] * Fraction.from_float(float(h[i, j])) * vector[j]
                        for i in range(len(vector)) for j in range(len(vector)))
        return numerator / sum(v * v for v in vector)

    budget = ExactGuardBudget()

    def point(h, c, roots, weights, energy, ids):
        return exact_reported_point(h, c, np.array(roots, dtype=np.float64),
                                    np.array(weights, dtype=np.float64), float(energy),
                                    target_ids=ids, budget=budget)

    h = np.array([[-7., -1.25], [-1.25, -5.]])
    c = np.array([[.75], [-.5]])
    q = independent_fraction_q(h, c[:, 0])
    p0 = point(h, c, [float(q)], [1.], float(q), ["signed-state"])
    check("signed_offdiagonal_exact_full_ij_reference", p0.weighted_quotient == q)
    norm = Fraction(13, 16)
    check("actual_norm_not_assumed_one", p0.components[0]["norm"] ==
          {"numerator": str(norm.numerator), "denominator": str(norm.denominator)})
    phase = point(h, -c, [float(q)], [1.], float(q), ["signed-state"])
    scaled = point(h, 2 * c, [float(q)], [1.], float(q), ["signed-state"])
    changed = point(h, np.array([[.75], [.5]]), [float(q)], [1.], float(q), ["signed-state"])
    check("whole_column_phase_invariance", phase.weighted_quotient == q)
    check("actual_norm_scale_invariance", scaled.weighted_quotient == q)
    check("relative_phase_interference_retained", changed.weighted_quotient != q)
    h1 = h - np.eye(2)
    q1 = independent_fraction_q(h1, c[:, 0])
    p1 = point(h1, c, [float(q1)], [1.], float(q1), ["signed-state"])

    def line(previous, current, slope):
        return {"parent_energy_ha": previous.energy_ha, "trial_energy_ha": current.energy_ha,
                "step": 1., "directional_derivative": float(slope),
                "energy_change_ha": current.energy_ha - previous.energy_ha,
                "armijo_rhs_ha": 1e-4 * float(slope), "attempt": 0, "backtrack": 0,
                "stable_callback_used": False}

    good = reported_acceptance_decision(p0, p1, line(p0, p1, -1.))
    check("well_resolved_reported_change_qualified", good["qualified"])
    check("qualified_implies_exact_matrix_expectation_Armijo", q1 - q <= Fraction.from_float(1e-4) * -1)
    below = float(np.nextafter(p0.energy_ha, -np.inf))
    unsafe = point(h, c, [below], [1.], below, ["signed-state"])
    bad = reported_acceptance_decision(p0, unsafe, line(p0, unsafe, -1e-16))
    check("rounded_reported_acceptance_observed", bad["actual_optimizer_comparison_passed"])
    check("unresolved_decrease_rejected_without_gate_change", not bad["qualified"])
    margin = Fraction(int(bad["margin_exact"]["numerator"]), int(bad["margin_exact"]["denominator"]))
    check("display_margin_rounded_up_and_not_used_for_decision",
          Fraction.from_float(bad["margin_upper_float_Ha"]) >= margin)
    h3 = np.array([[-8., .75, 0.], [.75, -4., -.5], [0., -.5, -2.]])
    c3 = np.array([[.5, -.25], [-.25, .75], [.75, .5]])
    w3 = [.25, .75]
    q3 = [independent_fraction_q(h3, c3[:, k]) for k in range(2)]
    e3 = sum(w * float(v) for w, v in zip(w3, q3, strict=True))
    averaged = point(h3, c3, [float(v) for v in q3], w3, e3, ["first", "second"])
    check("per_state_normalized_weighted_signed_reference",
          averaged.weighted_quotient == sum(Fraction.from_float(w) * q for w, q in zip(w3, q3, strict=True)))
    check("only_exact_H_zero_terms_omitted", averaged.components[0]["examined_pairs"] == 6
          and averaged.components[0]["nonzero_H_pairs"] == 5)

    probe = REPO / "docs/yb_mcdhf_rci_2026_10_11/correlated29_mcdhf_probe_preparation"
    progress = json.loads((probe / "actual/progress.json").read_text())
    record = next(v for v in progress["records"] if v["name"] == "recovered_coordinate_point")
    rows = record["targets"]
    with np.load(probe / "actual/recovered_coordinate_point.npz", allow_pickle=False) as archive:
        H, mixing, energies = [archive[key].copy() for key in ("H", "mixing", "energies")]
    cols = [int(v["column"]) for v in rows]
    ids = [json.dumps(v["UID"], separators=(",", ":")) for v in rows]
    weights = [float(v["tracking"]["per_level_weight"]) for v in rows]
    saved = point(H, mixing[:, cols], energies[cols], weights, record["EOL_Ha"], ids)
    decimal = json.loads((probe / "saved_array_audit/decomposition.json").read_text())
    decimal_rows = {v["label"]: v for v in decimal["targets"] if v["point"] == "recovered_coordinate_point"}
    with localcontext() as ctx:
        ctx.prec = 80
        for row, component in zip(rows, saved.components, strict=True):
            value = component["quotient"]
            exact_to_80 = Decimal(value["numerator"]) / Decimal(value["denominator"])
            check("saved_point_independent_Decimal80_" + row["label"],
                  abs(exact_to_80 - Decimal(decimal_rows[row["label"]]["Rayleigh"])) <= Decimal("1e-65"))
    check("no_new_points_or_extra_quotients", budget.points <= 10 and budget.quotients <= 20)
    check("bounded_examined_integer_pairs", budget.examined_pairs <= 150000)
    result = {"all_passed": all(v["passed"] for v in checks), "checks_count": len(checks), "checks": checks,
              "contract_sha256": sha(contract_path), "driver_sha256": sha(Path(__file__)),
              "consumer_sha256": sha(BASE / "exact_reported_guard.py"), "inputs_sha256": contract["inputs_sha256"],
              "actual_calls": {"H": 0, "CI": 0, "gradient": 0, "optimizer": 0, "Dirac": 0,
                               "hS": 0, "eigen": 0, "SVD": 0, "radial_increment": 0},
              "arithmetic_counters": budget.ledger(), "numpy_version": np.__version__,
              "saved_point_certificate": saved.ledger(), "analytic_qualified": good,
              "analytic_unresolved_rejected": bad, "elapsed_uncontrolled_seconds": time.monotonic() - started,
              "no_Yb_gradient_or_optimizer_qualification": True}
    (BASE / "array_qualification_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_passed", "checks_count", "arithmetic_counters", "elapsed_uncontrolled_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        (BASE / "array_qualification_failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
        raise
