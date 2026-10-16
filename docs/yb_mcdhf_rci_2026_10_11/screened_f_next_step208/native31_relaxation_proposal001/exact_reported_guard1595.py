"""Exact stored-binary64 expectation guard; no physical producer or solve.

The caller must validate complete retained-H ownership, eigenpairs, sectors,
target identity and radial constraints. This module proves only a finite
inequality for those supplied stored matrices and columns. Concurrent input
mutation is unsupported. No quotient replaces the reported objective.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


class ExactGuardBudgetExceeded(RuntimeError):
    """Stop with the caller's previous verified accepted checkpoint."""


@dataclass
class ExactGuardBudget:
    """Private declared <=17-point, <=85-root, <=5m examined-pair run."""

    points: int = 0
    quotients: int = 0
    examined_pairs: int = 0
    nonzero_H_pairs: int = 0

    def reserve(self, roots, pairs):
        if self.points + 1 > 17 or self.quotients + roots > 85 or self.examined_pairs + pairs > 5_000_000:
            raise ExactGuardBudgetExceeded("exact reported acceptance budget exhausted")
        self.points += 1
        self.quotients += roots
        self.examined_pairs += pairs

    def ledger(self):
        return {key: int(getattr(self, key)) for key in
                ("points", "quotients", "examined_pairs", "nonzero_H_pairs")}


def _fraction_record(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _dyadic(value):
    numerator, denominator = float(value).as_integer_ratio()
    return numerator, denominator.bit_length() - 1


def _upper_float(value):
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("nonfinite diagnostic conversion of exact margin")
    return math.nextafter(out, math.inf) if Fraction.from_float(out) < value else out


@dataclass(frozen=True)
class ExactReportedPoint:
    target_ids: tuple
    energy_ha: float
    offset: Fraction
    weighted_reported: Fraction
    weighted_quotient: Fraction
    components: tuple
    array_sha256: tuple

    def ledger(self):
        return {
            "target_ids": list(self.target_ids), "energy_ha": self.energy_ha,
            "quotient_minus_actual_EO": _fraction_record(self.offset),
            "weighted_reported": _fraction_record(self.weighted_reported),
            "weighted_quotient": _fraction_record(self.weighted_quotient),
            "actual_EO_minus_exact_weighted_reported": _fraction_record(
                Fraction.from_float(self.energy_ha) - self.weighted_reported),
            "targets": list(self.components), "array_sha256": list(self.array_sha256),
            "domain": "exact finite supplied binary64 H/columns/roots/weights",
            "physical_or_gradient_certificate": False,
        }


def exact_reported_point(H, target_columns, reported_roots, weights, energy_ha, *, target_ids, budget):
    """Consume full H and selected whole signed columns after caller gates.

    All exact nonzero column components are retained. Every upper-triangle
    pair in that support is examined; signed off-diagonal terms get factor2.
    Only exact H zeros are omitted from integer accumulation. No eigensolve,
    density contraction, radial increment, new H or gradient is performed.
    """
    arrays = [np.asarray(a) for a in (H, target_columns, reported_roots, weights)]
    if any(a.dtype != np.dtype(np.float64) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("exact guard requires finite real binary64 arrays")
    h, c, roots, w = arrays
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] > 1595
            or c.ndim != 2 or c.shape[0] != h.shape[0]
            or not 1 <= c.shape[1] <= 5 or roots.shape != (c.shape[1],)
            or w.shape != roots.shape or not np.array_equal(h, h.T)
            or np.any(w <= 0) or abs(math.fsum(float(v) for v in w) - 1.) > 1e-12):
        raise ValueError("exact guard has invalid complete-H/target/weight shape")
    ids = tuple(target_ids)
    if len(ids) != c.shape[1] or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("exact guard requires unique canonical target identities")
    if type(energy_ha) is not float or not math.isfinite(energy_ha):
        raise ValueError("exact guard requires actual finite Python-float reported EOL")
    if not isinstance(budget, ExactGuardBudget):
        raise TypeError("exact guard requires its explicit shared run budget")
    supports = [np.flatnonzero(c[:, k] != 0).tolist() for k in range(c.shape[1])]
    if any(not rows for rows in supports):
        raise ValueError("exact guard rejects a zero target vector")
    budget.reserve(c.shape[1], sum(len(rows) * (len(rows) + 1) // 2 for rows in supports))
    reported, quotient, components = Fraction(0), Fraction(0), []
    for k, rows in enumerate(supports):
        dyads = [_dyadic(c[i, k]) for i in rows]
        c_exponent = max(e for _, e in dyads)
        integers = [v << (c_exponent - e) for v, e in dyads]
        norm_integer = sum(v * v for v in integers)
        norm = Fraction(norm_integer, 1 << (2 * c_exponent))
        pairs = []
        for a, i in enumerate(rows):
            for b in range(a, len(rows)):
                j = rows[b]
                if h[i, j] != 0:
                    numerator, exponent = _dyadic(h[i, j])
                    factor = integers[a] * integers[b] * (1 if a == b else 2)
                    pairs.append((numerator * factor, exponent))
        budget.nonzero_H_pairs += len(pairs)
        h_exponent = max((e for _, e in pairs), default=0)
        expectation_integer = sum(v << (h_exponent - e) for v, e in pairs)
        q = Fraction(expectation_integer, norm_integer << h_exponent)
        weight, root = Fraction.from_float(float(w[k])), Fraction.from_float(float(roots[k]))
        quotient += weight * q
        reported += weight * root
        components.append({"target_id": ids[k], "norm": _fraction_record(norm),
                           "quotient": _fraction_record(q), "reported": _fraction_record(root),
                           "quotient_minus_reported": _fraction_record(q - root),
                           "support": len(rows), "examined_pairs": len(rows) * (len(rows) + 1) // 2,
                           "nonzero_H_pairs": len(pairs)})
    hashes = tuple(hashlib.sha256(a.tobytes(order="C")).hexdigest() for a in arrays)
    return ExactReportedPoint(ids, energy_ha, quotient - Fraction.from_float(energy_ha),
                              reported, quotient, tuple(components), hashes)


def reported_acceptance_decision(previous, current, accepted_line_search):
    """Require sufficient exact stored-quotient margin under original RHS.

    This is a checkpoint qualification/stop guard, never an optimizer energy
    callback. On failure archive the candidate and previous verified point,
    then stop before any subsequent update. The float margin is display-only.
    """
    if not isinstance(previous, ExactReportedPoint) or not isinstance(current, ExactReportedPoint):
        raise TypeError("decision requires two exact endpoint point certificates")
    if previous.target_ids != current.target_ids:
        raise ValueError("decision target identity or order changed")
    line = accepted_line_search
    expected = {"parent_energy_ha", "trial_energy_ha", "step", "directional_derivative",
                "energy_change_ha", "armijo_rhs_ha", "attempt", "backtrack", "stable_callback_used"}
    if not isinstance(line, dict) or set(line) != expected or line["stable_callback_used"] is not False:
        raise ValueError("decision requires the exact ordinary reported comparison schema")
    fields = ("parent_energy_ha", "trial_energy_ha", "step", "directional_derivative", "energy_change_ha", "armijo_rhs_ha")
    if any(type(line[k]) is not float or not math.isfinite(line[k]) for k in fields):
        raise ValueError("decision comparison fields must be finite Python floats")
    if (type(line["attempt"]) is not int or line["attempt"] not in (0, 1)
            or type(line["backtrack"]) is not int or not 0 <= line["backtrack"] < 28
            or line["step"] <= 0 or line["directional_derivative"] >= 0):
        raise ValueError("decision has invalid accepted numerical step metadata")
    if (line["parent_energy_ha"].hex() != previous.energy_ha.hex()
            or line["trial_energy_ha"].hex() != current.energy_ha.hex()
            or line["energy_change_ha"].hex() != (current.energy_ha - previous.energy_ha).hex()
            or line["armijo_rhs_ha"].hex() != (1e-4 * line["step"] * line["directional_derivative"]).hex()):
        raise ValueError("decision metadata does not reproduce the original comparison")
    change, rhs = Fraction.from_float(line["energy_change_ha"]), Fraction.from_float(line["armijo_rhs_ha"])
    exact_change = Fraction.from_float(current.energy_ha) - Fraction.from_float(previous.energy_ha)
    exact_rhs = Fraction.from_float(1e-4) * Fraction.from_float(line["step"]) * Fraction.from_float(line["directional_derivative"])
    subtraction_error, rhs_error = abs(exact_change - change), abs(exact_rhs - rhs)
    margin = abs(previous.offset) + abs(current.offset) + subtraction_error + rhs_error
    qualified = change + margin <= rhs
    actual_q_change = current.weighted_quotient - previous.weighted_quotient
    if qualified and not actual_q_change <= exact_rhs:
        raise AssertionError("exact conservative inequality does not imply its stated bound")
    return {"qualified": bool(qualified), "actual_optimizer_comparison_passed": bool(change <= rhs),
            "margin_upper_float_Ha": _upper_float(margin), "margin_exact": _fraction_record(margin),
            "exact_reported_subtraction_error": _fraction_record(subtraction_error),
            "exact_RHS_rounding_error": _fraction_record(rhs_error),
            "stored_quotient_change": _fraction_record(actual_q_change),
            "guard_lhs": _fraction_record(change + margin), "recorded_rhs": _fraction_record(rhs),
            "exact_product_rhs": _fraction_record(exact_rhs),
            "scope": "finite stored-H normalized expectations; no C-prime/radial/physical error bound"}
