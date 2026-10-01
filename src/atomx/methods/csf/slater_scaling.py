"""Series-level Slater-integral scaling (F^k calibration).

THE PROBLEM (measured 2026-10-01, see docs/SLATER_SCALING_SPEC.md):
frozen-DF 4f^n CI overestimates the Slater integrals — inter-term
splittings +12..37% (F^k-sorted), while spin-orbit (fine structure)
is accurate at the 2-7% level.  Second-order MBPT corrections into
high virtuals are orthogonality-suppressed and cannot close the gap
(the Sigma2 arc verdict).

THE SOLUTION (proven on Pr -> Eu/Nd transfer): scale each R^k radial
integral by (1 + s_k) with THREE series parameters (s2, s4, s6) fitted
ONCE on one ion's NIST levels.  Measured: Pr fit RMS 210 cm^-1 (12
levels); transferred unchanged: Eu3+ 5D0 +3164 -> -257 cm^-1, Nd3+
4F3/2 +34.1% -> -1.5%.  The fitted magnitudes (-8..-27%) match the
textbook correlation-screening (Cowan kappa) reduction.

CONVENTION: the assembler computes antisymmetrised elements
``anti = v(ab,cd) - v(ab,dc) + delta(ab,cd)``; the correct scaling
increment is therefore

    delta(ab,cd) = sum_k s_k [ term_k(ab,cd) - term_k(ab,dc) ]

(scaling only the first ordering would break antisymmetrisation).

PROTOCOL (fit/transfer):
  1. build a frozen-core DF workspace for the SOURCE ion (e.g. 4f^2);
  2. fit (s2, s4, s6) to its NIST free-ion levels (near-linear in the
     scalings -> one numerical Jacobian + a few Gauss-Newton steps);
  3. FREEZE the parameters and apply the SAME delta to any other ion
     of the series (predictive — no per-ion fitting).

Layering note: this module decomposes elements with
``two_body_angular`` and ``MAT._rk`` — the same private integrator the
assembler uses.  A public per-k term iterator in the MAT layer is the
right long-term home (see spec follow-ups).
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from atomx.angular.csf_hamiltonian import two_body_angular
import atomx.physics.csf.matrix as MAT

__all__ = ["make_slater_delta", "free_ion_levels", "fit_slater_scaling",
           "multiset_subtract"]

_SCALE_KS = (2, 4, 6)


def multiset_subtract(w, w_hi, tol: float = 1e-8):
    """Remove the ``w_hi`` entries from ``w`` (nearest-match, in place).

    The J-manifold extraction ``spec(J) = spec(M=J) - spec(M=J+2)``
    needs FULL sector spectra — partial spectra corrupt the subtraction
    (pass ``n_states`` covering every sector).
    """
    if len(w_hi) > len(w):
        raise ValueError(
            "multiset_subtract: high sector larger than low — "
            "spectra truncated or non-positive-M sectors used?")
    wl = list(w)
    for x in w_hi:
        j = min(range(len(wl)), key=lambda i: abs(wl[i] - x))
        gap = abs(wl[j] - x)
        if len(wl) > 1:
            others = sorted(abs(wl[i] - x) for i in range(len(wl))
                            if i != j)
            if others and others[0] < 1e-6 and gap > 1e-10:
                # near-degenerate alternative matches: the nearest-copy
                # heuristic may have grabbed the WRONG copy — flag it
                import warnings
                warnings.warn(
                    f"multiset_subtract ambiguous match (gap {gap:.3e} "
                    f"Ha, alternative {others[0]:.3e}); level "
                    "assignment may be unstable", stacklevel=2)
        wl.pop(j)
    return np.array(wl)


def make_slater_delta(orbitals_dict, bsp_data, scale: Mapping[int, float]):
    """two_body_delta scaling R^k by (1 + scale[k]); k outside the
    mapping is left untouched.  Returns ``delta(ia, ib, ic, id_,
    spinors)`` over spinor INDEX tuples (indices of the caller's
    table; labels are read from ``spinors``).
    """
    if isinstance(scale, (tuple, list)):
        scale = dict(zip(_SCALE_KS, scale))
    scale_map = {int(k): float(v) for k, v in scale.items()
                 if float(v) != 0.0}
    if not scale_map:
        return None

    def delta(ia, ib, ic, id_, spinors):
        a, b, c, d = (spinors[ia], spinors[ib], spinors[ic],
                      spinors[id_])
        t1 = two_body_angular(a[1], a[2], b[1], b[2],
                              c[1], c[2], d[1], d[2])
        t2 = two_body_angular(a[1], a[2], b[1], b[2],
                              d[1], d[2], c[1], c[2])
        if not t1 and not t2:
            return 0.0
        Pa, Qa = orbitals_dict[(a[0], a[1])]
        Pc, Qc = orbitals_dict[(c[0], c[1])]
        Pb, Qb = orbitals_dict[(b[0], b[1])]
        Pd, Qd = orbitals_dict[(d[0], d[1])]
        total = 0.0
        for terms, sign in ((t1, 1.0), (t2, -1.0)):
            for k, coef in terms:
                sk = scale_map.get(k)
                if sk is None:
                    continue
                total += sign * sk * coef * MAT._rk(
                    Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd, bsp_data, k)
        return total

    return delta


def free_ion_levels(orbitals_dict, bsp_data, Z, ns, patterns, scale=None,
                    twoJ_values=None, cm: float = 219474.63):
    """Pure-J ladder of a pattern set, optionally with Slater scaling.

    Returns a dict ``{J: sorted energies in cm^-1}`` via the
    M-sector subtraction ``spec(M=2J) - spec(M=2J+2)`` (full spectra —
    dense diagonalisation of each sector).  ``twoJ_values`` defaults to
    every distinct 2M the pattern set supports.
    """
    from atomx.physics.csf import sparse_ci as SC

    if twoJ_values is None:
        # M >= 0 only: the containment spec(M) >= spec(M+2) holds for
        # NON-NEGATIVE M (the J-manifold ladder runs up from M=0 or
        # 1/2); negative-M sectors have the inclusion reversed and
        # would exhaust the subtraction.
        seen = []
        for m2 in range(0, 31):
            masks, _ = SC.enumerate_determinants(patterns, orbitals_dict,
                                                 m2)
            if masks:
                seen.append(m2)
        twoJ_values = seen      # the TOP sector carries J = M_max
        # (spec(M_max) - spec(empty) = the stretched manifold)
    delta = (make_slater_delta(orbitals_dict, bsp_data, scale)
             if scale else None)
    sector_cache: Dict[int, np.ndarray] = {}

    def sector(m2):
        if m2 not in sector_cache:
            masks, spinors = SC.enumerate_determinants(patterns,
                                                       orbitals_dict, m2)
            if not masks:
                sector_cache[m2] = np.array([])
            elif delta is None:
                w, _ = SC.sparse_ci_spectrum(orbitals_dict, bsp_data, Z,
                                             ns, patterns, m2,
                                             n_states=10 ** 7)
                sector_cache[m2] = w
            else:
                dd = lambda a, b, c, d: delta(a, b, c, d, spinors)
                H = SC.assemble_sparse_hamiltonian(
                    masks, spinors, orbitals_dict, bsp_data, Z, ns,
                    two_body_delta=dd)
                sector_cache[m2] = np.linalg.eigvalsh(H.toarray())
        return sector_cache[m2]

    out: Dict[float, np.ndarray] = {}
    for twoJ in twoJ_values:
        w, w_hi = sector(twoJ), sector(twoJ + 2)
        if len(w) == 0:
            continue
        out[twoJ / 2] = np.sort(multiset_subtract(w, w_hi) * cm)
    return out


def fit_slater_scaling(residuals_fn, x0=None, h: float = 0.02,
                       max_iter: int = 8, tol: float = 1.0):
    """Gauss-Newton fit of (s2, s4, s6).

    ``residuals_fn(scale_tuple) -> np.array`` returns
    ``E_calc(scale) - E_ref`` per assigned level.  Level energies are
    analytic (nearly linear) in the scalings, so a numerical Jacobian
    converges in a handful of steps.  Returns ``(x, rms_history)``.
    """
    x = np.zeros(3) if x0 is None else np.asarray(x0, float).copy()
    hist = []
    diagnostics = {}
    for _ in range(max_iter):
        r = np.asarray(residuals_fn(tuple(x)), float)
        rms = float(np.sqrt(np.mean(r ** 2)))
        hist.append(rms)
        Jm = np.column_stack([
            (np.asarray(residuals_fn(tuple(x + e)), float) - r) / h
            for e in [(h, 0, 0), (0, h, 0), (0, 0, h)]])
        # conditioning diagnostics: the s4/s6 ratio direction is known
        # weakly pinned — report it instead of silently swallowing it
        sv = np.linalg.svd(Jm, compute_uv=False)
        diagnostics["cond"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else             float("inf")
        step, *_ = np.linalg.lstsq(Jm, -r, rcond=None)
        r_new = np.asarray(residuals_fn(tuple(x + step)), float)
        rms_new = float(np.sqrt(np.mean(r_new ** 2)))
        if rms_new >= rms - tol:
            # backtracking line search before giving up (review #7)
            improved = False
            for beta in (0.5, 0.25, 0.1):
                r_bt = np.asarray(residuals_fn(tuple(x + beta * step)),
                                   float)
                if float(np.sqrt(np.mean(r_bt ** 2))) < rms - tol:
                    x = x + beta * step
                    improved = True
                    break
            if not improved:
                break
        else:
            x = x + step
    # parameter error bars from (J^T J)^-1 * residual variance
    try:
        dof = max(len(r) - len(x), 1)
        cov = np.linalg.inv(Jm.T @ Jm) * (float(np.sum(r ** 2)) / dof)
        diagnostics["sigma"] = np.sqrt(np.diag(cov)).tolist()
    except np.linalg.LinAlgError:
        diagnostics["sigma"] = None
    return x, hist, diagnostics
