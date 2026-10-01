"""Tabulated hydrogenic self-energy backend.

The numerical F(Zα) table lives in :mod:`atomx.data.mohr_se` with full
citations; this module owns only the interpolation logic and the
SCF-vs-hydrogenic overlap screening that convert that table into a
per-orbital self-energy shift.

Table selection follows FAC's ``HydrogenicSelfEnergy``:

* ``n <= 2`` (``1s1/2``, ``2s1/2``, ``2p1/2``, ``2p3/2``) — the fine
  ``Z`` grid of ``faclib/coulomb.c:_qed_se0 + _qed_se1``;
* ``3 <= n <= 5`` with ``|kappa| <= 3`` — FAC's ``_sd`` table, which
  carries the full ``l``/``j`` dependence;
* anything else — FAC's ``Klamaq`` analytic estimate, using its Bethe-log
  constants through ``n=8`` and NIST TN 1467 values for ``9 <= n <= 200``.
  For s states, use NIST Bethe logs and restore the CODATA leading logarithm
  and s-state constant (A41 and A40); the FAC expression omits these terms.
  This fallback still omits higher powers of Zα and is not precision QED.

The previous implementation fell back to ``F(1s)/n³`` outside the ``n <= 3``
table, which discards the ``l`` and ``j`` dependence entirely (a ``4f5/2``
and a ``4d3/2`` orbital received *identical* shifts) and has the wrong sign
for high-``l`` states.  See
``docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` findings C2 and M1.
"""

from __future__ import annotations

import re

import numpy as np

from atomx.constants import ALPHA
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend
from atomx.data.mohr_se import (
    SD_ROWS,
    SD_TABLE,
    SD_Z_GRID,
    Z_GRID,
    klamaq_f,
)
from atomx.data.mohr_se import TABLE as _SE_TABLE
from atomx.physics.dirac import hydrogenic_dirac_coulomb_expectation

# Numerical safety floor for comparing the hydrogenic ⟨-Z/r⟩ = -Z²/n² to
# the SCF value before taking their ratio.  The hydrogenic denominator is
# always non-zero for Z ≥ 1 and n ≥ 1, so this only guards against a
# future caller sending Z = 0 or n = 0; the value is arbitrary as long as
# it sits far below the smallest physical Z²/n² ~ 1 Ha.
_NUCLEAR_ATTRACTION_FLOOR = 1e-20

# Minimum peak amplitude of the SCF large-component P(r) below which we
# treat the orbital as numerically null and skip the overlap correction.
# Well below any physically normalised orbital, which peaks at O(1).
_SCF_ORBITAL_NULL_THRESHOLD = 1e-10


def _interp_table(z_grid, f_row, Z: float) -> float:
    """Shape-preserving (pchip) interpolation in Z, clamped at the ends.

    ``pchip`` rather than linear because the K/L-shell ``F`` has curvature
    between the tabulated Z nodes; the two agree to <1e-4 where the grid
    is fine (the ``_qed_za`` grid is unit-spaced) and pchip is
    monotonicity-preserving, so it cannot overshoot.
    """
    z_vals = np.asarray(z_grid, dtype=float)
    f_vals = np.asarray(f_row, dtype=float)
    Z = float(Z)
    if z_vals[0] >= Z:
        return float(f_vals[0])
    if z_vals[-1] <= Z:
        return float(f_vals[-1])
    if Z in z_vals:  # exact grid hit: no interpolation roundoff
        return float(f_vals[int(np.searchsorted(z_vals, Z))])
    try:
        from scipy.interpolate import PchipInterpolator

        return float(PchipInterpolator(z_vals, f_vals)(Z))
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return float(np.interp(Z, z_vals, f_vals))


def _interpolate_se_f(Z: float, n: int, kappa: int) -> float:
    """``F(Zα)`` for a hydrogenic ``(n, kappa)`` orbital at nuclear charge ``Z``.

    ``Z`` is allowed to be a float so that interpolation on a screened or
    effective charge remains continuous.  The selection mirrors FAC's
    ``HydrogenicSelfEnergy`` (see the module docstring).
    """
    key = (n, kappa)
    if key in _SE_TABLE:
        return _interp_table(Z_GRID, [_SE_TABLE[key][z] for z in Z_GRID], Z)
    if key in SD_ROWS:
        return _interp_table(SD_Z_GRID, SD_TABLE[SD_ROWS.index(key)], Z)
    value = float(klamaq_f(n, kappa))
    if kappa == -1:
        # CODATA 2018, Eqs. (14)-(15): s states have A41=4/3 and
        # A40=10/9-(4/3)ln k0. FAC's Klamaq expression instead supplies
        # -(4/3)ln k0+1/2, extending the non-s magnetic formula to l=0.
        # This restores the leading terms, not an all-order QED result.
        from atomx.data.nist_bethe_log import BETHE_LOG_NIST
        value = (4.0/3.0)*np.log((ALPHA*Z)**-2) + 10.0/9.0 - (4.0/3.0)*BETHE_LOG_NIST[n][0]
    return value


def _overlap_ratio(P, Q, kappa, Z, n, bsp_data):
    """SCF nuclear attraction divided by its exact Dirac-Coulomb reference.

    Using −Z²/n² here spuriously screens even an unscreened relativistic
    one-electron orbital and double-counts its relativistic contraction.
    """
    from atomx.numerics.radial_workspace import eval_orbital_pair, workspace_arrays

    gr, grw = workspace_arrays(bsp_data)
    fP, fQ = eval_orbital_pair(P, Q, bsp_data)
    rho = fP * fP + fQ * fQ

    vnuc_scf = np.sum(grw * rho * (-Z / gr))
    vnuc_hydro = hydrogenic_dirac_coulomb_expectation(n, kappa, Z)

    if abs(vnuc_hydro) < _NUCLEAR_ATTRACTION_FLOOR:
        return 1.0
    return vnuc_scf / vnuc_hydro


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@register_backend
class MohrBackend(SEBackend):
    """Tabulated Mohr F(Zα) with orbital-overlap screening."""

    name = "mohr"
    description = "Tabulated Mohr (1982/1992) F(Zα), orbital-overlap screening."

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        aZ = ALPHA * Z
        use_overlap = bsp_data is not None

        per_orbital: dict = {}
        E_SE = 0.0
        for P, Q, occ, kappa, label in orbitals:
            match = re.match(r"(\d+)", label)
            if match is None:
                raise ValueError(f"self-energy orbital has no principal n: {label!r}")
            n = int(match.group(1))
            F_val = _interpolate_se_f(Z, n, kappa)

            # E_SE (per electron) = (α/π)(Zα)^4/n^3 × F × mc²
            # In Hartree with mc² = 1/α²  ⇒  (Zα)^4 × F / (π α n^3)
            E_a = aZ ** 4 / (np.pi * ALPHA * n ** 3) * F_val

            if use_overlap and np.max(np.abs(P)) > _SCF_ORBITAL_NULL_THRESHOLD:
                ratio = _overlap_ratio(P, Q, kappa, Z, n, bsp_data)
                E_a *= ratio

            per_orbital[label] = E_a
            E_SE += occ * E_a

        return SEResult(E_SE=E_SE, per_orbital=per_orbital, method=self.name)
