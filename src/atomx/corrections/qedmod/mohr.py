"""Mohr tabulated self-energy backend.

The numerical F(Zα) table lives in :mod:`atomx.data.mohr_se` with full
citations; this module owns only the interpolation logic and the
SCF-vs-hydrogenic overlap screening that convert that table into a
per-orbital self-energy shift.

Orbitals outside the explicit table (n ≥ 4, higher-l) fall back to the
hydrogenic n⁻³ scaling of the 1s1/2 value — this is the intrinsic
limit of the Mohr tabulation; use the ``welton``, ``shabaev_local``
or ``flambaum_ginges`` backends for systematically better coverage.
"""

from __future__ import annotations

import numpy as np

from atomx.constants import ALPHA
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend
from atomx.data.mohr_se import TABLE as _SE_TABLE


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


def _interpolate_se_f(Z: float, n: int, kappa: int) -> float:
    """Interpolate F(Zα) from the :mod:`atomx.data.mohr_se` table.

    Linear in Z within the grid; hydrogenic n⁻³ fallback outside the
    table.  ``Z`` is allowed to be a float so that continuous
    interpolation on nuclear screening or finite-Z corrections can be
    supported by callers.
    """
    key = (n, kappa)
    if key in _SE_TABLE:
        table = _SE_TABLE[key]
        # Exact grid hit: return the tabulated value verbatim to avoid
        # floating-point roundoff from the two-point formula below.
        if Z in table:
            return table[Z]

        z_vals = sorted(table.keys())
        f_vals = [table[z] for z in z_vals]

        if Z <= z_vals[0]:
            return f_vals[0]
        if Z >= z_vals[-1]:
            return f_vals[-1]

        for i in range(len(z_vals) - 1):
            if z_vals[i] <= Z <= z_vals[i + 1]:
                t = (Z - z_vals[i]) / (z_vals[i + 1] - z_vals[i])
                return f_vals[i] + t * (f_vals[i + 1] - f_vals[i])

    # Fallback: hydrogenic n^-3 scaling of the 1s value
    f_1s = _interpolate_se_f(Z, 1, -1)
    return f_1s / n ** 3


def _overlap_ratio(P, Q, kappa, Z, n, bsp_data):
    """⟨ψ_SCF|-Z/r|ψ_SCF⟩ / ⟨ψ_hydro|-Z/r|ψ_hydro⟩ with the hydrogenic value
    taken as -Z^2/n^2 (point-nucleus leading term)."""
    from atomx.corrections.workspace import eval_orbital_pair, workspace_arrays

    gr, grw = workspace_arrays(bsp_data)
    fP, fQ = eval_orbital_pair(P, Q, bsp_data)
    rho = fP * fP + fQ * fQ

    vnuc_scf = np.sum(grw * rho * (-Z / gr))
    vnuc_hydro = -Z * Z / (n * n)

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
    description = "Tabulated Mohr (1982/1992) F(Zα), hydrogenic n^-3 fallback."

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        aZ = ALPHA * Z
        use_overlap = bsp_data is not None

        per_orbital: dict = {}
        E_SE = 0.0
        for P, Q, occ, kappa, label in orbitals:
            n = int(label[0])
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
