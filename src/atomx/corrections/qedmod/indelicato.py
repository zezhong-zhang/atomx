"""FAC/Indelicato-Mohr hydrogenic self-energy backend.

Historical backend name.  It now reads the *same* corrected FAC tables as
:mod:`atomx.corrections.qedmod.mohr` (``faclib/coulomb.c:_qed_se0`` +
``_qed_se1`` for the K/L shells), which is what makes the ``"auto"``
policy's switch at ``Z = 37`` harmless: before the tables were unified,
``auto`` jumped between two mutually inconsistent parameterisations and
the 1s self-energy dropped by 65 eV across that boundary.

Table provenance: P. Indelicato and P. J. Mohr, Phys. Rev. A 58, 165
(1998); V. M. Shabaev, I. I. Tupitsyn, and V. A. Yerokhin, Phys. Rev. A
88, 012513 (2013), as compiled in FAC ``faclib/coulomb.c``.  The single
source of truth is :mod:`atomx.data.mohr_se`.
"""

from __future__ import annotations

import re

import numpy as np

from atomx.constants import ALPHA
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend
from atomx.corrections.qedmod.mohr import _overlap_ratio
from atomx.data.mohr_se import TABLE as _SE_TABLE
from atomx.data.mohr_se import Z_GRID

# ``TABLE[(n, kappa)][Z] -> F(alpha Z)``.
# Rows are FAC ``faclib/coulomb.c:_qed_se0`` + ``_qed_se1`` at the
# tabulated Fermi nuclear radius. The interpolation is shape-preserving
# PCHIP with SciPy as a required dependency, not a linear fallback.
# ``TABLE[(n, kappa)][Z] -> F(alpha Z)`` for the K/L shells, derived from
# the single tabulated source (:mod:`atomx.data.mohr_se`).  Kept as a
# public name for callers that introspect the coverage.
TABLE: dict[tuple[int, int], dict[int, float]] = {
    key: {z: row[z] for z in Z_GRID}
    for key, row in _SE_TABLE.items()
}

_SCF_ORBITAL_NULL_THRESHOLD = 1e-10


def _interpolate_indelicato_f(Z: float, n: int, kappa: int) -> float:
    """FAC/Indelicato ``F(Zα)``.

    This is now an alias for :func:`atomx.corrections.qedmod.mohr._interpolate_se_f`:
    both backends read the same corrected FAC tables, so the ``"auto"``
    policy's historical switch at ``Z = 37`` no longer produces a
    discontinuity in a physical observable (it used to drop the 1s
    self-energy by 65 eV between ``Z = 36`` and ``Z = 37``).
    """
    from atomx.corrections.qedmod.mohr import _interpolate_se_f

    return _interpolate_se_f(Z, n, kappa)


@register_backend
class IndelicatoBackend(SEBackend):
    """FAC/Indelicato-Mohr K/L-shell self-energy table."""

    name = "indelicato"
    description = "FAC/Indelicato-Mohr high-Z K/L-shell self-energy table."

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
            F_val = _interpolate_indelicato_f(Z, n, kappa)
            E_a = aZ ** 4 / (np.pi * ALPHA * n ** 3) * F_val

            if use_overlap and np.max(np.abs(P)) > _SCF_ORBITAL_NULL_THRESHOLD:
                E_a *= _overlap_ratio(P, Q, kappa, Z, n, bsp_data)

            per_orbital[label] = E_a
            E_SE += occ * E_a

        return SEResult(E_SE=E_SE, per_orbital=per_orbital, method=self.name)
