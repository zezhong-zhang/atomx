"""FAC/Indelicato-Mohr hydrogenic self-energy backend.

The values below reproduce the diagonal K/L-shell self-energy functions
used by FAC 1.1.x in ``faclib/coulomb.c``:

* ``_qed_se0``: point-nucleus F(alpha Z) remainder table.
* ``_qed_se1``: FAC finite-nuclear-size delta for the tabulated Fermi
  radii.  FAC applies Mohr's finite-size prescription when a different
  rms radius is requested; atomx currently has no per-call rms in the
  self-energy backend interface, so this backend uses FAC's tabulated
  Fermi-radius correction for the default finite-size path.

Table provenance: P. Indelicato and P. J. Mohr, Phys. Rev. A 58, 165
(1998); V. M. Shabaev, I. I. Tupitsyn, and V. A. Yerokhin, Phys. Rev. A
88, 012513 (2013), as compiled in FAC ``faclib/coulomb.c``.  Each row
comment names the exact FAC static array source.
"""

from __future__ import annotations

import numpy as np

from atomx.constants import ALPHA
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend
from atomx.corrections.qedmod.mohr import _overlap_ratio

# ``TABLE[(n, kappa)][Z] -> F(alpha Z)``.
# Rows are FAC ``faclib/coulomb.c:_qed_se0`` + ``_qed_se1`` at the
# tabulated Fermi nuclear radius.  Grid values are every 10 charge units;
# the backend uses a shape-preserving cubic spline when SciPy is present
# and linear interpolation otherwise.
TABLE: dict[tuple[int, int], dict[int, float]] = {
    # 1s1/2: FAC _qed_se0[0] + _qed_se1[0]
    (1, -1): {
        10: 4.65413, 20: 3.24616, 30: 2.55177, 40: 2.13475,
        50: 1.86331, 60: 1.68201, 70: 1.56368, 80: 1.49549,
        90: 1.47183, 100: 1.49613,
    },
    # 2s1/2: FAC _qed_se0[1] + _qed_se1[1]
    (2, -1): {
        10: 4.89439, 20: 3.50655, 30: 2.83860, 40: 2.45432,
        50: 2.22327, 60: 2.09234, 70: 2.03810, 80: 2.05371,
        90: 2.14267, 100: 2.32470,
    },
    # 2p1/2: FAC _qed_se0[2] + _qed_se1[2]
    (2, 1): {
        10: -0.11484, 20: -0.09252, 30: -0.06433, 40: -0.03105,
        50: 0.00800, 60: 0.05471, 70: 0.11256, 80: 0.18767,
        90: 0.29115, 100: 0.44501,
    },
    # 2p3/2: FAC _qed_se0[3] + _qed_se1[3]
    (2, -2): {
        10: 0.13036, 20: 0.14384, 30: 0.16065, 40: 0.17960,
        50: 0.20005, 60: 0.22162, 70: 0.24404, 80: 0.26707,
        90: 0.29038, 100: 0.31353,
    },
}

_SCF_ORBITAL_NULL_THRESHOLD = 1e-10


def _interpolate_indelicato_f(Z: float, n: int, kappa: int) -> float:
    """Interpolate FAC/Indelicato F(alpha Z) for K/L-shell states."""
    key = (n, kappa)
    if key not in TABLE:
        f_1s = _interpolate_indelicato_f(Z, 1, -1)
        return f_1s / n ** 3

    table = TABLE[key]
    if Z in table:
        return table[Z]

    z_vals = np.array(sorted(table), dtype=float)
    f_vals = np.array([table[int(z)] for z in z_vals], dtype=float)
    z = float(np.clip(Z, z_vals[0], z_vals[-1]))

    try:
        from scipy.interpolate import PchipInterpolator

        return float(PchipInterpolator(z_vals, f_vals)(z))
    except Exception:
        return float(np.interp(z, z_vals, f_vals))


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
            n = int(label[0])
            F_val = _interpolate_indelicato_f(Z, n, kappa)
            E_a = aZ ** 4 / (np.pi * ALPHA * n ** 3) * F_val

            if use_overlap and np.max(np.abs(P)) > _SCF_ORBITAL_NULL_THRESHOLD:
                E_a *= _overlap_ratio(P, Q, kappa, Z, n, bsp_data)

            per_orbital[label] = E_a
            E_SE += occ * E_a

        return SEResult(E_SE=E_SE, per_orbital=per_orbital, method=self.name)
