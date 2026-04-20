"""Tabulated parameters for FAC/Gu's local-exchange functional.

The two length-30 tables ``_HX0[k]`` and ``_HX1[k]`` are indexed by
principal quantum number ``n = 2..31`` (``n-2`` as array index, clipped
at ``NHXN-1 = 29``). FAC's ``GetHXS`` chooses between branch ``k = 0``
(small ``|ihx|``) and ``k = 1`` (large ``|ihx|``, the default) and
evaluates

    α_xc(Z, N) = clip(_HX0[k][n-2] + _HX1[k][n-2] · Z², 0.1, 0.9)

Source: ``fac/faclib/radial.c:44-77`` (static arrays ``_hx0``, ``_hx1``),
verified bit-identical by ``docs/superpowers/plans/2026-04-20-fac-calibration``.

These tables are Gu's empirical fit of the local-exchange coefficient
against full Dirac-Hartree-Fock orbital energies; they are what make
FAC's local-exchange DFS better than pure Slater (α = 2/3) or Kohn-Sham
(α = 1) at reproducing DHF results.
"""

from __future__ import annotations

import numpy as np

#: Number of tabulated rows (n = 2..31).
NHXN: int = 30

#: Constant term of α_xc(Z, N), per branch (``k``) and per ``n-2``.
_HX0: np.ndarray = np.array(
    [
        [
            0.49464443, 0.49448420, 0.47671812, 0.47368338, 0.46881982,
            0.46506770, 0.46239243, 0.45992017, 0.45810070, 0.45699729,
            0.45327464, 0.45147903, 0.45018784, 0.44901484, 0.44783271,
            0.44677685, 0.44601044, 0.44668505, 0.44690311, 0.44747163,
            0.44809634, 0.44896082, 0.45032069, 0.45146050, 0.45283192,
            0.45420841, 0.45580976, 0.45472978, 0.45249835, 0.45000000,
        ],
        [
            0.60392457, 0.59638380, 0.54396742, 0.53136923, 0.53471311,
            0.53206107, 0.52694942, 0.52089717, 0.51642186, 0.53134076,
            0.48894837, 0.47899983, 0.47580524, 0.47583937, 0.47630660,
            0.47516619, 0.47327929, 0.47544085, 0.47804916, 0.48123176,
            0.48489680, 0.48861150, 0.49402954, 0.49848388, 0.50336365,
            0.50842387, 0.51389326, 0.53564154, 0.55618328, 0.55000000,
        ],
    ]
)

#: Z² slope of α_xc(Z, N), per branch (``k``) and per ``n-2``.
_HX1: np.ndarray = np.array(
    [
        [
            -2.24999797e-05, -2.09402638e-05, -1.97068001e-05, -1.86529759e-05,
            -1.73178539e-05, -1.95629593e-05, -2.21086675e-05, -2.11895870e-05,
            -2.01426809e-05, -1.98210143e-05, -1.92998222e-05, -1.86920564e-05,
            -1.82455335e-05, -1.84696909e-05, -1.87351162e-05, -1.80283589e-05,
            -1.73407781e-05, -1.69192139e-05, -1.63363906e-05, -1.58294488e-05,
            -1.53107601e-05, -1.53014957e-05, -1.54457903e-05, -1.50586829e-05,
            -1.47144622e-05, -1.43879181e-05, -1.40649716e-05, -1.40581261e-05,
            -1.38378555e-05, -1.40000000e-05,
        ],
        [
            -4.83974367e-05, -2.69133100e-05, -2.41090948e-05, -2.21989673e-05,
            -1.94076264e-05, -1.97061814e-05, -2.07157339e-05, -1.88678066e-05,
            -1.77080751e-05, -2.23815832e-05, -1.76272492e-05, -1.70901358e-05,
            -1.70196986e-05, -1.72463938e-05, -1.72980241e-05, -1.61671348e-05,
            -1.50179117e-05, -1.48463176e-05, -1.47797417e-05, -1.47793769e-05,
            -1.48393532e-05, -1.53064076e-05, -1.62570909e-05, -1.63962294e-05,
            -1.65777367e-05, -1.67306107e-05, -1.69330862e-05, -1.95968009e-05,
            -2.12996663e-05, -2.00000000e-05,
        ],
    ]
)

#: The free-electron-gas exchange coefficient ``(3/(4π²))^(1/3) ≈ 0.4236``.
#: FAC uses ``hxs <= -5`` as a sentinel for "return this value instead".
SLATER_ALPHA_LDA: float = float((3.0 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0))


def hxs_row(k: int, n: int) -> tuple[float, float]:
    """Return ``(HX0[k, n-2], HX1[k, n-2])`` with the FAC index clip.

    FAC's code clips ``n`` to ``NHXN``; for ``n > 31`` it reuses the
    last row. We replicate that exactly.
    """
    if k not in (0, 1):
        raise ValueError(f"branch k must be 0 or 1, got {k}")
    idx = min(max(n - 2, 0), NHXN - 1)
    return float(_HX0[k, idx]), float(_HX1[k, idx])
