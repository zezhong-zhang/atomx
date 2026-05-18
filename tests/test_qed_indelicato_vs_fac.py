import math

import pytest

from atomx.constants import ALPHA
from atomx.corrections.qedmod.base import get_backend
from atomx.corrections.qedmod.indelicato import _interpolate_indelicato_f


FAC_F = {
    (54, 1, -1): 1.7819343084925525,
    (54, 2, -1): 2.160477504261656,
    (74, 1, -1): 1.5305906787997485,
    (74, 2, -1): 2.041045004066176,
    (92, 1, -1): 1.4728122399999999,
    (92, 2, -1): 2.1695846802553596,
}

GROUND_LEVELS = [
    (54, "H-like Xe", [(1, -1, 1.0, "1s")]),
    (54, "He-like Xe", [(1, -1, 2.0, "1s")]),
    (54, "Li-like Xe", [(1, -1, 2.0, "1s"), (2, -1, 1.0, "2s")]),
    (74, "H-like W", [(1, -1, 1.0, "1s")]),
    (74, "He-like W", [(1, -1, 2.0, "1s")]),
    (74, "Li-like W", [(1, -1, 2.0, "1s"), (2, -1, 1.0, "2s")]),
    (92, "H-like U", [(1, -1, 1.0, "1s")]),
    (92, "He-like U", [(1, -1, 2.0, "1s")]),
    (92, "Li-like U", [(1, -1, 2.0, "1s"), (2, -1, 1.0, "2s")]),
]


def _fac_self_energy(Z, n, kappa):
    f_val = FAC_F[(Z, n, kappa)]
    return (ALPHA * Z) ** 4 / (math.pi * ALPHA * n ** 3) * f_val


@pytest.mark.parametrize(
    "Z, expected_f",
    [
        (54, 1.78314 - 0.00125),
        (74, 1.53568 - 0.00484),
        (82, 1.49569 - 0.00840),
        (92, 1.49092 - 0.01841),
    ],
)
def test_indelicato_1s_matches_fac_static_table(Z, expected_f):
    actual = _interpolate_indelicato_f(Z, 1, -1)
    assert actual == pytest.approx(expected_f, rel=5e-3)


@pytest.mark.parametrize("Z", [54, 74, 82, 92])
def test_indelicato_is_high_z_closeout_relative_to_mss(Z):
    """FAC high-Z 1s F values are the physical ~1.5-1.8 range, not the
    historic MSS table's >15 high-Z extrapolation."""
    from atomx.corrections.qedmod.mohr import _interpolate_se_f

    fac_f = _interpolate_indelicato_f(Z, 1, -1)
    mss_f = _interpolate_se_f(Z, 1, -1)
    assert fac_f > 1.0
    assert fac_f < 2.0
    assert mss_f / fac_f > 8.0


@pytest.mark.parametrize("Z, iso, shells", GROUND_LEVELS)
def test_indelicato_matches_fac_high_z_ground_level_rows(Z, iso, shells):
    backend = get_backend("indelicato")
    orbitals = [
        (None, None, occupation, kappa, label)
        for n, kappa, occupation, label in shells
    ]

    actual = backend.compute(orbitals, bsp_data=None, Z=Z)
    expected_total = sum(
        occupation * _fac_self_energy(Z, n, kappa)
        for n, kappa, occupation, label in shells
    )

    assert actual.E_SE == pytest.approx(expected_total, rel=5e-7), iso
    assert actual.per_orbital["1s"] == pytest.approx(_fac_self_energy(Z, 1, -1), rel=5e-7), iso


@pytest.mark.parametrize("Z", [54, 74, 92])
def test_leading_1s_residual_percent_vs_fac_table_is_pinned(Z):
    backend = get_backend("indelicato")
    actual = backend.compute([(None, None, 1.0, -1, "1s")], bsp_data=None, Z=Z)
    expected = _fac_self_energy(Z, 1, -1)
    residual_pct = abs(actual.per_orbital["1s"] - expected) / abs(expected) * 100.0

    assert residual_pct <= 0.5
