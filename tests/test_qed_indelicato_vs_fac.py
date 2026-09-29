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
def test_mohr_and_indelicato_tables_are_the_same_function(Z):
    """The two self-energy backends must agree — there is one table.

    Historically they did not: ``data/mohr_se.py`` held values that grew
    with Z (F = 15.6 at Z=54, 21.6 at Z=92) while FAC's ``_qed_se0`` falls
    (1.78, 1.47), so ``qed_backend="auto"`` produced a 6.3x discontinuity
    in the 1s self-energy at Z=36->37.  Both now read
    :mod:`atomx.data.mohr_se`.  See
    docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md finding C2.
    """
    from atomx.corrections.qedmod.mohr import _interpolate_se_f

    fac_f = _interpolate_indelicato_f(Z, 1, -1)
    mohr_f = _interpolate_se_f(Z, 1, -1)
    # The physical range: the 1s self-energy function falls with Z.
    assert 1.0 < fac_f < 2.0
    assert mohr_f == pytest.approx(fac_f, rel=1e-12)
    # And it must decrease monotonically across this panel.
    f_low = _interpolate_se_f(54, 1, -1)
    f_high = _interpolate_se_f(92, 1, -1)
    assert f_high < f_low


@pytest.mark.parametrize("n,kappa,label", [
    (10, 1, "10p1/2"),
    (12, -2, "12p3/2"),
])
def test_indelicato_reads_full_multi_digit_principal_n(n, kappa, label):
    backend = get_backend("indelicato")
    actual = backend.compute([(None, None, 1.0, kappa, label)],
                             bsp_data=None, Z=42)
    expected = ((ALPHA * 42) ** 4 / (math.pi * ALPHA * n ** 3)
                * _interpolate_indelicato_f(42, n, kappa))
    assert actual.per_orbital[label] == pytest.approx(expected, rel=1e-12)


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

    # ``FAC_F`` above are FAC *run* outputs, which include FAC's MOHRFIN
    # correction for the element's actual nuclear rms radius.  atomx uses
    # the tabulated-radius values (it has no per-call rms in the SE
    # interface), a convention difference of <= 2.5e-4.  The gate is
    # therefore 1e-3 rather than 5e-7; the tabulated-radius values
    # themselves are pinned to 5e-3 by
    # ``test_indelicato_1s_matches_fac_static_table``.
    assert pytest.approx(expected_total, rel=1e-3) == actual.E_SE, iso
    assert actual.per_orbital["1s"] == pytest.approx(
        _fac_self_energy(Z, 1, -1), rel=1e-3), iso


@pytest.mark.parametrize("Z", [54, 74, 92])
def test_leading_1s_residual_percent_vs_fac_table_is_pinned(Z):
    backend = get_backend("indelicato")
    actual = backend.compute([(None, None, 1.0, -1, "1s")], bsp_data=None, Z=Z)
    expected = _fac_self_energy(Z, 1, -1)
    residual_pct = abs(actual.per_orbital["1s"] - expected) / abs(expected) * 100.0

    assert residual_pct <= 0.5
