"""Round-trip integrity of the :mod:`atomx.data` submodule."""

from __future__ import annotations


def test_mohr_table_values_match_fac():
    """``F(Zα)`` spot values against FAC ``faclib/coulomb.c``.

    This module used to pin the atomx <= 0.2 literals (``10.3169``,
    ``21.6412``, ...).  Those agreed with the published hydrogenic
    self-energy function only at Z=1 and had the wrong Z-dependence
    everywhere else (2.5x at Z=10, 14.7x at Z=92), so "stability" here
    meant "stays wrong".  The values are now FAC's, and the physics
    gates live in ``tests/test_qed_self_energy_table.py``.
    """
    from atomx.data.mohr_se import SD_ROWS, SD_TABLE, SD_Z_GRID, TABLE, Z_GRID

    assert TABLE[(1, -1)][1] == 10.31679          # FAC _qed_se0[0][0]
    assert TABLE[(1, -1)][92] == 1.47251          # FAC _qed_se0[0][91]+se1
    assert TABLE[(2, -1)][10] == 4.89439
    assert len(Z_GRID) == 112 and Z_GRID[0] == 1 and Z_GRID[-1] == 120
    # n=3..5 rows come from FAC _sd; 3d5/2 is positive, 3d3/2 negative.
    i52 = SD_ROWS.index((3, -3))
    i32 = SD_ROWS.index((3, 2))
    assert SD_TABLE[i52][0] == 0.04
    assert SD_TABLE[i32][0] == -0.044
    assert SD_Z_GRID == (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                         110, 115, 120)


def test_mohr_interpolator_matches_table_values():
    """Calling ``_interpolate_se_f`` at a tabulated (Z, n, κ) reproduces
    the tabulated value exactly (no interpolation error)."""
    from atomx.corrections.qedmod.mohr import _interpolate_se_f
    from atomx.data.mohr_se import TABLE

    for (n, kappa), row in TABLE.items():
        for Z, F_expected in row.items():
            assert _interpolate_se_f(Z, n, kappa) == F_expected


def test_hyperfine_presets_structure_matches_legacy_tuple():
    """``_hyperfine_preset('h1s')`` still returns the canonical
    4-tuple layout that the SCF ladder code consumes."""
    from atomx.methods.scf_observables import _hyperfine_preset

    assert _hyperfine_preset("h1s") == (True, -95.0e-6, -2.0e-6, -41.0e-6)
    assert _hyperfine_preset("d1s") == (True, -95.0e-6, -0.5e-6, -138.0e-6)
    assert _hyperfine_preset("he3_1s") == (True, -95.0e-6, -4.0e-5, -210.0e-6)


def test_flambaum_ginges_constants_match_paper_values():
    """The Flambaum–Ginges fit constants ported to the data module
    agree with the values originally inlined into the backend."""
    from atomx.data.flambaum_ginges import (
        A_CUTOFF_COEFF,
        A_HIGH_FREQ_COEFFS,
        A_HIGH_FREQ_Z_REF,
        B_LOW_FREQ_CONST,
        B_LOW_FREQ_LINEAR,
        WK_DENOM_SCALE,
        WK_LEADING,
    )

    assert (B_LOW_FREQ_CONST, B_LOW_FREQ_LINEAR) == (0.074, 0.35)
    assert A_HIGH_FREQ_COEFFS == (1.071, -1.976, -2.128, 0.169)
    assert A_HIGH_FREQ_Z_REF == 80
    assert A_CUTOFF_COEFF == 0.07
    assert (WK_LEADING, WK_DENOM_SCALE) == (0.092, 1.62)


def test_nuclear_structure_constants_round_trip():
    """The Zemach / polarisability re-exports in
    :mod:`atomx.properties.hyperfine.qed_corrections` still resolve to the
    original CODATA / Carlson numbers via the data module."""
    from atomx.properties.hyperfine.qed_corrections import (
        DEUTERON_POLARIZABILITY_D1S,
        DEUTERON_ZEMACH_RADIUS_FM,
        HELIUM3_POLARIZABILITY,
        HELIUM3_ZEMACH_RADIUS_FM,
        PROTON_POLARIZABILITY_H1S,
        PROTON_ZEMACH_RADIUS_FM,
    )

    assert PROTON_ZEMACH_RADIUS_FM == 1.054
    assert DEUTERON_ZEMACH_RADIUS_FM == 2.593
    assert HELIUM3_ZEMACH_RADIUS_FM == 2.528
    assert PROTON_POLARIZABILITY_H1S == 1.3e-6
    assert DEUTERON_POLARIZABILITY_D1S == 1.7e-6
    assert HELIUM3_POLARIZABILITY == 3.1e-6


def test_data_load_dispatcher_finds_every_table():
    """:func:`atomx.data.load` can resolve each registered table."""
    from atomx.data import available_tables, load

    for name in available_tables():
        payload = load(name)
        assert "CITATION" in payload
        assert isinstance(payload["CITATION"], str) and payload["CITATION"]
