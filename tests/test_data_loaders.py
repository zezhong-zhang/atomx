"""Round-trip integrity of the :mod:`atomx.data` submodule."""

from __future__ import annotations


def test_mohr_table_values_are_stable():
    """Mohr F(Zα) table values that were hard-coded in atomx ≤ 0.2
    continue to match their original data module."""
    from atomx.data.mohr_se import TABLE, Z_GRID

    # Spot-check the endpoints and a few interior values.
    assert TABLE[(1, -1)][1] == 10.3169
    assert TABLE[(1, -1)][92] == 21.6412
    assert TABLE[(2, -1)][10] == 11.8666
    assert TABLE[(3, -3)][92] == -0.038
    assert Z_GRID == (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 92)


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
    from atomx.methods.dirac_fock import _hyperfine_preset

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
    :mod:`atomx.prop.hyperfine.qed_corrections` still resolve to the
    original CODATA / Carlson numbers via the data module."""
    from atomx.prop.hyperfine.qed_corrections import (
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
