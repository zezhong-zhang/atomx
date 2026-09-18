"""Physical constants — CODATA 2022 (NIST 2024) by default.

atomx retains CODATA 2018 values for historical reproducibility but
defaults to CODATA 2022.  These tests pin the active values and also
confirm that switching to the 2018 table gives the legacy numbers.
"""

import pytest

from atomx.constants import (
    ALPHA,
    BOHR_RADIUS_M,
    CODATA,
    DEFAULT_CODATA_VERSION,
    FINE_STRUCTURE_CONSTANT,
    HARTREE_CM1,
    HARTREE_EV,
    SPEED_OF_LIGHT,
)


def test_default_codata_is_2022():
    """CODATA 2022 (NIST 2024) is the default recommendation."""
    assert DEFAULT_CODATA_VERSION == "2022"


# ---------------------------------------------------------------------------
# Active constants (CODATA 2022)
# ---------------------------------------------------------------------------

def test_hartree_ev_codata2022():
    # 1 Ha / 1 eV from CODATA 2022 = 27.211386245981 eV (±1e-10)
    assert pytest.approx(27.211386245981, rel=2e-10) == HARTREE_EV


def test_hartree_cm1_codata2022():
    # 1 Ha in cm^-1  ≈ 219474.6313632
    assert pytest.approx(219474.6313632, rel=1e-9) == HARTREE_CM1


def test_fine_structure_constant_codata2022():
    # CODATA 2022: α = 7.2973525643e-3
    assert pytest.approx(7.2973525643e-3, rel=1e-10) == FINE_STRUCTURE_CONSTANT


def test_speed_of_light_inverse_alpha():
    """c = 1/α in atomic units."""
    assert pytest.approx(1.0 / ALPHA, rel=1e-14) == SPEED_OF_LIGHT


def test_bohr_radius_codata2022():
    # CODATA 2022: a_0 = 5.29177210544 e-11 m
    assert pytest.approx(5.29177210544e-11, rel=1e-10) == BOHR_RADIUS_M


# ---------------------------------------------------------------------------
# Legacy CODATA 2018 table access
# ---------------------------------------------------------------------------

def test_codata_2018_still_available():
    """CODATA 2018 values retained in the CODATA dict."""
    assert "2018" in CODATA
    d = CODATA["2018"]
    assert d["_alpha"] == pytest.approx(7.2973525693e-3, rel=1e-10)
    assert d["_a0"] == pytest.approx(5.29177210903e-11, rel=1e-10)


def test_codata_2022_and_2018_differ_at_ppb_level():
    """The two tables should differ at the ~10^-9 level (measured
    constants refined; defined constants identical)."""
    a18 = CODATA["2018"]["_alpha"]
    a22 = CODATA["2022"]["_alpha"]
    diff = abs(a22 - a18) / a18
    # ~10^-9 level
    assert 1e-10 < diff < 1e-6

    # Defined constants agree exactly
    for key in ("_c", "_hplanck", "_e", "_k", "_Nav"):
        assert CODATA["2018"][key] == CODATA["2022"][key]
