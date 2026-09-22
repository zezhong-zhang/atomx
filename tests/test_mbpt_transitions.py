"""MBPT(2) transition-energy regression tests."""

import pytest

from atomx import Atom
from atomx.methods.mbpt_transitions import (
    mbpt2_orbital_energy_shift,
    mbpt2_transition_energy,
)


def test_single_occupied_orbital_returns_zero():
    """A reference with only one occupied orbital has no diagonal
    Brueckner contribution (Σ_{b≠a} sum is empty)."""
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    shift = mbpt2_orbital_energy_shift(atom._result, "1s1/2")
    assert shift == 0.0


def test_invalid_orbital_label():
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    with pytest.raises(ValueError):
        mbpt2_orbital_energy_shift(atom._result, "9z9/2")


def test_transition_energy_returns_dict_with_all_keys():
    """mbpt2_transition_energy returns the full 5-key dict."""
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    # Build a degenerate self-transition just to exercise the pipeline
    res = mbpt2_transition_energy(atom._result, "1s1/2", "1s1/2")
    for key in ("omega_DHF_au", "delta_E_alpha_au", "delta_E_beta_au",
                 "omega_corrected_au", "omega_corrected_eV"):
        assert key in res
    # Self-transition: ω = 0
    assert res["omega_DHF_au"] == 0.0
    assert res["omega_corrected_au"] == 0.0


def test_transition_energy_self_consistent():
    """For closed-shell single-orbital reference the corrected ω
    equals the DHF ω."""
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    res = mbpt2_transition_energy(atom._result, "1s1/2", "1s1/2")
    assert res["omega_corrected_au"] == pytest.approx(res["omega_DHF_au"],
                                                       abs=1e-12)
