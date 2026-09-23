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


def test_transition_energy_is_exactly_zero_for_a_single_orbital_reference():
    """He 1s2: one occupied orbital, so both shifts are 0 by construction.

    Explicitly a *definitional* case: the previous name
    (``test_transition_energy_self_consistent``) and its assertion
    ``omega_corrected_au == omega_DHF_au`` read as a physics gate, but the
    test above has already established that both sides are exactly 0.0
    (audit finding F13: 0 == 0).

    The non-zero case is NOT covered, and cannot be cheaply: the diagonal
    MBPT(2) self-energy needs an atom with two occupied subshells, and
    ``mbpt2_orbital_energy_shift`` builds a full virtual basis per kappa —
    measured > 4 min for Be 2s1/2 and > 8 min for Ne 2p on this machine,
    against a 4-second file and a ~550 s suite.  Audit finding F16 stays
    open with that measurement; what this file gates today is the API
    contract, the exact-zero limit, and the unit conversion.
    """
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    res = mbpt2_transition_energy(atom._result, "1s1/2", "1s1/2")
    assert res["delta_E_alpha_au"] == 0.0
    assert res["delta_E_beta_au"] == 0.0
    assert res["omega_DHF_au"] == 0.0
    assert res["omega_corrected_au"] == 0.0
    assert res["omega_corrected_eV"] == 0.0
