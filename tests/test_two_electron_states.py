"""Unit tests for the shared two-electron LS term layer (Milestone 0)."""

from __future__ import annotations

import types
import typing

import pytest

from atomx.constants import HARTREE_EV
from atomx.scattering.r_matrix.target_states import (
    TargetState,
    helike_target_states,
)
from atomx.scattering.two_electron.states import TwoElectronTerm, helike_terms

# FAC reference: He-like Ne (Z=10) level energies relative to the 1s^2 ground.
# Generated with FAC 1.1.5, no Breit/QED, structure only.
_FAC_NE_X_RELATIVE_EV = {
    "1s^2 ^1S": 0.0,
    "1s2s ^3S": 904.38,
    "1s2s ^1S": 915.33,
    "1s2p ^3P": 914.25,  # average of J=0,1,2 levels around 914.1 eV
    "1s2p ^1P": 922.02,
}
_FAC_NE_X_GROUND_EV = -2557.39


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED (Z=10, n_grid=2000): the hydrogenic-plus-first-order-Coulomb "
    "model puts the excited terms 45-54 eV below FAC 1.1.5 (1s2s 3S 850.35 "
    "vs 904.38, 1s2s 1S 862.29 vs 915.33, 1s2p 3P 867.60 vs 914.25, "
    "1s2p 1P 876.89 vs 922.02).  The gate is tightened from the old "
    "rel=0.10 (+-90 eV, which cannot distinguish anything at this scale) to "
    "the 20 eV at which the He-like excitation scale actually lives, so it "
    "is red and stays red: the missing physics is the Z=10 screening the "
    "model does not carry, not a tolerance.  The orbital-insensitive "
    "singlet-triplet splittings ARE reproduced and are gated by "
    "test_helike_ne_singlet_triplet_splittings_match_fac."))
@pytest.mark.parametrize(
    "label",
    ["1s2s ^3S", "1s2s ^1S", "1s2p ^3P", "1s2p ^1P"],
)
def test_helike_ne_term_energies_vs_fac(label: str):
    """Absolute excitation energies vs FAC, at the 20 eV physics scale."""
    terms = helike_terms(Z=10, n_max=2, n_grid=2000)
    term_map = {t.label: t for t in terms}
    assert label in term_map, f"missing term {label}"
    term = term_map[label]
    atomx_ground_ev = term_map["1s^2 ^1S"].energy_au * HARTREE_EV
    e_rel_ev = term.energy_au * HARTREE_EV - atomx_ground_ev
    fac_rel = _FAC_NE_X_RELATIVE_EV[label]
    assert e_rel_ev == pytest.approx(fac_rel, abs=20.0), (
        f"{label}: atomx E_rel={e_rel_ev:.2f} eV, FAC={fac_rel:.2f} eV"
    )


def test_helike_ne_singlet_triplet_splittings_match_fac():
    """The splittings the hydrogenic model gets right, to 2 eV.

    ``E(1s2s 1S) - E(1s2s 3S)`` and ``E(1s2p 1P) - E(1s2p 3P)`` are set by
    the exchange integral, which the model evaluates well even where the
    absolute term positions are 50 eV low (measured: 11.94 vs FAC 10.95 and
    9.29 vs 7.77 eV).  These are the orbital-insensitive gates; FAC's
    absolute scale is gated by the xfail above.
    """
    terms = helike_terms(Z=10, n_max=2, n_grid=2000)
    tm = {t.label: t.energy_au * HARTREE_EV for t in terms}
    s_splitting = tm["1s2s ^1S"] - tm["1s2s ^3S"]
    p_splitting = tm["1s2p ^1P"] - tm["1s2p ^3P"]
    fac_s = (_FAC_NE_X_RELATIVE_EV["1s2s ^1S"]
             - _FAC_NE_X_RELATIVE_EV["1s2s ^3S"])
    fac_p = (_FAC_NE_X_RELATIVE_EV["1s2p ^1P"]
             - _FAC_NE_X_RELATIVE_EV["1s2p ^3P"])
    assert s_splitting == pytest.approx(fac_s, abs=2.0), (s_splitting, fac_s)
    assert p_splitting == pytest.approx(fac_p, abs=2.0), (p_splitting, fac_p)


def test_helike_terms_count():
    """For n_max=2 we expect 1s^2 + 1s2s (singlet/triplet) + 1s2p (singlet/triplet)."""
    terms = helike_terms(Z=6, n_max=2)
    labels = [t.label for t in terms]
    assert labels == [
        "1s^2 ^1S",
        "1s2s ^3S",
        "1s2s ^1S",
        "1s2p ^3P",
        "1s2p ^1P",
    ]


def test_helike_parity():
    terms = helike_terms(Z=6, n_max=2)
    for term in terms:
        assert term.parity == (-1) ** term.L


def test_helike_target_states_wrap_terms():
    """``helike_target_states`` should produce TargetState objects carrying terms."""
    targets = helike_target_states(Z=6, n_max=2)
    assert set(targets.keys()) == {
        "1s^2 ^1S",
        "1s2s ^3S",
        "1s2s ^1S",
        "1s2p ^3P",
        "1s2p ^1P",
    }
    ground = targets["1s^2 ^1S"]
    assert ground.S_t == 0.0
    assert ground.l == 0
    assert ground.two_electron_term is not None
    excited = targets["1s2p ^1P"]
    assert excited.S_t == 0.0
    assert excited.l == 1
    assert excited.two_electron_term.label == "1s2p ^1P"


def test_target_state_type_hints_resolve_two_electron_term():
    """Regression (HYP-1688): ``get_type_hints(TargetState)`` must resolve the
    ``two_electron_term`` forward reference to the real ``TwoElectronTerm``
    class instead of raising ``NameError``.

    ``TwoElectronTerm`` lives in a module that also consumes ``TargetState``,
    so the annotation is only resolvable if the import cycle between
    ``r_matrix.target_states`` and ``two_electron.states`` stays broken.
    """
    hints = typing.get_type_hints(TargetState)
    assert "two_electron_term" in hints
    hint = hints["two_electron_term"]
    assert typing.get_origin(hint) in (typing.Union, types.UnionType)
    args = typing.get_args(hint)
    assert args == (TwoElectronTerm, type(None))
