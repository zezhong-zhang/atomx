"""Unit tests for the shared two-electron LS term layer (Milestone 0)."""

from __future__ import annotations

import types
import typing

import pytest

from atomx.scattering.two_electron.states import TwoElectronTerm, helike_terms
from atomx.scattering.r_matrix.target_states import (
    TargetState,
    helike_target_states,
)
from atomx.constants import HARTREE_EV


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


@pytest.mark.parametrize(
    "label",
    ["1s^2 ^1S", "1s2s ^3S", "1s2s ^1S", "1s2p ^3P", "1s2p ^1P"],
)
def test_helike_ne_term_energies_vs_fac(label: str):
    """Hydrogenic + Coulomb excitation energies should reproduce FAC to ~10 %."""
    terms = helike_terms(Z=10, n_max=2, n_grid=2000)
    term_map = {t.label: t for t in terms}
    assert label in term_map, f"missing term {label}"
    term = term_map[label]
    atomx_ground_ev = term_map["1s^2 ^1S"].energy_au * HARTREE_EV
    e_rel_ev = term.energy_au * HARTREE_EV - atomx_ground_ev
    fac_rel = _FAC_NE_X_RELATIVE_EV[label]
    assert e_rel_ev == pytest.approx(fac_rel, rel=0.10), (
        f"{label}: atomx E_rel={e_rel_ev:.2f} eV, FAC={fac_rel:.2f} eV"
    )


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
