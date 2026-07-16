"""Exchange-fn injection + nuclear model factory in ShootingSolver
(review E2/E3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.methods.dirac_fock import Subshell  # noqa: E402
from atomx.numerics.shooting.solver import ShootingSolver  # noqa: E402

SUBS = [Subshell(n=1, kappa=-1, occupation=2),
        Subshell(n=2, kappa=-1, occupation=2),
        Subshell(n=2, kappa=1, occupation=2),
        Subshell(n=2, kappa=-2, occupation=4)]


def test_injected_alpha_reproduces_gu_path():
    a = ShootingSolver(10, SUBS, exchange="gu")
    alpha = a._effective_alpha()
    b = ShootingSolver(10, SUBS,
                       exchange_alpha_fn=lambda Z, N: alpha)
    assert b.solve().total_energy == a.solve().total_energy


def test_injected_alpha_changes_physics():
    a = ShootingSolver(10, SUBS, alpha_x=1.5)
    b = ShootingSolver(10, SUBS,
                       exchange_alpha_fn=lambda Z, N: 1.0)
    assert a.solve().total_energy != b.solve().total_energy


def test_nuclear_string_resolves_mass():
    assert ShootingSolver(26, SUBS, nuclear="fermi").aw > 50.0
    assert ShootingSolver(26, SUBS, nuclear="point").aw == 0.0
    with pytest.raises(ValueError, match="nuclear"):
        ShootingSolver(26, SUBS, nuclear="blob")


def test_legacy_aw_wins():
    s = ShootingSolver(26, SUBS, aw=55.9, nuclear="fermi")
    assert s.aw == 55.9
