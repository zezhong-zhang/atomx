"""Decoupled ``numerics`` (radial ENGINE) × ``exchange`` (MODEL) Atom API.

These tests pin:

- ``numerics=None`` resolves to the shooting engine for every model it
  supports (hf/slater/gu); the B-spline-only parametric potentials
  (tfda/gsz) resolve to bspline, and multi-config input yields to the
  B-spline MCDHF machinery;
- the removed legacy 'fac'/'shooting' exchange strings raise a
  ValueError naming the canonical replacement;
- the shooting × hf cell solves (non-local DHF via the
  inhomogeneous-equation method) and matches the B-spline DHF;
- the same MODEL on both ENGINES agrees physically (bspline+gu vs
  shooting+gu);
- ``alpha_x`` reaches the engine.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402

FE17 = dict(Z=26, charge=16, config="1s2 2s2 2p6")


# ----------------------------------------------------------------------
# Removed legacy exchange strings raise with the canonical replacement
# ----------------------------------------------------------------------

@pytest.mark.parametrize("legacy,replacement", [
    ("fac", "numerics='shooting', exchange='gu'"),
    ("shooting", "numerics='shooting', exchange='slater'"),
])
def test_removed_legacy_exchange_raises_with_replacement(legacy, replacement):
    with pytest.raises(ValueError, match="was removed"):
        Atom(**FE17, exchange=legacy)
    try:
        Atom(**FE17, exchange=legacy)
    except ValueError as err:
        assert replacement in str(err)


# ----------------------------------------------------------------------
# numerics=None resolves to the shooting engine for every model it supports
# ----------------------------------------------------------------------

def test_default_numerics_hf_equals_shooting_hf():
    default = Atom(**FE17, exchange="hf").solve()
    explicit = Atom(**FE17, numerics="shooting", exchange="hf").solve()
    assert explicit.total_energy == default.total_energy
    assert explicit.orbital_energies == default.orbital_energies


def test_default_numerics_slater_equals_shooting_slater():
    default = Atom(**FE17, exchange="slater").solve()
    explicit = Atom(**FE17, numerics="shooting", exchange="slater").solve()
    assert explicit.total_energy == default.total_energy
    assert explicit.orbital_energies == default.orbital_energies


def test_gu_defaults_to_shooting_engine():
    """numerics=None + exchange='gu' routes via the shooting engine."""
    default = Atom(**FE17, exchange="gu", breit=False, qed=False).solve()
    explicit = Atom(**FE17, numerics="shooting", exchange="gu",
                    breit=False, qed=False).solve()
    assert explicit.total_energy == default.total_energy
    # Pin the Fe XVII shooting/gu value — this must stay bit-identical
    # to the historical exchange="fac" result (correction-free: the
    # historical value predates the breit/qed=True defaults).
    assert default.total_energy == -1148.4141858935911


# ----------------------------------------------------------------------
# Support matrix validation
# ----------------------------------------------------------------------

def test_shooting_hf_solves():
    """shooting × hf (formerly the ValueError cell) routes to the
    inhomogeneous-equation DHF solver and agrees with the B-spline DHF.

    The tight cross-engine gates (1e-4 total / 5e-4 orbital relative,
    also for He and Ne) live in tests/test_shooting_hf.py."""
    shooting = Atom(**FE17, numerics="shooting", exchange="hf").solve()
    bspline = Atom(**FE17, numerics="bspline", exchange="hf").solve()
    assert shooting.converged
    assert shooting.provenance == {"numerics": "shooting", "exchange": "hf"}
    rel = abs(shooting.total_energy - bspline.total_energy) / abs(
        bspline.total_energy)
    assert rel < 1e-4


def test_bad_numerics_raises():
    with pytest.raises(ValueError, match="numerics"):
        Atom(**FE17, numerics="finite-difference")


def test_multi_config_yields_to_bspline_but_explicit_shooting_raises():
    """configs=[...] needs the B-spline MCDHF machinery: the defaulted
    shooting engine silently yields to bspline, while an EXPLICIT
    numerics='shooting' is a contradiction and raises."""
    a = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
    assert a.numerics == "bspline"
    with pytest.raises(ValueError, match="multi-configuration"):
        Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"], numerics="shooting")


def test_bad_exchange_raises():
    with pytest.raises(ValueError, match="exchange"):
        Atom(**FE17, exchange="xalpha")


def test_alpha_x_rejected_for_non_slater():
    with pytest.raises(ValueError, match="alpha_x"):
        Atom(**FE17, exchange="hf", alpha_x=0.7)


# ----------------------------------------------------------------------
# Same model, different engine: physics agrees
# ----------------------------------------------------------------------

def test_bspline_gu_matches_shooting_gu():
    """Gu's α_xc(Z, N) model on both engines: same physics, different
    numerics — Fe XVII total energies agree to < 0.5%."""
    # Local-exchange SCF on the B-spline engine needs ~45 iterations
    # for Fe XVII (just past the default 40 cap).
    bspline = Atom(**FE17, numerics="bspline", exchange="gu").solve(
        max_iter=200)
    shooting = Atom(**FE17, numerics="shooting", exchange="gu").solve()
    assert bspline.converged
    rel = abs(bspline.total_energy - shooting.total_energy) / abs(
        shooting.total_energy)
    assert rel < 5e-3


# ----------------------------------------------------------------------
# alpha_x propagation
# ----------------------------------------------------------------------

def test_alpha_x_propagates_to_shooting():
    default = Atom(**FE17, numerics="shooting", exchange="slater").solve()
    a15 = Atom(**FE17, numerics="shooting", exchange="slater",
               alpha_x=1.5).solve()
    a10 = Atom(**FE17, numerics="shooting", exchange="slater",
               alpha_x=1.0).solve()
    # Engine default is α = 3/2 — passing it explicitly is a no-op…
    assert a15.total_energy == default.total_energy
    # …while a different coefficient changes the exchange physics.
    assert a10.total_energy != a15.total_energy


def test_alpha_x_propagates_to_bspline():
    default = Atom(**FE17, numerics="bspline", exchange="slater").solve()
    explicit = Atom(**FE17, numerics="bspline", exchange="slater",
                    alpha_x=2.0 / 3.0).solve()
    other = Atom(**FE17, numerics="bspline", exchange="slater",
                 alpha_x=1.0).solve()
    assert explicit.total_energy == default.total_energy
    assert other.total_energy != explicit.total_energy


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected", [
    (dict(exchange="hf"), ("shooting", "hf")),
    (dict(exchange="slater"), ("shooting", "slater")),
    (dict(numerics="bspline", exchange="hf"), ("bspline", "hf")),
    (dict(numerics="shooting", exchange="gu"), ("shooting", "gu")),
    (dict(numerics="shooting", exchange="slater"), ("shooting", "slater")),
    (dict(numerics="bspline", exchange="gu"), ("bspline", "gu")),
])
def test_provenance_records_engine_and_model(kwargs, expected):
    a = Atom(**FE17, **kwargs).solve()
    engine, model = expected
    assert a.provenance == {"numerics": engine, "exchange": model}
    # provenance intentionally NOT mirrored into energy_components:
    # numeric consumers (compare, viz) iterate that dict as floats
    assert "numerics" not in (a.energy_components or {})
