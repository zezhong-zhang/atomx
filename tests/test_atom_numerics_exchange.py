"""Decoupled ``numerics`` (radial ENGINE) × ``exchange`` (MODEL) Atom API.

The legacy single ``exchange`` string conflated the two axes
('fac'/'shooting' picked the ODE shooting engine, 'hf'/'slater' the
B-spline engine).  These tests pin:

- legacy aliases resolve to the same (engine, model) pair bit-identically;
- the unsupported shooting × hf cell raises;
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
# Back-compat: legacy exchange strings == explicit (numerics, exchange)
# ----------------------------------------------------------------------

def test_legacy_fac_equals_shooting_gu():
    legacy = Atom(**FE17, exchange="fac").solve()
    explicit = Atom(**FE17, numerics="shooting", exchange="gu").solve()
    assert explicit.total_energy == legacy.total_energy
    assert explicit.orbital_energies == legacy.orbital_energies


def test_legacy_shooting_equals_shooting_slater():
    legacy = Atom(**FE17, exchange="shooting").solve()
    explicit = Atom(**FE17, numerics="shooting", exchange="slater").solve()
    assert explicit.total_energy == legacy.total_energy
    assert explicit.orbital_energies == legacy.orbital_energies


def test_legacy_hf_equals_bspline_hf():
    legacy = Atom(**FE17, exchange="hf").solve()
    explicit = Atom(**FE17, numerics="bspline", exchange="hf").solve()
    assert explicit.total_energy == legacy.total_energy
    assert explicit.orbital_energies == legacy.orbital_energies


def test_legacy_slater_equals_bspline_slater():
    legacy = Atom(**FE17, exchange="slater").solve()
    explicit = Atom(**FE17, numerics="bspline", exchange="slater").solve()
    assert explicit.total_energy == legacy.total_energy
    assert explicit.orbital_energies == legacy.orbital_energies


def test_gu_defaults_to_shooting_engine():
    """numerics=None + exchange='gu' matches the legacy 'fac' routing."""
    legacy = Atom(**FE17, exchange="fac").solve()
    gu = Atom(**FE17, exchange="gu").solve()
    assert gu.total_energy == legacy.total_energy


# ----------------------------------------------------------------------
# Support matrix validation
# ----------------------------------------------------------------------

def test_shooting_hf_raises():
    with pytest.raises(ValueError,
                       match="non-local HF exchange requires "
                             "numerics='bspline'"):
        Atom(**FE17, numerics="shooting", exchange="hf")


def test_bad_numerics_raises():
    with pytest.raises(ValueError, match="numerics"):
        Atom(**FE17, numerics="finite-difference")


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
    (dict(exchange="hf"), ("bspline", "hf")),
    (dict(exchange="slater"), ("bspline", "slater")),
    (dict(exchange="fac"), ("shooting", "gu")),
    (dict(exchange="shooting"), ("shooting", "slater")),
    (dict(numerics="bspline", exchange="gu"), ("bspline", "gu")),
])
def test_provenance_records_engine_and_model(kwargs, expected):
    a = Atom(**FE17, **kwargs).solve()
    engine, model = expected
    assert a.provenance == {"numerics": engine, "exchange": model}
    # provenance intentionally NOT mirrored into energy_components:
    # numeric consumers (compare, viz) iterate that dict as floats
    assert "numerics" not in (a.energy_components or {})
