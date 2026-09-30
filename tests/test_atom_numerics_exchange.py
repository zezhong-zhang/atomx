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

from atomx import Atom, Exchange  # noqa: E402

FE17 = dict(Z=26, charge=16, config="1s2 2s2 2p6")


# ----------------------------------------------------------------------
# Removed legacy exchange strings raise with the canonical replacement
# ----------------------------------------------------------------------

@pytest.mark.parametrize("legacy,replacement", [
    ("fac", "engine='shooting', exchange='gu'"),
    ("shooting", "engine='shooting', exchange='slater'"),
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
    explicit = Atom(**FE17, engine="shooting", exchange="hf").solve()
    assert explicit.total_energy == default.total_energy
    assert explicit.orbital_energies == default.orbital_energies


def test_default_numerics_slater_equals_shooting_slater():
    default = Atom(**FE17, exchange="slater").solve()
    explicit = Atom(**FE17, engine="shooting", exchange="slater").solve()
    assert explicit.total_energy == default.total_energy
    assert explicit.orbital_energies == default.orbital_energies


@pytest.mark.snapshot
def test_gu_defaults_to_shooting_engine():
    """numerics=None + exchange='gu' routes via the shooting engine."""
    default = Atom(**FE17, exchange="gu", corrections="none").solve()
    explicit = Atom(
                    **FE17,
                    engine="shooting",
                    exchange="gu",
                    corrections="none",
               ).solve()
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
    shooting = Atom(**FE17, engine="shooting", exchange="hf").solve()
    bspline = Atom(**FE17, engine="bspline", exchange="hf").solve()
    assert shooting.converged
    assert shooting.provenance == {"engine": "shooting", "exchange": "hf",
                                   "averaging": "hamiltonian"}
    rel = abs(shooting.total_energy - bspline.total_energy) / abs(
        bspline.total_energy)
    assert rel < 1e-4


def test_bad_engine_raises():
    with pytest.raises(ValueError, match="engine must be"):
        Atom(**FE17, engine="finite-difference")


def test_multi_config_defaults_to_bspline_and_accepts_explicit_full_shooting():
    a = Atom(Z=4, config=["1s2 2s2", "1s2 2p2"])
    assert a.engine == "bspline"
    a = Atom(Z=4, config=["1s2 2s2", "1s2 2p2"], engine="shooting")
    assert a.engine == "shooting"


def test_bad_exchange_raises():
    with pytest.raises(ValueError, match="exchange"):
        Atom(**FE17, exchange="xalpha")


def test_alpha_x_rejected_for_non_slater():
    with pytest.raises(ValueError, match="alpha_x"):
        Atom(**FE17, exchange=Exchange(model="hf", alpha_x=0.7))


# ----------------------------------------------------------------------
# Same model, different engine: physics agrees
# ----------------------------------------------------------------------

def test_bspline_gu_matches_shooting_gu():
    """Gu's α_xc(Z, N) model on both engines: same physics, different
    numerics — Fe XVII total energies must agree to < 1e-5 relative.

    Shared kernel: the α_xc table (``atomx.physics.local_exchange``) and the
    jj-average angular coefficients.  NOT shared: the radial solver — the
    B-spline engine builds Galerkin matrices, the shooting engine integrates
    the Salvat-style ODEs.  Measured 2026-09-22: 1.204e-4 Ha
    (rel 1.050e-7); the previous gate, rel < 5e-3, was 5.73 Ha = 156 eV and
    47600x looser.  A sibling gate (``test_atom_surface_equivalence.py``)
    pins a *non-converged* B-spline gu energy, so the engines are known not
    to solve identical equations to machine precision; 1e-5 is ~95x the
    measured agreement.
    """
    # Local-exchange SCF on the B-spline engine needs ~45 iterations
    # for Fe XVII (just past the default 40 cap).
    bspline = Atom(**FE17, engine="bspline", exchange="gu").solve(
        max_iter=200)
    shooting = Atom(**FE17, engine="shooting", exchange="gu").solve()
    assert bspline.converged
    rel = abs(bspline.total_energy - shooting.total_energy) / abs(
        shooting.total_energy)
    assert rel < 1e-5, (
        f"bspline={bspline.total_energy:.9f} shooting="
        f"{shooting.total_energy:.9f} rel={rel:.3e} (measured 1.050e-7)"
    )


# ----------------------------------------------------------------------
# alpha_x propagation
# ----------------------------------------------------------------------

def test_alpha_x_propagates_to_shooting():
    default = Atom(**FE17, engine="shooting", exchange="slater").solve()
    a15 = Atom(
               **FE17,
               engine="shooting",
               exchange=Exchange(model="slater", alpha_x=1.5),
          ).solve()
    a10 = Atom(
               **FE17,
               engine="shooting",
               exchange=Exchange(model="slater", alpha_x=1.0),
          ).solve()
    # Engine default is α = 3/2 — passing it explicitly is a no-op…
    assert a15.total_energy == default.total_energy
    # …while a different coefficient changes the exchange physics.
    assert a10.total_energy != a15.total_energy


def test_alpha_x_propagates_to_bspline():
    default = Atom(**FE17, engine="bspline", exchange="slater").solve()
    explicit = Atom(
                    **FE17,
                    engine="bspline",
                    exchange=Exchange(model="slater", alpha_x=1.5),
               ).solve()
    other = Atom(
                 **FE17,
                 engine="bspline",
                 exchange=Exchange(model="slater", alpha_x=1.0),
            ).solve()
    assert explicit.total_energy == default.total_energy
    assert other.total_energy != explicit.total_energy


def test_slater_default_is_engine_independent():
    shooting = Atom(**FE17, engine="shooting", exchange="slater")
    bspline = Atom(**FE17, engine="bspline", exchange="slater")
    assert shooting.alpha_x == pytest.approx(1.5)
    assert bspline.alpha_x == pytest.approx(1.5)


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected", [
    # The DHF Hamiltonian average and local-potential models remain distinct.
    (dict(exchange="hf"), ("shooting", "hf", "hamiltonian")),
    (dict(exchange="slater"), ("shooting", "slater", "statistical")),
    (dict(engine="bspline", exchange="hf"), ("bspline", "hf", "hamiltonian")),
    (dict(engine="shooting", exchange="gu"),
     ("shooting", "gu", "statistical")),
    (dict(engine="shooting", exchange="slater"),
     ("shooting", "slater", "statistical")),
    (dict(engine="bspline", exchange="gu"),
     ("bspline", "gu", "statistical")),
])
def test_provenance_records_engine_and_model(kwargs, expected):
    a = Atom(**FE17, **kwargs).solve()
    engine, model, averaging = expected
    assert a.provenance == {"engine": engine, "exchange": model,
                            "averaging": averaging}
    # provenance intentionally NOT mirrored into energy_components:
    # numeric consumers (compare, viz) iterate that dict as floats
    assert "engine" not in (a.energy_components or {})
