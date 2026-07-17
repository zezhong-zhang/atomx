"""``averaging`` as an orthogonal axis: numerics × exchange × averaging.

Before this change the ``use_ci_avg`` gate in ``Atom.solve`` required
``numerics != "shooting"``, so ``averaging="ci"`` was SILENTLY ignored
on the shooting engine (Co shooting ci vs statistical came out
bit-identical).  These tests pin the decoupling:

- shooting × hf × ci routes through the MCDHF outer loop (Guerra
  jj_configs pair weights in ShootingHFSolver + the grid-flavour CI
  kernel) and actually differs from shooting × hf × statistical on an
  open shell (Co 3d7 4s2);
- the CI-averaged totals agree across engines (the engines share no
  radial numerics — B-spline Galerkin + cell integrals vs ODE shooting
  + SLAG6 — so this validates the whole engine-generic pipeline;
  measured 7.3e-8 relative on Co);
- explicit ``averaging="ci"`` with a local-exchange model raises;
- the DEFAULT ``averaging="ci"`` with slater/gu falls back quietly to
  statistical (erroring would break every slater/gu call);
- provenance records the averaging that actually ran;
- the closed-shell fast path (single jj-CSF) still collapses to the
  plain single-config route, bit-identically.

Note the two engines' "statistical" baselines legitimately differ on
open shells: the shooting single-config route uses the fractional
``q(q-1)`` products (``ShootingHFSolver`` with ``jj_configs=None``)
while the bspline route passes statistical Guerra jj_configs — so the
cross-engine gate is on the CI totals, not the ci-vs-statistical delta.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.methods.mcdhf.solver import MCDHFResult, MCDHFSolver  # noqa: E402
from atomx.numerics.shooting.solver import ShootingResult  # noqa: E402

#: Co 3d7 4s2 — open shell, 4 jj-CSFs, the case the stale gate silently
#: degraded to statistical.  Bare HF (breit/qed off) so the totals pin
#: the SCF + CI numerics, not the correction stack.
CO = dict(Z=27, breit=False, qed=False)

#: Known-good baselines (see MEMORY: atomx known state 2026-07).
CO_BSPLINE_CI = -1392.0165
CO_SHOOTING_STATISTICAL = -1391.9970


@pytest.fixture(scope="module")
def co_solved():
    out = {}
    for engine, avg in (("shooting", "ci"), ("shooting", "statistical"),
                        ("bspline", "ci")):
        out[(engine, avg)] = Atom(numerics=engine, exchange="hf",
                                  averaging=avg, **CO).solve()
    return out


# ----------------------------------------------------------------------
# shooting × hf × ci routes through MCDHF and actually differs
# ----------------------------------------------------------------------

def test_shooting_ci_routes_to_mcdhf(co_solved):
    a = co_solved[("shooting", "ci")]
    assert isinstance(a.result, MCDHFResult)
    assert a.result.converged
    assert a.provenance == {"numerics": "shooting", "exchange": "hf",
                            "averaging": "ci"}


def test_shooting_ci_differs_from_shooting_statistical(co_solved):
    """The old gate made these bit-identical; now the Guerra CI weights
    actually reach the shooting Fock operator (measured delta
    -0.0196 Ha — ground-state weighting binds deeper than the
    fractional statistical average)."""
    e_ci = co_solved[("shooting", "ci")].total_energy
    e_st = co_solved[("shooting", "statistical")].total_energy
    assert e_ci - e_st < -1e-3


def test_shooting_statistical_route_unchanged(co_solved):
    a = co_solved[("shooting", "statistical")]
    assert isinstance(a.result, ShootingResult)
    assert a.provenance == {"numerics": "shooting", "exchange": "hf",
                            "averaging": "statistical"}
    assert a.total_energy == pytest.approx(CO_SHOOTING_STATISTICAL, abs=2e-3)


# ----------------------------------------------------------------------
# Cross-engine agreement of the CI-averaged totals + bspline unchanged
# ----------------------------------------------------------------------

def test_ci_totals_agree_across_engines(co_solved):
    """Engine-generic MCDHF: same Guerra outer loop, disjoint radial
    numerics.  Measured 7.3e-8 relative at the time of writing; the
    1e-6 gate leaves slack for grid/basis retunes while staying far
    below the 1.4e-5 relative ci-vs-statistical physics signal."""
    e_shoot = co_solved[("shooting", "ci")].total_energy
    e_bspl = co_solved[("bspline", "ci")].total_energy
    assert abs(e_shoot - e_bspl) / abs(e_bspl) < 1e-6


def test_bspline_ci_baseline_unchanged(co_solved):
    a = co_solved[("bspline", "ci")]
    assert isinstance(a.result, MCDHFResult)
    assert a.provenance == {"numerics": "bspline", "exchange": "hf",
                            "averaging": "ci"}
    assert a.total_energy == pytest.approx(CO_BSPLINE_CI, abs=2e-3)


def test_shooting_ci_breit_qed_route_through_grid_seam():
    """One-shot Breit/QED on shooting+ci run on the grid-flavour
    corrections workspace — pinned to the B-spline ci values (measured
    cross-engine: E_Breit 2.3e-4, E_QED 9.3e-6 relative)."""
    a = Atom(Z=27, numerics="shooting", exchange="hf",
             averaging="ci").solve()
    comps = a.result.energy_components
    assert comps["E_Breit"] == pytest.approx(1.6978, abs=5e-3)
    assert comps["E_QED_total"] == pytest.approx(1.8419, abs=5e-3)
    assert a.total_energy == pytest.approx(-1388.4768, abs=5e-3)


# ----------------------------------------------------------------------
# ci × local-exchange: explicit raises, default falls back quietly
# ----------------------------------------------------------------------

@pytest.mark.parametrize("exchange", ["gu", "slater"])
def test_explicit_ci_with_local_exchange_raises(exchange):
    with pytest.raises(ValueError, match="averaging='ci'"):
        Atom(Z=27, exchange=exchange, averaging="ci")


@pytest.mark.parametrize("exchange", ["gu", "slater"])
def test_default_ci_with_local_exchange_falls_back_quietly(exchange):
    """averaging="ci" is the DEFAULT — erroring would break every
    slater/gu call, so the unspecified default degrades to statistical
    and provenance records what actually ran."""
    a = Atom(Z=2, exchange=exchange, breit=False, qed=False).solve()
    assert a.converged
    assert a.provenance["averaging"] == "statistical"


# ----------------------------------------------------------------------
# Closed shell: the ci fast path still collapses to plain DHF
# ----------------------------------------------------------------------

def test_closed_shell_shooting_ci_collapses_to_single_config_route():
    """He 1s2 → one jj-CSF: the Guerra ci weights ARE the statistical
    weights, so the ci route reuses the plain shooting solve
    bit-identically and provenance keeps averaging="ci"."""
    ci = Atom(Z=2, numerics="shooting", exchange="hf", averaging="ci",
              breit=False, qed=False).solve()
    st = Atom(Z=2, numerics="shooting", exchange="hf",
              averaging="statistical", breit=False, qed=False).solve()
    assert isinstance(ci.result, ShootingResult)
    assert ci.total_energy == st.total_energy
    assert ci.provenance["averaging"] == "ci"
    assert st.provenance["averaging"] == "statistical"


# ----------------------------------------------------------------------
# MCDHFSolver engine validation
# ----------------------------------------------------------------------

def _co_csfs():
    from atomx.methods.mcdhf.config import enumerate_jj_csfs
    return enumerate_jj_csfs("1s2 2s2 2p6 3s2 3p6 3d7 4s2")


def test_mcdhf_engine_validates():
    with pytest.raises(ValueError, match="engine"):
        MCDHFSolver(Z=27, csfs=_co_csfs(), engine="finite-difference")


def test_mcdhf_shooting_engine_rejects_bspline_only_options():
    csfs = _co_csfs()
    with pytest.raises(ValueError, match="radial_optimization"):
        MCDHFSolver(Z=27, csfs=csfs, engine="shooting",
                    radial_optimization="fac_frozen")
    with pytest.raises(ValueError, match="per_block_orbital_sets"):
        MCDHFSolver(Z=27, csfs=csfs, engine="shooting",
                    per_block_orbital_sets=True)
    with pytest.raises(ValueError, match="nuclear"):
        MCDHFSolver(Z=27, csfs=csfs, engine="shooting", nuclear="uniform")
