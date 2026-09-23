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

from atomx import Atom, Correlation  # noqa: E402
from atomx.methods.mcdhf.solver import MCDHFResult, MCDHFSolver  # noqa: E402
from atomx.numerics.shooting.solver import ShootingResult  # noqa: E402

#: Co 3d7 4s2 — open shell, 4 jj-CSFs, the case the stale gate silently
#: degraded to statistical.  Bare HF (breit/qed off) so the totals pin
#: the SCF + CI numerics, not the correction stack.
CO = dict(Z=27, corrections="none")

#: Known-good baselines.  The two *CI* totals were re-pinned on 2026-09-21
#: with the exact CI matrix (``docs/CORRELATION_SD_RESEARCH_2026-09-20.md``
#: §8 option A): the old kernel's configuration-average diagonal plus
#: J-corrections put bspline CI at -1392.0165 and shooting CI at -1390.0324
#: ... the bspline one moves to -1392.1100 because Co 3d7 has repeated
#: parentage that the old kernel deliberately zeroed (its R5) and an
#: open-shell diagonal it averaged (its R1).
#:
#: What the cross-engine gate below establishes, precisely: the two engines
#: share no *radial* numerics (B-spline Galerkin/cell-integral vs shooting
#: SLAG6), so agreement to 7.3e-8 relative validates the radial integral
#: implementations and the SCF fixed point.  It does **not** validate the CI
#: matrix: both engines call the same ``matrix.build_ci_matrix`` ->
#: ``exact_block_matrix`` assembly, so a wrong angular kernel or a wrong
#: assembly would move both totals *identically* and this gate would not
#: notice (audit finding F4 — the docstring here used to claim the opposite).
#: The CI matrix itself is gated behind the m-scheme assembly comparison and
#: the analytic Slater-Condon identities; the Breit/QED numbers below are
#: being migrated from the scalar to the Gaunt operator, and the orchestrator
#: owns re-pinning any that move.
CO_BSPLINE_CI = -1392.1100
CO_SHOOTING_STATISTICAL = -1391.9970


@pytest.fixture(scope="module")
def co_solved():
    out = {}
    for engine, avg in (("shooting", "ci"), ("shooting", "statistical"),
                        ("bspline", "ci")):
        out[(engine, avg)] = Atom(
                                  engine=engine,
                                  exchange="hf",
                                  correlation=Correlation(averaging=avg),
                                  **CO,
                             ).solve()
    return out


# ----------------------------------------------------------------------
# shooting × hf × ci routes through MCDHF and actually differs
# ----------------------------------------------------------------------

def test_shooting_ci_routes_to_mcdhf(co_solved):
    a = co_solved[("shooting", "ci")]
    assert isinstance(a.result, MCDHFResult)
    assert a.result.converged
    assert a.provenance == {"engine": "shooting", "exchange": "hf",
                            "averaging": "ci"}


def test_shooting_ci_exposes_grid_waves(co_solved):
    """The grid-flavour MCDHF result must not require B-spline metadata."""
    a = co_solved[("shooting", "ci")]
    waves = a.waves
    assert waves.basis_type == "grid"
    assert waves.grid.size > 0
    assert waves.orbitals
    assert waves.resolve_all("1s") == ["1s1/2"]
    assert waves.orbitals["1s1/2"].P.shape == waves.grid.shape


def test_shooting_ci_differs_from_shooting_statistical(co_solved):
    """The old gate made these bit-identical; now the Guerra CI weights
    actually reach the shooting Fock operator (measured delta
    -0.0196 Ha — ground-state weighting binds deeper than the
    fractional statistical average)."""
    e_ci = co_solved[("shooting", "ci")].total_energy
    e_st = co_solved[("shooting", "statistical")].total_energy
    assert e_ci - e_st < -1e-3


@pytest.mark.snapshot
def test_shooting_statistical_route_unchanged(co_solved):
    a = co_solved[("shooting", "statistical")]
    assert isinstance(a.result, ShootingResult)
    assert a.provenance == {"engine": "shooting", "exchange": "hf",
                            "averaging": "statistical"}
    assert a.total_energy == pytest.approx(CO_SHOOTING_STATISTICAL, abs=2e-3)


# ----------------------------------------------------------------------
# Cross-engine agreement of the CI-averaged totals + bspline unchanged
# ----------------------------------------------------------------------

def test_ci_totals_agree_across_engines(co_solved):
    """Engine-generic MCDHF: same Guerra outer loop, disjoint radial
    numerics.  Measured 7.3e-8 relative at the time of writing; the
    1e-6 gate leaves slack for grid/basis retunes while staying far
    below the 1.4e-5 relative ci-vs-statistical physics signal.

    Scope: this is a *radial-integral* gate.  Both engines build the CI with
    the same ``build_ci_matrix`` assembly, so agreement here says nothing
    about the angular algebra or the matrix elements — a wrong matrix would
    move both totals by the same amount.  See the note on ``CO_BSPLINE_CI``.
    """
    e_shoot = co_solved[("shooting", "ci")].total_energy
    e_bspl = co_solved[("bspline", "ci")].total_energy
    assert abs(e_shoot - e_bspl) / abs(e_bspl) < 1e-6


@pytest.mark.snapshot
def test_bspline_ci_baseline_unchanged(co_solved):
    a = co_solved[("bspline", "ci")]
    assert isinstance(a.result, MCDHFResult)
    assert a.provenance == {"engine": "bspline", "exchange": "hf",
                            "averaging": "ci"}
    assert a.total_energy == pytest.approx(CO_BSPLINE_CI, abs=2e-3)


@pytest.mark.snapshot
def test_shooting_ci_breit_qed_route_through_grid_seam():
    """One-shot Breit/QED on shooting+ci run on the grid-flavour
    corrections workspace — pinned to the B-spline ci values (measured
    cross-engine: E_Breit 2.3e-4, E_QED 1.0e-5 relative).

    Re-pinned 2026-09-19: the QED total moved from 1.8419 to 0.3821 Ha
    when the self-energy table was corrected to FAC's F(Zα) (the old
    values were 2-5x too large at Z=27, and the n^-3 fallback gave the
    3d/4s orbitals the 1s shape).  See
    docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md C2/M1.
    """
    a = Atom(
             Z=27,
             engine="shooting",
             exchange="hf",
             correlation=Correlation(averaging="ci"),
        ).solve()
    comps = a.result.energy_components
    # Re-pinned 2026-09: this route is multi-CSF, so it applies the exact
    # Gaunt operator in its CI blocks, and `E_Breit` is the shift that is
    # in the total (it used to report 1.6974, the scalar that was never
    # applied).  DBSR_HF `an=27 term=AV mbreit=1` on the same 3d7 4s2
    # average of configuration gives Breit = 0.532518297380245, so the
    # measured 0.5698939033522947 sits +7.0 % above it — the documented
    # omitted retardation, the same sign and size as Ne/Ar (+4.2/+6.2 %).
    assert comps["E_Breit"] == pytest.approx(0.532518297380245, rel=0.08)
    assert comps["E_QED_total"] == pytest.approx(0.3821, abs=5e-3)
    # CI total re-pinned with the exact matrix (was -1389.9370); it tracks
    # the B-spline CI total through the same Guerra outer loop, which the
    # cross-engine gate above enforces independently.
        # Re-pinned 2026-09-21 for the Gaunt Breit OPERATOR, now the default.
    # The previous value encoded the scalar
    # ``corrections.breit.compute_breit_energy``, which is ~3.0x too
    # large against DBSR_HF (``dbsr_hf an=Z term=AV mbreit=1``): Ne
    # 1s2 2s2 2p6 gives DBSR_HF 0.016643629692418 Ha, the Gaunt
    # operator 0.017340 (+4.2%, its documented omitted retardation)
    # and the scalar 0.050449 (3.03x); He 1s2 gives the same 3.01x, so
    # the scalar's error is structure-independent.
    assert a.total_energy == pytest.approx(-1391.1579, abs=5e-3)


# ----------------------------------------------------------------------
# ci × local-exchange: explicit raises, default falls back quietly
# ----------------------------------------------------------------------

@pytest.mark.parametrize("exchange", ["gu", "slater"])
def test_explicit_ci_with_local_exchange_raises(exchange):
    with pytest.raises(ValueError, match="averaging='ci'"):
        Atom(Z=27, exchange=exchange, correlation=Correlation(averaging="ci"))


@pytest.mark.parametrize("exchange", ["gu", "slater"])
def test_default_ci_with_local_exchange_falls_back_quietly(exchange):
    """averaging="ci" is the DEFAULT — erroring would break every
    slater/gu call, so the unspecified default degrades to statistical
    and provenance records what actually ran."""
    a = Atom(Z=2, exchange=exchange, corrections="none").solve()
    assert a.converged
    assert a.provenance["averaging"] == "statistical"


# ----------------------------------------------------------------------
# Closed shell: the ci fast path still collapses to plain DHF
# ----------------------------------------------------------------------

def test_closed_shell_shooting_ci_collapses_to_single_config_route():
    """He 1s2 → one jj-CSF: the Guerra ci weights ARE the statistical
    weights, so the ci route reuses the plain shooting solve
    bit-identically and provenance keeps averaging="ci"."""
    ci = Atom(
              Z=2,
              engine="shooting",
              exchange="hf",
              correlation=Correlation(averaging="ci"),
              corrections="none",
         ).solve()
    st = Atom(
              Z=2,
              engine="shooting",
              exchange="hf",
              correlation=Correlation(averaging="statistical"),
              corrections="none",
         ).solve()
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
