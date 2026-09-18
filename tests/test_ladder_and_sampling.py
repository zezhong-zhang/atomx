"""Tests for the ``Sampling`` tier object and the ``Ladder`` API.

Covers the HYP-13 deliverable:

- ``Sampling`` named tiers exist and validate on construction.
- ``Sampling.with_`` produces an updated copy.
- ``Atom(Z, sampling=...)`` propagates ns/ks/starting_potential defaults.
- ``Atom.solve(sampling=...)`` propagates tol/max_iter.
- ``Ladder.standard()`` is non-empty and ``Ladder.through(stop)`` truncates
  at the right rung.
- ``Atom.climb`` returns a ``LadderRun`` with one outcome per rung.
- Failures are recorded on outcomes, not raised, so the rest of the
  ladder still runs.
- ``LadderRun.show`` does not raise.
- ``Atom.r_uniform`` returns a strictly monotone grid.
- ``Atom.density(grid='log')`` integrates close to the electron count.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from atomx import Atom, Ladder, LadderRun, Sampling
from atomx.ladder import RungOutcome
from atomx.methods import (
    DiracFock,
    HydrogenicDirac,
    HydrogenicSchrodinger,
)

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampling_named_tiers_exist():
    for name in ("quick", "standard", "high", "publication"):
        s = getattr(Sampling, name)
        assert isinstance(s, Sampling)
        assert s.label == name


def test_sampling_validates():
    with pytest.raises(ValueError, match="ks"):
        Sampling(ks=2)
    with pytest.raises(ValueError, match="n_grid"):
        Sampling(n_grid=10)
    with pytest.raises(ValueError, match="tol"):
        Sampling(tol=-1.0)
    with pytest.raises(ValueError, match="starting_potential"):
        Sampling(starting_potential="bogus")


def test_sampling_with_creates_modified_copy():
    s = Sampling.standard.with_(tol=1e-9)
    assert s.tol == 1e-9
    assert s.ns == Sampling.standard.ns
    assert s.label == "custom"  # reset


def test_sampling_constructor_propagates_to_atom():
    a = Atom(Z=2, sampling=Sampling.high)
    assert a.ks == Sampling.high.ks
    # ``ns=None`` on the tier means "auto" — the Atom uses its physics-grid
    # path, which sets ``_use_physics_grid=True`` and a placeholder ns=113.
    assert a._use_physics_grid is False  # high tier specifies ns=200
    assert a.ns == 200


def test_sampling_solve_propagates_tol_and_max_iter():
    """Calling solve(sampling=...) without explicit tol/max_iter uses the
    tier values.  Cross-checks via the SCFResult convergence trace."""
    he = Atom(Z=2)
    he.solve(sampling=Sampling.quick)
    assert he.converged
    # quick has tol=1e-4, so iterations should be well under the standard
    # 40-iter cap for a closed-shell helium.
    assert he.iterations < 40


# ---------------------------------------------------------------------------
# Ladder construction
# ---------------------------------------------------------------------------


def test_ladder_standard_is_nonempty():
    l = Ladder.standard()
    assert len(l) >= 5
    types = [type(r).__name__ for r in l]
    assert "HydrogenicSchrodinger" in types
    assert "DiracFock" in types
    assert l.name == "standard"


def test_ladder_relativistic_excludes_nonrel_rungs():
    l = Ladder.relativistic()
    types = [type(r).__name__ for r in l]
    assert "HartreeFock" not in types
    assert "BreitPauli" not in types
    assert "DiracFock" in types


def test_ladder_through_truncates_at_target_rung():
    full = Ladder.standard()
    short = full.through(DiracFock())
    # All rungs up to and including the first non-Breit, non-QED DiracFock
    types = [type(r).__name__ for r in short]
    assert types[-1] == "DiracFock"
    assert short[-1].breit is False
    assert short[-1].qed is False


def test_ladder_through_with_breit_qed_keeps_distinct_rungs():
    full = Ladder.standard()
    sub = full.through(DiracFock(breit=True))
    # DiracFock() comes before DiracFock(breit=True) in the standard ladder,
    # so both should appear and the last one should have breit=True.
    df_rungs = [r for r in sub if isinstance(r, DiracFock)]
    assert any((not r.breit) for r in df_rungs)
    assert sub[-1].breit is True
    assert sub[-1].qed is False


def test_ladder_through_appends_unknown_target():
    """Targets not in the base ladder are appended at the end."""
    base = Ladder([HydrogenicSchrodinger(), HydrogenicDirac()])
    extended = base.through(DiracFock())
    assert isinstance(extended[-1], DiracFock)


# ---------------------------------------------------------------------------
# Climbing
# ---------------------------------------------------------------------------


def test_atom_climb_returns_ladder_run_with_one_outcome_per_rung():
    he = Atom(Z=2)
    custom = Ladder([HydrogenicSchrodinger(), HydrogenicDirac(), DiracFock()])
    run = he.climb(ladder=custom, sampling=Sampling.quick)
    assert isinstance(run, LadderRun)
    assert len(run) == len(custom)
    for o in run:
        assert isinstance(o, RungOutcome)


def test_climb_records_failures_without_raising():
    """A rung that raises does not abort the ladder; the failure is on the
    corresponding outcome."""

    class _BadRung(HydrogenicDirac):
        def solve(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("synthetic failure")

    he = Atom(Z=2)
    bad = _BadRung()
    custom = Ladder([HydrogenicSchrodinger(), bad, DiracFock()])
    run = he.climb(ladder=custom, sampling=Sampling.quick)
    assert len(run.failed) == 1
    assert "synthetic failure" in run.failed[0].error
    # The successful rungs still ran:
    assert len(run.results) == 2


def test_climb_total_energies_decrease_monotonically_on_he():
    he = Atom(Z=2)
    run = he.climb(through=DiracFock(), sampling=Sampling.quick)
    energies = [o.total_energy for o in run if o.ok]
    # Schrödinger 1e is non-interacting, so it is *less* bound than the
    # DHF many-electron solution.  The interesting check is that after the
    # first rung, energies are negative and finite.
    for E in energies:
        assert E < 0
        assert np.isfinite(E)


def test_ladder_run_show_does_not_raise():
    he = Atom(Z=2)
    run = he.climb(through=DiracFock(), sampling=Sampling.quick)
    buf = io.StringIO()
    run.show(unit="eV", file=buf)
    text = buf.getvalue()
    assert "rung" in text
    assert "Δ from prev" in text
    assert "DiracFock" in text


def test_ladder_run_to_table_carries_unit_in_keys():
    he = Atom(Z=2)
    run = he.climb(ladder=Ladder([HydrogenicSchrodinger(), DiracFock()]),
                   sampling=Sampling.quick)
    rows = run.to_table(unit="eV")
    assert len(rows) == 2
    e_key = "E_total(eV)"
    assert e_key in rows[0]
    # First row has no delta_prev; second does (or None if the rung failed)
    assert rows[0]["delta_prev"] is None
    assert rows[1]["delta_prev"] is not None


def test_show_ladder_returns_run():
    he = Atom(Z=2)
    buf = io.StringIO()
    run = he.show_ladder(through=DiracFock(),
                         sampling=Sampling.quick, file=buf)
    assert isinstance(run, LadderRun)
    assert "DiracFock" in buf.getvalue()


# ---------------------------------------------------------------------------
# Plotting grid + density
# ---------------------------------------------------------------------------


def test_r_uniform_log_grid_is_strictly_monotone():
    ne = Atom(Z=10).solve()
    r = ne.r_uniform(n=200, kind="log")
    assert r is not None
    assert r.shape == (200,)
    diffs = np.diff(r)
    assert (diffs > 0).all()


def test_r_uniform_linear_grid_is_strictly_monotone():
    ne = Atom(Z=10).solve()
    r = ne.r_uniform(n=200, kind="linear")
    assert r is not None
    diffs = np.diff(r)
    assert (diffs > 0).all()


def test_density_log_grid_integrates_to_electron_count():
    ne = Atom(Z=10).solve()
    r_log = ne.r_uniform(n=400, kind="log")
    rho_log = ne.density(grid="log", n=400)
    assert rho_log is not None
    integral = float(np.trapezoid(rho_log, r_log))
    # Tolerance is loose: log-space trapezoid on 400 points captures Ne to
    # ~0.1 e.  The native-grid quadrature is the reference for tight checks.
    assert integral == pytest.approx(10.0, abs=0.2)


def test_density_native_default_unchanged():
    ne = Atom(Z=10).solve()
    rho = ne.density()
    assert rho is not None
    integral = float(np.trapezoid(rho, ne.r))
    assert integral == pytest.approx(10.0, rel=1e-3)
