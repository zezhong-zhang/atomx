"""Breit + QED corrections on the ODE-shooting engine vs B-spline DHF.

The CorrectionStack adapters historically required a B-spline workspace
and silently skipped every correction for ``ShootingResult`` — so
``Atom(numerics='shooting', exchange='hf', breit=True, qed=True)``
reported NO correction components.  The adapters now carry a
radial-grid branch (``_has_radial_grid_data``): the same physics
(Uehling VP, Mohr/Shabaev self-energy, full Grant–Pyper Breit) is
evaluated directly from the shooting grid orbitals ``(r, difr, P, Q)``
through the :mod:`atomx.corrections.workspace` seam.

Cross-engine gates (independent radial numerics — Galerkin B-splines vs
Salvat power-series ODE shooting — same physics, same Fermi nuclear
model on both sides):

    |shooting_corr / bspline_corr − 1| <= 2e-3   per component,
                                                 He / Ne / Fe XVII.

Measured slack at the time of writing: <= 2.3e-4 on Breit, <= 4e-6 on
SE, <= 2e-5 on VP (all three atoms).

Why the Fermi nucleus on both engines: with a point nucleus the
polynomial B-spline basis cannot represent the Dirac r^γ cusp
(γ = sqrt(1 − (Zα)²) < 1) and its short-range VP folding drifts to
3.8e-3 from the analytic hydrogenic value at Z = 26, while the shooting
power-series origin start reproduces the analytic VP to 1.7e-5.  The
Fermi distribution regularises the origin for both engines; it is also
the Atom default.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.corrections.stack import CorrectionStack  # noqa: E402

#: Hard cross-engine gate per correction component.
CORRECTION_RTOL = 2e-3

CASES = {
    "He": dict(Z=2, config="1s2"),
    "Ne": dict(Z=10, config="1s2 2s2 2p6"),
    "FeXVII": dict(Z=26, config="1s2 2s2 2p6", charge=16),
}

COMPONENTS = ("E_Breit", "E_SE", "E_VP")


@pytest.fixture(scope="module")
def solved():
    """Each case solved once per engine with Breit + QED enabled."""
    out = {}
    for name, kw in CASES.items():
        shooting = Atom(numerics="shooting", exchange="hf",
                        breit=True, qed=True, **kw).solve()
        bspline = Atom(numerics="bspline", exchange="hf",
                       breit=True, qed=True, **kw).solve()
        out[name] = (shooting, bspline)
    return out


# ---------------------------------------------------------------------------
# The corrections must actually run (no silent skip)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_shooting_reports_all_components(solved, name):
    """The historical bug: components silently absent on shooting."""
    shooting, _ = solved[name]
    assert shooting.converged
    comp = shooting.energy_components
    for key in COMPONENTS + ("E_QED_total", "se_method"):
        assert key in comp, (
            f"{name}: {key} missing from shooting energy_components "
            f"(corrections were silently skipped?)")
    assert comp["E_QED_total"] == pytest.approx(
        comp["E_SE"] + comp["E_VP"], rel=1e-12)


def test_stack_apply_on_shooting_is_skip_free(solved):
    """A default stack applied to a ShootingResult runs every backend."""
    shooting, _ = solved["Ne"]
    stack = CorrectionStack(
        se_backend="mohr", include_vp=True, include_breit=True)
    applied = stack.apply(shooting.result)
    assert {"self_energy[mohr]", "uehling_vp", "breit"} <= set(applied)
    assert "_skipped" not in applied
    for _key, cr in applied.items():
        assert cr.energy_shift == cr.energy_shift  # not NaN


def test_shabaev_local_backend_runs_on_shooting(solved):
    """The stack's default SE backend also takes the grid pathway."""
    shooting, _ = solved["Ne"]
    stack = CorrectionStack(
        se_backend="shabaev_local", include_vp=False, include_breit=False)
    applied = stack.apply(shooting.result)
    key = "self_energy[shabaev_local]"
    assert "_skipped" not in applied
    assert applied[key].energy_shift > 0
    assert applied[key].per_orbital  # per-orbital breakdown populated


def test_corrections_shift_total_energy(solved):
    """total_energy includes the applied shifts (finalize seam ran).

    The invariant is E(breit+qed) == E(bare SCF) + ΔE_Breit + ΔE_SE +
    ΔE_VP, so the baseline must be explicitly correction-free
    (breit/qed default True since the 2026-07 defaults flip)."""
    shooting_plain = Atom(numerics="shooting", exchange="hf",
                          breit=False, qed=False, **CASES["Ne"]).solve()
    shooting, _ = solved["Ne"]
    comp = shooting.energy_components
    shift = comp["E_Breit"] + comp["E_SE"] + comp["E_VP"]
    assert shooting.total_energy == pytest.approx(
        shooting_plain.total_energy + shift, abs=5e-9)


# ---------------------------------------------------------------------------
# Cross-engine validation gates (the point of the exercise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("component", COMPONENTS)
def test_cross_engine_agreement(solved, name, component):
    shooting, bspline = solved[name]
    s = shooting.energy_components[component]
    b = bspline.energy_components[component]
    assert b != 0.0
    rel = abs(s / b - 1.0)
    assert rel <= CORRECTION_RTOL, (
        f"{name} {component}: shooting {s!r} vs bspline {b!r} "
        f"(rel {rel:.2e} > {CORRECTION_RTOL})")


@pytest.mark.parametrize("name", list(CASES))
def test_correction_signs(solved, name):
    """Physics sanity on the shooting side: Breit > 0 (repulsive),
    SE > 0 (Lamb shift reduces binding), Uehling VP < 0 (attractive)."""
    shooting, _ = solved[name]
    comp = shooting.energy_components
    assert comp["E_Breit"] > 0
    assert comp["E_SE"] > 0
    assert comp["E_VP"] < 0


# ---------------------------------------------------------------------------
# Flavour gating stays honest
# ---------------------------------------------------------------------------


def test_schrodinger_shooting_still_skips():
    """Dirac-only corrections must not run on Schrödinger orbitals."""
    from atomx.methods.dirac_fock import Subshell
    from atomx.numerics.shooting.solver import ShootingSolver

    res = ShootingSolver(
        Z=2, subshells=[Subshell(n=1, kappa=-1, occupation=2)],
        n_grid=400, equation="schrodinger").solve()
    stack = CorrectionStack(
        se_backend="mohr", include_vp=True, include_breit=True)
    applied = stack.apply(res)
    assert set(applied) == {"_skipped"}
    for reason in applied["_skipped"].provenance.values():
        assert "Dirac" in reason
