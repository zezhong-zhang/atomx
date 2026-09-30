"""Cross-engine uniformity gates for the two SCF ORBITAL SETS.

WHAT THIS FILE GATES (audit S4 finding 21 — C-degenerate for the
machinery claim).  ``src/atomx/properties/transitions/multipole.py:35``
states "One implementation of each radial kernel serves both engines":
``radial_eL``/``radial_mL`` are called once, on orbitals that the
workspace seam has already evaluated from either flavour.  The <= 2e-3
rate/line-strength agreement below therefore measures the two *SCF
solvers* (Galerkin B-splines vs Salvat power-series ODE shooting) — how
well they produce the same orbitals — and is blind to any error *inside*
``radial_eL``/``radial_mL`` (the exact class of error the pre-fix M1/E2
prefactor bug, commit 621fe99, was).  The kernels themselves are gated
by ``test_radial_eL_matches_hydrogenic_closed_forms`` at the end of this
file, which uses only closed-form hydrogenic moments.

The transition pipeline (``compute_transitions`` → ``radial_eL`` /
``radial_eL_velocity`` / ``radial_mL`` → Einstein rates, both gauges)
historically consumed B-spline coefficient vectors + ``bsp_data`` only.
It now runs on ShootingResult grid orbitals through the
:mod:`atomx.numerics.radial_workspace` seam — the SAME radial-integral
implementations serve both flavours (see
``atomx/properties/transitions/multipole.py`` and
``TransitionCalculator._single_config_payload``).

Cross-engine gates (independent radial numerics — Galerkin B-splines vs
Salvat power-series ODE shooting — same non-local HF exchange, same
Fermi nuclear model, ``breit=False, qed=False`` for determinism):

1. every E1 rate agrees to <= 2e-3 relative (Ne, Fe XVII ground);
2. every E1/M1/E2 LINE STRENGTH agrees to <= 2e-3 relative, except the
   relativistically-forbidden same-(l, j) M1 residuals (see below);
3. gauge diagnostic |A_len/A_vel - 1| on shooting within 2x of bspline;
4. CRM.from_atom on a DEFAULT (shooting) atom builds and its
   steady-state populations match the bspline-built CRM to 1e-2.

Two documented channel families structurally CANNOT meet a raw-rate
2e-3 gate on independent engines — they are gated separately with the
measured numbers spelled out (NOT silently loosened):

* Fine-structure lines (2p1/2 -> 2p3/2 M1/E2): the line strengths agree
  to <= 3e-5, but omega is the tiny FS splitting (4.5e-3 Ha in Ne),
  itself a near-cancelling difference of two orbital energies that only
  agree to ~2e-5 relative across engines.  A ∝ omega^(2L+1)·S amplifies
  the measured relOmega (5.1e-3 Ne / 1.9e-3 Fe XVII) into raw-rate
  deviations of 1.5e-2/2.6e-2 (Ne M1/E2) and 5.8e-3/9.7e-3 (Fe XVII).
  Gate: S <= 2e-3 (machinery) + relOmega <= 1e-2 (SCF-level bound).

* Same-(l, j) M1 channels (1s1/2 -> 2s1/2): the long-wavelength M1
  radial integral between same-kappa orbitals is a relativistically-
  forbidden residual — the nonrelativistic M1 operator ∝ J has no
  off-diagonal radial matrix elements, so the integral is a near-total
  cancellation of order (alpha*Z)^2 hypersensitive to ~4e-5 orbital
  differences (measured S deviation: 4.3e-2 Ne, 1.7e-2 Fe XVII, on
  rates 8+ orders below the E1s).  Gate: S ratio dev <= 0.10 +
  absolute-scale suppression vs the strongest E1.

The open-shell excited pair (Ne 1s2 2s2 2p5 3s1) uses statistical
jj-average coefficients on both engines. Shooting uses sequential Schmidt
orthogonalization, an explicitly approximate treatment of the open-shell
Lagrange multipliers (numerics/shooting/hartree_fock.py); B-splines now optimize occupied
rotations under the metric constraints. All open-shell rates therefore
use the existing 3e-2 model-comparison band. The previous special 2e-3
claim for the 2p3/2 -> 3s1/2 rate is no longer justified: the measured
difference is 4.24e-3 after correcting the B-spline constraints. Closed-shell
and analytic-kernel gates above retain their original tolerances.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom, Correlation  # noqa: E402

#: Hard cross-engine gate on rates / line strengths (machinery observable).
RATE_RTOL = 2e-3
#: Gauge-quality gate: shooting deviation within 2x of bspline deviation
#: (tiny absolute floor so a ~0 bspline deviation can't produce flakiness).
GAUGE_FACTOR = 2.0
GAUGE_FLOOR = 1e-4
#: SCF-level bound on the fine-structure splitting agreement (documented).
FS_OMEGA_RTOL = 1e-2
#: Documented gate for the relativistically-forbidden same-(l,j) M1
#: residuals (measured 4.3e-2 Ne / 1.7e-2 Fe XVII — see module docstring).
FORBIDDEN_M1_RTOL = 0.10
#: Documented gate for open-shell channels touching the pre-existing
#: 2p1/2 eigenvalue gap (measured <= 2.3e-2 — see module docstring).
OPEN_SHELL_RTOL = 3e-2

MULTIPOLES = ["E1", "M1", "E2"]

CASES = {
    "Ne": dict(Z=10, config="1s2 2s2 2p6"),
    "FeXVII": dict(Z=26, config="1s2 2s2 2p6", charge=16),
}

#: Open-shell excited pair providing a physical 2p -> 3s E1 line.
#: ``averaging='statistical'`` on both engines: the bspline default
#: (``averaging='ci'``) would promote the open-shell config to a jj-CSF
#: MCDHF solve, which is a different model than shooting's statistical
#: jj-average — not a like-for-like engine comparison.
EXCITED_CASES = {
    "Ne_2p5_3s": dict(Z=10, config="1s2 2s2 2p5 3s1"),
}


def _key(tr):
    return (tr.initial_label, tr.final_label, tr.multipole)


def _label_suffix(label):
    """'2p3/2' -> 'p3/2' (the (l, j) part of a subshell label)."""
    return label.lstrip("0123456789")


def _is_forbidden_m1(tr):
    """Same-(l, j) M1: relativistically-forbidden long-wavelength residual."""
    return (tr.multipole == "M1"
            and _label_suffix(tr.initial_label) == _label_suffix(tr.final_label))


def _solve_both(kw, averaging=None):
    """Solve on both engines with the same exchange/CI protocol.

    ``averaging`` is a Correlation field: the multi-configuration protocol
    is carried by the same object as the expansion, so it is passed as
    ``correlation=Correlation(averaging=...)``.
    """
    ci = None if averaging is None else Correlation(averaging=averaging)
    shooting = Atom(
                    engine="shooting",
                    exchange="hf",
                    corrections="none",
                    correlation=ci,
                    **kw,
               ).solve()
    bspline = Atom(
                   engine="bspline",
                   exchange="hf",
                   corrections="none",
                   correlation=ci,
                   **kw,
              ).solve()
    assert shooting.converged and bspline.converged
    return shooting, bspline


def _transition_maps(shooting, bspline):
    ts = shooting.transitions(multipoles=MULTIPOLES, gauge="both")
    tb = bspline.transitions(multipoles=MULTIPOLES, gauge="both")
    ds = {_key(t): t for t in ts}
    db = {_key(t): t for t in tb}
    assert set(ds) == set(db), (
        f"engines disagree on the transition line set: "
        f"shooting-only {set(ds) - set(db)}, bspline-only {set(db) - set(ds)}")
    return ds, db


@pytest.fixture(scope="module")
def solved():
    """Ground cases solved once per engine, transitions matched by line."""
    out = {}
    for name, kw in CASES.items():
        shooting, bspline = _solve_both(kw)
        ds, db = _transition_maps(shooting, bspline)
        out[name] = (shooting, bspline, ds, db)
    return out


@pytest.fixture(scope="module")
def solved_excited():
    out = {}
    for name, kw in EXCITED_CASES.items():
        shooting, bspline = _solve_both(kw, averaging="statistical")
        ds, db = _transition_maps(shooting, bspline)
        out[name] = (shooting, bspline, ds, db)
    return out


# ---------------------------------------------------------------------------
# Gate 1: E1 rates, both gauges' primary value (length), <= 2e-3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_e1_rates_cross_engine(solved, name):
    _, _, ds, db = solved[name]
    e1_keys = [k for k in ds if k[2] == "E1"]
    assert e1_keys, f"{name}: no E1 lines found"
    for k in e1_keys:
        s, b = ds[k].A_rate_per_s, db[k].A_rate_per_s
        rel = abs(s / b - 1.0)
        assert rel <= RATE_RTOL, (
            f"{name} {k}: E1 rate shooting {s!r} vs bspline {b!r} "
            f"(rel {rel:.2e} > {RATE_RTOL})")


@pytest.mark.parametrize("name", list(CASES))
def test_e1_velocity_rates_cross_engine(solved, name):
    """The velocity-gauge machinery must be engine-uniform too."""
    _, _, ds, db = solved[name]
    for k in (k for k in ds if k[2] == "E1"):
        s, b = ds[k].A_velocity, db[k].A_velocity
        rel = abs(s / b - 1.0)
        assert rel <= RATE_RTOL, (
            f"{name} {k}: velocity-gauge E1 rate rel dev {rel:.2e} > {RATE_RTOL}")


# ---------------------------------------------------------------------------
# Gate 2: M1/E2 line strengths (the radial-machinery observable), <= 2e-3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_line_strengths_cross_engine(solved, name):
    """S = (angular x radial)^2 is omega-free: it isolates the radial
    machinery from cross-engine SCF-eigenvalue differences.  Gated for
    every E1/M1/E2 line except the same-(l,j) M1 residuals (below)."""
    _, _, ds, db = solved[name]
    checked = 0
    for k, t in ds.items():
        if _is_forbidden_m1(t):
            continue
        rel = abs(t.line_strength / db[k].line_strength - 1.0)
        assert rel <= RATE_RTOL, (
            f"{name} {k}: line strength rel dev {rel:.2e} > {RATE_RTOL}")
        checked += 1
    assert checked >= 6  # E1 x4 + FS M1 + FS E2 at minimum


@pytest.mark.parametrize("name", list(CASES))
def test_m1_e2_rates_at_matched_omega(solved, name):
    """A ∝ omega^(2L+1)·S exactly, so the omega-matched rate
    A_shooting vs A_bspline·(omega_s/omega_b)^(2L+1) gates the full
    rate pipeline (angular + radial + rate formula) at 2e-3 without
    conflating it with the engines' FS-splitting agreement."""
    _, _, ds, db = solved[name]
    for k, t in ds.items():
        if k[2] == "E1" or _is_forbidden_m1(t):
            continue
        L = int(k[2][1:])
        b = db[k]
        scale = (abs(t.energy_diff_hartree) / abs(b.energy_diff_hartree)) ** (2 * L + 1)
        rel = abs(t.A_rate_per_s / (b.A_rate_per_s * scale) - 1.0)
        assert rel <= RATE_RTOL, (
            f"{name} {k}: omega-matched rate rel dev {rel:.2e} > {RATE_RTOL}")


@pytest.mark.parametrize("name", list(CASES))
def test_fs_splitting_bounds_raw_m1_e2_rates(solved, name):
    """Documented raw-rate residual on fine-structure lines.

    Raw FS M1/E2 rates deviate by (2L+1)·relOmega — measured
    1.5e-2 (M1) / 2.6e-2 (E2) for Ne and 5.8e-3 / 9.7e-3 for Fe XVII —
    entirely from the FS-splitting omega, whose cross-engine agreement
    (5.1e-3 / 1.9e-3) is an SCF eigenvalue property, not transitions
    machinery (the line strengths agree to <= 3e-5).  Assert the
    omega agreement stays within the documented SCF-level bound so a
    machinery regression can't hide behind this residual."""
    _, _, ds, db = solved[name]
    fs_keys = [k for k, t in ds.items()
               if k[2] != "E1" and not _is_forbidden_m1(t)]
    assert fs_keys, f"{name}: expected FS M1/E2 lines"
    for k in fs_keys:
        om_s = abs(ds[k].energy_diff_hartree)
        om_b = abs(db[k].energy_diff_hartree)
        rel_om = abs(om_s / om_b - 1.0)
        assert rel_om <= FS_OMEGA_RTOL, (
            f"{name} {k}: FS-splitting cross-engine dev {rel_om:.2e} "
            f"> {FS_OMEGA_RTOL} — SCF eigenvalue agreement regressed")


# ---------------------------------------------------------------------------
# Documented exception: relativistically-forbidden same-(l,j) M1 residuals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_forbidden_m1_residual_documented(solved, name):
    """1s1/2 -> 2s1/2 M1 CANNOT meet the 2e-3 gate: the long-wavelength
    integral is a near-total relativistic cancellation (see module
    docstring; measured S deviation 4.3e-2 Ne / 1.7e-2 Fe XVII).  Gate
    at the documented 0.10 and pin the absolute suppression so the
    channel stays physically irrelevant for rate totals."""
    _, _, ds, db = solved[name]
    forb = {k: t for k, t in ds.items() if _is_forbidden_m1(t)}
    assert forb, f"{name}: expected a 1s->2s M1 residual line"
    strongest_e1 = max(t.A_rate_per_s for k, t in ds.items() if k[2] == "E1")
    for k, t in forb.items():
        rel = abs(t.line_strength / db[k].line_strength - 1.0)
        assert rel <= FORBIDDEN_M1_RTOL, (
            f"{name} {k}: forbidden-M1 residual S dev {rel:.2e} "
            f"> {FORBIDDEN_M1_RTOL}")
        assert t.A_rate_per_s < 1e-5 * strongest_e1, (
            f"{name} {k}: 'forbidden' M1 rate {t.A_rate_per_s:.3e} is not "
            f"suppressed vs strongest E1 {strongest_e1:.3e} — the "
            f"long-wavelength residual grew; investigate")


# ---------------------------------------------------------------------------
# Gate 3: gauge invariance quality on shooting within 2x of bspline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_gauge_ratio_within_2x_of_bspline(solved, name):
    _, _, ds, db = solved[name]
    checked = 0
    for k, t in ds.items():
        if k[2] != "E1":
            continue
        dev_s = abs(t.A_velocity / t.A_length - 1.0)
        dev_b = abs(db[k].A_velocity / db[k].A_length - 1.0)
        assert dev_s <= GAUGE_FACTOR * dev_b + GAUGE_FLOOR, (
            f"{name} {k}: gauge deviation on shooting {dev_s:.3e} exceeds "
            f"{GAUGE_FACTOR}x bspline {dev_b:.3e}")
        checked += 1
    assert checked >= 4


def test_gauge_both_contract_on_shooting(solved):
    """gauge='both' populates both fields and keeps A_rate length-gauge."""
    _, _, ds, _ = solved["Ne"]
    for t in ds.values():
        assert t.A_length is not None
        assert t.A_velocity is not None
        assert t.A_rate_per_s == t.A_length
        assert t.gauge == "length"


# ---------------------------------------------------------------------------
# Excited pair: a physical 2p -> 3s E1 line on both engines
# ---------------------------------------------------------------------------


def test_excited_2p_3s_e1_cross_engine(solved_excited):
    """Compare open-shell models within their documented approximation band."""
    _, _, ds, db = solved_excited["Ne_2p5_3s"]

    k = ("2p3/2", "3s1/2", "E1")
    assert k in ds, f"expected {k} in the excited-config line set"
    for key in (k2 for k2 in ds if k2[2] == "E1"):
        rel = abs(ds[key].A_rate_per_s / db[key].A_rate_per_s - 1.0)
        assert rel <= OPEN_SHELL_RTOL, (
            f"Ne excited {key}: E1 rel dev {rel:.2e} > {OPEN_SHELL_RTOL} "
            f"(documented open-shell SCF gap bound)")


# ---------------------------------------------------------------------------
# Gate 4: CRM.from_atom on the DEFAULT (shooting) engine
# ---------------------------------------------------------------------------


def test_crm_from_default_atom_matches_bspline(solved):
    """CRM built from a default-engine atom (shooting since the 2026-07
    flip) must produce the same steady-state populations as the
    bspline-built CRM to 1e-2."""
    from atomx.plasma.crm import CollisionalRadiativeModel

    default_atom = Atom(Z=10, corrections="none").solve()
    assert default_atom.provenance.get("engine") == "shooting"

    _, bspline, _, _ = solved["Ne"]
    crm_s = CollisionalRadiativeModel.from_atom(default_atom)
    crm_b = CollisionalRadiativeModel.from_atom(bspline)
    assert crm_s.n_levels == crm_b.n_levels > 0
    assert [lv.label for lv in crm_s.levels] == [lv.label for lv in crm_b.levels]

    for ne_cm3, Te_eV in ((1e12, 50.0), (1e18, 20.0)):
        ps = crm_s.steady_state(ne_cm3=ne_cm3, Te_eV=Te_eV)
        pb = crm_b.steady_state(ne_cm3=ne_cm3, Te_eV=Te_eV)
        assert abs(ps.sum() - 1.0) < 1e-9 and abs(pb.sum() - 1.0) < 1e-9
        # Relative on populated levels, absolute floor for empty ones.
        dev = np.max(np.abs(ps - pb) / np.maximum(np.abs(pb), 1e-12))
        assert dev <= 1e-2, (
            f"CRM steady-state pops diverge across engines at "
            f"ne={ne_cm3:g}, Te={Te_eV}: max dev {dev:.3e} > 1e-2")


# ---------------------------------------------------------------------------
# Flavour gating stays honest
# ---------------------------------------------------------------------------


def test_schrodinger_shooting_transitions_raise():
    """Transitions need Dirac spinors: Schrödinger shooting must raise a
    clear NotImplementedError, not silently produce garbage."""
    from types import SimpleNamespace

    from atomx.methods.dirac_fock import Subshell
    from atomx.numerics.shooting.local import ShootingSolver
    from atomx.properties.transitions.calculator import TransitionCalculator

    res = ShootingSolver(
        Z=2, subshells=[Subshell(n=1, kappa=-1, occupation=2)],
        n_grid=400, equation="schrodinger").solve()
    atom = SimpleNamespace(converged=True, Z=2, _result=res,
                           _csf_result=None, _bsp_data=None)
    calc = TransitionCalculator(atom)
    with pytest.raises(NotImplementedError, match="radial workspace"):
        calc.compute_all(multipoles=["E1"])


# ---------------------------------------------------------------------------
# Independent kernel oracle (not routed through the cross-engine comparison)
# ---------------------------------------------------------------------------

def test_radial_eL_matches_hydrogenic_closed_forms():
    """``radial_eL`` against exact hydrogenic moments — no engine involved.

    A-class oracle: the kernel is defined as
    ``∫[P_f P_i + Q_f Q_i] r^L dr``, so for a normalised orbital with
    itself it must return the closed-form hydrogenic moments (Bethe &
    Salpeter; non-relativistic values, which the Dirac orbital approaches
    as O((Z alpha)^2) at Z=1):

        <r>   = (3n² − l(l+1)) / (2Z)                = 1.5   (1s), 6.0  (2s)
        <r²>  = n²(5n² + 1 − 3l(l+1)) / (2Z²)        = 3.0   (1s), 42.0 (2s)

    Measured (ns=80/ks=9): 1.4999734 / 2.9999068 (1s) and 5.9998835 /
    41.998496 (2s), i.e. −1.8e-5 … −3.6e-5 as expected.  Neither engine's
    SCF enters here, so this is the file's only gate on the kernel
    *formula* rather than on orbital agreement.
    """
    from atomx.methods.hydrogenic_reference import (
        hydrogenic_dirac_wavefunction,
    )
    from atomx.properties.transitions.multipole import radial_eL

    for n, r_exp, r2_exp in ((1, 1.5, 3.0), (2, 6.0, 42.0)):
        P, Q, bsp, _E = hydrogenic_dirac_wavefunction(n, -1, 1, ns=80, ks=9)
        r1 = radial_eL(P, Q, P, Q, bsp, 1)
        r2 = radial_eL(P, Q, P, Q, bsp, 2)
        assert pytest.approx(r_exp, rel=1e-3) == r1, f"<r>({n}s) = {r1!r}"
        assert pytest.approx(r2_exp, rel=1e-3) == r2, f"<r^2>({n}s) = {r2!r}"
