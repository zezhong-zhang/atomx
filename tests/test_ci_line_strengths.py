"""Exact J-resolved CI line strengths vs clean-room FAC (Fe XVII 3C/3D).

Validates :mod:`atomx.properties.transitions.ci_lines` — the determinant-CI
replacement for the UTA-distributed configuration-average line strengths:

- per-line Einstein A against a live clean-room FAC 1.1.5 TRTable
  (tests/data/fe17_tr_fac_cleanroom.json, provenance inside): Ne-like
  Fe XVII 2p5 3s -> 2p6 and 2p5 3d -> 2p6 (the 3C/3D multiplet);
- the J-resolved RELATIVE line strengths within each multiplet (the
  quantity the UTA path cannot produce) to 5%;
- the exact sum rule: sum of CI line strengths over every level pair ==
  mixing-invariant determinant trace to 1e-10;
- one-electron consistency: for one hole + one electron the multiplet
  total equals the sum of jj one-electron channel strengths
  |<kappa_h||C^1||kappa_e>|^2 R^2 exactly; per-line values differ from
  the pure channels only by the physical 2p1/2/2p3/2 hole-channel CI
  mixing (which FAC shares — pinned by the per-line FAC gates);
- compute_transitions(line_resolution='ci') wiring + UTA fallback.

Measured at introduction (2026-07-17, numerics='shooting'/exchange='gu'
pure-Coulomb basis vs SetSE/VP/MS/Breit-off FAC, OptimizeRadial on the
ground config): per-line A ratios 0.99-1.03, in-multiplet fractions
within 3.2%, 3C/3D ratio 0.962 of FAC, energies +2.4 to +2.9 eV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.constants import ALPHA, AU_RATE_PER_S, HARTREE_EV  # noqa: E402
from atomx.properties.transitions.ci_lines import (  # noqa: E402
    CISizeError,
    ci_line_strengths,
    ground_config_of_pair,
)

FIXTURE = Path(__file__).parent / "data" / "fe17_tr_fac_cleanroom.json"


def _fac_lines():
    """FAC reference: config -> [(E_eV, S_standard, A_per_s)] by energy.

    Reconstruction verified exact during the 2026-07 A_r audit:
    S_standard = rme^2 / 2, A = (4/3) alpha^3 omega^3 S / (2J_u + 1).
    """
    fac = json.loads(FIXTURE.read_text())
    out: dict = {}
    for line in fac["lines"]:
        omega = line["E_upper_eV"] / HARTREE_EV
        S = line["rme"] ** 2 / 2.0
        A = (4.0 / 3.0) * ALPHA ** 3 * omega ** 3 * S \
            / (line["two_J"] + 1) * AU_RATE_PER_S
        out.setdefault(line["config"], []).append(
            (line["E_upper_eV"], S, A))
    for cfg in out:
        out[cfg].sort()
    return out


@pytest.fixture(scope="module")
def fe17_ground():
    """Closed-shell Fe XVII ground in the FAC-equivalent DFS-gu basis."""
    g = Atom(
             Z=26,
             charge=16,
             config="1s2 2s2 2p6",
             engine="shooting",
             exchange="gu",
             corrections="none",
        )
    g.solve()
    assert g.converged
    return g._result


@pytest.fixture(scope="module")
def linesets(fe17_ground):
    return {cfg: ci_line_strengths(fe17_ground, cfg, "2p6", multipole="E1")
            for cfg in ("2p5 3s1", "2p5 3d1")}


# ---------------------------------------------------------------------------
# structure of the line lists
# ---------------------------------------------------------------------------

def test_line_counts_and_j_selection(linesets):
    """Only the J=1 upper levels decay E1 to the J=0 ground."""
    assert len(linesets["2p5 3s1"].lines) == 2
    assert len(linesets["2p5 3d1"].lines) == 3
    for ls in linesets.values():
        for ln in ls.lines:
            assert ln.upper.two_J == 2
            assert ln.lower.two_J == 0
            assert ln.upper.parity == -1 and ln.lower.parity == +1
    # level inventories: 2p5 3s has 4 levels, 2p5 3d has 12 (as FAC)
    assert len(linesets["2p5 3s1"].levels_i) == 4
    assert len(linesets["2p5 3d1"].levels_i) == 12


# ---------------------------------------------------------------------------
# FAC parity gates
# ---------------------------------------------------------------------------

@pytest.mark.external_fac
def test_per_line_A_vs_fac(linesets):
    """Per-line Einstein A within ±8 % of clean-room FAC.

    Measured 2026-09-17: 1.021 / 1.011 (3s) and 1.034 / 1.033 / 0.994 (3d),
    i.e. every line inside 3.4 % — 2.3x margin inside the gate.  The previous
    gate ([0.85, 1.20]) was justified by "the documented radial-basis
    difference (~10-25% in A per the A_r audit)"; that spread belongs to the
    *DR stabilizing widths* (frozen-orbital vs optimized basis, see
    ``test_dr_manifold_e2e``), not to this comparison: with the FAC-equivalent
    shooting/gu basis the per-line A's reproduce FAC to ≤ 3.4 %."""
    fac = _fac_lines()
    for cfg, ls in linesets.items():
        ref = fac[cfg]
        lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
        assert len(lines) == len(ref)
        for (E_fac, _S_fac, A_fac), ln in zip(ref, lines, strict=False):
            ratio = ln.A_per_s / A_fac
            assert 0.92 <= ratio <= 1.08, (
                f"{cfg}: line at {E_fac:.1f} eV A ratio {ratio:.3f} outside "
                "[0.92, 1.08] (gate 8%; measured worst 3.4%)")


@pytest.mark.external_fac
def test_transition_energies_vs_fac(linesets):
    """CI transition energies within 6 eV of FAC (measured +2.4..+2.9)."""
    fac = _fac_lines()
    for cfg, ls in linesets.items():
        lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
        for (E_fac, _S, _A), ln in zip(fac[cfg], lines, strict=False):
            dE = ln.omega_au * HARTREE_EV - E_fac
            assert abs(dE) <= 6.0, f"{cfg}: dE = {dE:.2f} eV at {E_fac:.1f} eV"


@pytest.mark.external_fac
def test_relative_intensities_vs_fac(linesets):
    """In-multiplet S fractions within 5% of FAC — THE J-resolved gate.

    This is the quantity the UTA population factor cannot produce: the
    split of one configuration-pair total across same-J lines.  Measured
    2026-07-17: fraction ratios 1.004/0.994 (3s), 1.031/1.031/0.992 (3d)."""
    fac = _fac_lines()
    for cfg, ls in linesets.items():
        ref = fac[cfg]
        lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
        tot_fac = sum(S for (_E, S, _A) in ref)
        tot_ax = sum(ln.S for ln in lines)
        for (E_fac, S_fac, _A), ln in zip(ref, lines, strict=False):
            frac_fac = S_fac / tot_fac
            frac_ax = ln.S / tot_ax
            assert frac_ax / frac_fac == pytest.approx(1.0, abs=0.05), (
                f"{cfg}: line at {E_fac:.1f} eV fraction "
                f"{frac_ax:.5f} vs FAC {frac_fac:.5f}")


@pytest.mark.external_fac
def test_3c_3d_ratio_vs_fac(linesets):
    """The Fe XVII 3C/3D line-strength ratio matches FAC to 5%.

    FAC clean-room: S(3C)/S(3D) = 3.876; atomx measured 3.729 (0.962)."""
    lines = sorted(linesets["2p5 3d1"].lines, key=lambda ln: ln.omega_au)
    S_3d, S_3c = lines[1].S, lines[2].S
    fac = _fac_lines()["2p5 3d1"]
    ratio_fac = fac[2][1] / fac[1][1]
    ratio_ax = S_3c / S_3d
    assert ratio_ax / ratio_fac == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# exact internal identities
# ---------------------------------------------------------------------------

def test_sum_rule_vs_determinant_trace(linesets):
    """Sum of CI line strengths == configuration-pair determinant trace.

    The trace sum_{det_i, det_f, q} |<det_f|d_q|det_i>|^2 is mixing
    invariant (independent of H and the CI vectors), so this pins the
    Wigner-Eckart reconstruction S = (2J_i+1) sum_{M_f} |amp|^2, the
    Slater-Condon phases, and the level bookkeeping at once."""
    for cfg, ls in linesets.items():
        assert ls.trace > 0.0
        assert ls.S_sum == pytest.approx(ls.trace, rel=1e-10), cfg


def test_sum_rule_e2_cross_config(fe17_ground):
    """Generic-EL path: E2 between 2p5 3s and 2p5 3d obeys the same
    sum rule (both configs odd parity — E2 allowed, E1 forbidden)."""
    ls = ci_line_strengths(fe17_ground, "2p5 3d1", "2p5 3s1", multipole="E2")
    assert ls.trace > 0.0
    assert ls.S_sum == pytest.approx(ls.trace, rel=1e-10)
    e1 = ci_line_strengths(fe17_ground, "2p5 3d1", "2p5 3s1", multipole="E1")
    assert e1.trace == 0.0 and not e1.lines


def test_one_electron_consistency(linesets):
    """One hole + one electron: CI multiplet total == the jj one-electron
    channel strengths |<kappa_h||C^1||kappa_e>|^2 R^2 exactly (1e-10),
    computed with the same angular convention (reduced_C == angular_eL
    up to sign) and the same fine-grid radial integrals.

    Per line the CI values differ from the pure channels by the physical
    2p1/2/2p3/2 hole-channel mixing (measured: CI/pure = 0.820 for the
    2p3/2-hole line, 1.397 for the 2p1/2-hole line) — the redistribution
    FAC shares, pinned per line by the external_fac gates above."""
    from atomx.angular.coefficients import reduced_C
    from atomx.properties.transitions.angular import angular_eL

    ls = linesets["2p5 3s1"]
    man = ls.manifold
    channels = {}
    for kh in (1, -2):
        oh = next(o for o in man.orbitals if (o.n, o.kappa) == (2, kh))
        oe = next(o for o in man.orbitals if (o.n, o.kappa) == (3, -1))
        red = reduced_C(kh, -1, 1)
        assert abs(red) == pytest.approx(abs(angular_eL(kh, -1, 1)), rel=1e-12)
        channels[kh] = red ** 2 * man._rdip(oh, oe) ** 2

    # exact multiplet-total identity
    assert ls.S_sum == pytest.approx(sum(channels.values()), rel=1e-10)

    # per-line: bracketed by the pure channels, redistributed by mixing
    lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
    S_2p32_line, S_2p12_line = lines[0].S, lines[1].S
    assert 0.7 <= S_2p32_line / channels[-2] <= 1.0   # measured 0.820
    assert 1.0 <= S_2p12_line / channels[1] <= 1.7    # measured 1.397
    # the redistribution is strength-conserving
    assert S_2p32_line + S_2p12_line == pytest.approx(
        channels[-2] + channels[1], rel=1e-10)


def test_m1_not_implemented(fe17_ground):
    with pytest.raises(NotImplementedError):
        ci_line_strengths(fe17_ground, "2p5 3s1", "2p5 3s1", multipole="M1")


def test_det_cap_raises(fe17_ground):
    with pytest.raises(CISizeError):
        ci_line_strengths(fe17_ground, "2p5 3d1", "2p6",
                          multipole="E1", det_cap=3)


def test_ground_config_of_pair():
    assert ground_config_of_pair(
        "1s2 2s2 2p5 3d1", "1s2 2s2 2p6") == "1s2 2s2 2p6"
    assert ground_config_of_pair("2p5 3s1", "2p5 3d1") == "2p6"
    with pytest.raises(ValueError):
        ground_config_of_pair("1s2 2s2 2p6", "1s2 2s2 2p6 3s1")  # N mismatch
    with pytest.raises(ValueError):
        ground_config_of_pair("2p4 3s1 3d1", "2p6")  # two holes


# ---------------------------------------------------------------------------
# compute_transitions wiring
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fe17_pair_atom():
    a = Atom(
             Z=26,
             charge=16,
             config=["1s2 2s2 2p6", "1s2 2s2 2p5 3d1"],
             corrections="none",
        )
    a.solve()
    assert a.converged
    return a


def test_compute_transitions_ci_wiring(fe17_pair_atom, linesets):
    """line_resolution='ci' routes a 2-config atom through the CI engine
    and each row records resolution='ci' with the engine's values."""
    rows = fe17_pair_atom.transitions(multipoles=["E1"],
                                      line_resolution="ci")
    assert len(rows) == 3
    assert all(r.resolution == "ci" for r in rows)
    engine = sorted(linesets["2p5 3d1"].lines, key=lambda ln: ln.omega_au)
    for ln, row in zip(engine, sorted(rows, key=lambda r: r.energy_eV), strict=False):
        assert row.A_rate_per_s == pytest.approx(ln.A_per_s, rel=1e-6)
        assert row.line_strength == pytest.approx(ln.S, rel=1e-6)
        assert row.gf_value == pytest.approx(ln.gf, rel=1e-6)


def test_default_stays_uta(fe17_pair_atom):
    """The default line_resolution remains the UTA path (flip deferred)."""
    rows = fe17_pair_atom.transitions(multipoles=["E1"])
    assert rows and all(r.resolution == "uta" for r in rows)


def test_ci_falls_back_for_velocity_gauge(fe17_pair_atom):
    """The CI engine is length-gauge; velocity requests fall back to UTA."""
    rows = fe17_pair_atom.transitions(multipoles=["E1"], gauge="velocity",
                                      line_resolution="ci")
    assert rows and all(r.resolution == "uta" for r in rows)
