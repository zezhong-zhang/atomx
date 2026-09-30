"""Exact J-resolved CI line strengths vs clean-room FAC (Fe XVII 3C/3D).

Validates :mod:`atomx.properties.transitions.ci_lines` — the determinant-CI
replacement for the UTA-distributed configuration-average line strengths:

- per-line Einstein A against retained FAC 1.1.5 TRTables for both
  static and finite-wavelength photons (fac_ne_like_ci_reference.json): Ne-like
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

Each code optimizes its own local ground-state potential; these orbitals
are not identical. Both use point nuclei, infinite mass and corrections
off. The external gates measure model agreement, not operator accuracy
on common spinors. FAC and Atomx photon modes must match explicitly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402
from atomx.properties.transitions.ci_lines import (  # noqa: E402
    CISizeError,
    ci_line_strengths,
    ground_config_of_pair,
)

FIXTURE = Path(__file__).parent / "data" / "fac_ne_like_ci_reference.json"


def _fac_lines(photon_approximation="finite_wavelength"):
    """FAC reference: config -> [(E_eV, S_standard, A_per_s)] by energy.

    Use FAC's directly printed A. Its E1 amplitude normalization gives
    S_standard = rme^2 / 2 in both photon modes.
    """
    fac = next(c for c in json.loads(FIXTURE.read_text())["cases"] if c["z"] == 26)
    config_by_id = {lv["ILEV"]: cfg for cfg, levels in fac["levels"].items() for lv in levels}
    out: dict = {}
    mode = 0 if photon_approximation == "finite_wavelength" else 1
    for line in fac["lines"][photon_approximation]:
        assert line["mode"] == mode and line["gauge"] == "length"
        S = line["reduced_matrix_element"] ** 2 / 2.0
        out.setdefault(config_by_id[line["upper"]], []).append(
            (line["DELTA_E_eV"], S, line["A_per_s"]))
    for cfg in out:
        out[cfg].sort()
    return out


@pytest.fixture(scope="module")
def fe17_ground():
    """Independent local DFS-gu basis with the same point-nucleus model."""
    g = Atom(
             Z=26,
             charge=16,
             config="1s2 2s2 2p6",
             engine="shooting",
             exchange="gu",
             nuclear="point",
             corrections="none",
        )
    g.solve()
    assert g.converged
    return g._result


@pytest.fixture(scope="module")
def linesets(fe17_ground):
    return {cfg: ci_line_strengths(fe17_ground, cfg, "2p6", multipole="E1")
            for cfg in ("2p5 3s1", "2p5 3d1")}


@pytest.fixture(scope="module", params=["finite_wavelength", "long_wavelength"])
def matched_linesets(request, fe17_ground, linesets):
    mode = request.param
    calculated = linesets if mode == "finite_wavelength" else {
        cfg: ci_line_strengths(fe17_ground, cfg, "2p6", photon_approximation=mode)
        for cfg in ("2p5 3s1", "2p5 3d1")}
    return calculated, _fac_lines(mode)


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
def test_per_line_A_vs_fac(matched_linesets):
    """Independent frozen-basis Fe XVII rates within 3% in both photon modes.

    The 2026-09-30 matched panel differs by at most 1.7%. FAC configuration
    relaxation shifts are disabled; this is not a common-spinor operator gate.
    """
    linesets, fac = matched_linesets
    for cfg, ls in linesets.items():
        ref = fac[cfg]
        lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
        assert len(lines) == len(ref)
        for (E_fac, _S_fac, A_fac), ln in zip(ref, lines, strict=True):
            ratio = ln.A_per_s / A_fac
            assert 0.97 <= ratio <= 1.03, (
                f"{cfg}: line at {E_fac:.1f} eV A ratio {ratio:.3f} outside "
                "[0.97, 1.03] (independent orbital models)")


@pytest.mark.external_fac
def test_transition_energies_vs_fac(matched_linesets):
    """Matched frozen CI excitation energies within 0.6 eV of FAC."""
    linesets, fac = matched_linesets
    for cfg, ls in linesets.items():
        lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
        for (E_fac, _S, _A), ln in zip(fac[cfg], lines, strict=True):
            dE = ln.omega_au * HARTREE_EV - E_fac
            assert abs(dE) <= 0.6, f"{cfg}: dE = {dE:.2f} eV at {E_fac:.1f} eV"


@pytest.mark.external_fac
def test_relative_intensities_vs_fac(matched_linesets):
    """In-multiplet S fractions within 5% of FAC — THE J-resolved gate.

    This checks the distribution across same-J lines, including CI mixing,
    with matching static or finite-wavelength photon operators."""
    linesets, fac = matched_linesets
    for cfg, ls in linesets.items():
        ref = fac[cfg]
        lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
        tot_fac = sum(S for (_E, S, _A) in ref)
        tot_ax = sum(ln.S for ln in lines)
        for (E_fac, S_fac, _A), ln in zip(ref, lines, strict=True):
            frac_fac = S_fac / tot_fac
            frac_ax = ln.S / tot_ax
            assert frac_ax / frac_fac == pytest.approx(1.0, abs=0.05), (
                f"{cfg}: line at {E_fac:.1f} eV fraction "
                f"{frac_ax:.5f} vs FAC {frac_fac:.5f}")


@pytest.mark.external_fac
def test_3c_3d_ratio_vs_fac(matched_linesets):
    """The Fe XVII 3C/3D line-strength ratio matches frozen FAC to 5%."""
    linesets, reference = matched_linesets
    lines = sorted(linesets["2p5 3d1"].lines, key=lambda ln: ln.omega_au)
    S_3d, S_3c = lines[1].S, lines[2].S
    fac = reference["2p5 3d1"]
    ratio_fac = fac[2][1] / fac[1][1]
    ratio_ax = S_3c / S_3d
    assert ratio_ax / ratio_fac == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# exact internal identities
# ---------------------------------------------------------------------------

def test_sum_rule_vs_determinant_trace(fe17_ground):
    """Sum of CI line strengths == configuration-pair determinant trace.

    The trace sum_{det_i, det_f, q} |<det_f|d_q|det_i>|^2 is mixing
    invariant (independent of H and the CI vectors), so this pins the
    Wigner-Eckart reconstruction S = (2J_i+1) sum_{M_f} |amp|^2, the
    Slater-Condon phases, and the level bookkeeping at once."""
    for cfg in ("2p5 3s1", "2p5 3d1"):
        ls = ci_line_strengths(fe17_ground, cfg, "2p6", photon_approximation="long_wavelength")
        assert ls.trace > 0.0
        assert ls.S_sum == pytest.approx(ls.trace, rel=1e-10), cfg


def test_sum_rule_e2_cross_config(fe17_ground):
    """Generic-EL path: E2 between 2p5 3s and 2p5 3d obeys the same
    sum rule (both configs odd parity — E2 allowed, E1 forbidden)."""
    ls = ci_line_strengths(fe17_ground, "2p5 3d1", "2p5 3s1", multipole="E2",
                           photon_approximation="long_wavelength")
    assert ls.trace > 0.0
    assert ls.S_sum == pytest.approx(ls.trace, rel=1e-10)
    e1 = ci_line_strengths(fe17_ground, "2p5 3d1", "2p5 3s1", multipole="E1",
                           photon_approximation="long_wavelength")
    assert e1.trace == 0.0 and not e1.lines


@pytest.mark.snapshot
def test_one_electron_consistency(fe17_ground):
    """One hole + one electron: CI multiplet total == the jj one-electron
    channel strengths |<kappa_h||C^1||kappa_e>|^2 R^2 exactly (1e-10),
    computed with the same angular convention (reduced_C == angular_eL
    up to sign) and the same fine-grid radial integrals.

    Per line the CI values differ from the pure channels by the physical
    2p1/2/2p3/2 hole-channel mixing.  Those weights are atomx's own measured
    numbers, so the test is snapshot-marked; what it must NOT do is accept
    the *no-mixing* value 1.0, which is what the previous brackets
    (``0.7 <= w <= 1.0`` and ``1.0 <= w <= 1.7``) both contained — audit
    finding F15.  The pins below are 0.05 wide around the measured 0.820 /
    1.397, five times closer than the old bracket ends to the value they
    were supposed to exclude.
    """
    from atomx.angular.coefficients import reduced_C
    from atomx.properties.transitions.angular import angular_eL

    ls = ci_line_strengths(fe17_ground, "2p5 3s1", "2p6",
                           photon_approximation="long_wavelength")
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

    # per-line: redistributed by the hole-channel mixing (measured weights)
    lines = sorted(ls.lines, key=lambda ln: ln.omega_au)
    S_2p32_line, S_2p12_line = lines[0].S, lines[1].S
    w_32 = S_2p32_line / channels[-2]
    w_12 = S_2p12_line / channels[1]
    assert w_32 == pytest.approx(0.820, abs=0.05), w_32
    assert w_12 == pytest.approx(1.397, abs=0.05), w_12
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


def test_compute_transitions_ci_wiring(fe17_pair_atom, linesets, fe17_ground):
    """line_resolution='ci' routes a 2-config atom through the CI engine
    and each row records resolution='ci' with the engine's values."""
    rows = fe17_pair_atom.transitions(multipoles=["E1"],
                                      line_resolution="ci", ci_reference=fe17_ground)
    assert len(rows) == 3
    assert all(r.resolution == "ci" for r in rows)
    engine = sorted(linesets["2p5 3d1"].lines, key=lambda ln: ln.omega_au)
    for ln, row in zip(engine, sorted(rows, key=lambda r: r.energy_eV), strict=False):
        assert row.A_rate_per_s == pytest.approx(ln.A_per_s, rel=1e-6)
        assert row.line_strength == pytest.approx(ln.S, rel=1e-6)
        assert row.gf_value == pytest.approx(ln.gf, rel=1e-6)


def test_mcdhf_default_uses_exact_csf_transitions(fe17_pair_atom):
    """MCDHF wavefunctions always use exact CSF transition tensors."""
    rows = fe17_pair_atom.transitions(multipoles=["E1"])
    assert rows and all(r.resolution == "ci" for r in rows)


def test_mcdhf_velocity_gauge_uses_exact_csf_transitions(fe17_pair_atom):
    """Velocity radial integrals use the same exact CSF tensor algebra."""
    rows = fe17_pair_atom.transitions(multipoles=["E1"], gauge="velocity",
                                      line_resolution="ci")
    assert rows and all(r.resolution == "ci" for r in rows)


def test_frequency_dependent_lines_have_no_static_trace(linesets):
    for ls in linesets.values():
        assert ls.photon_approximation == "finite_wavelength"
        assert ls.trace is None
        assert ls.S_sum == pytest.approx(sum(line.S for line in ls.lines), rel=1e-12)


def test_finite_frequency_ci_reversing_configurations_preserves_lines(fe17_ground, linesets):
    forward = sorted(linesets['2p5 3s1'].lines, key=lambda line: line.omega_au)
    backward = ci_line_strengths(fe17_ground, '2p6', '2p5 3s1')
    reverse = sorted(backward.lines, key=lambda line: line.omega_au)
    assert len(forward) == len(reverse)
    for a, b in zip(forward, reverse):
        assert a.omega_au == pytest.approx(b.omega_au, rel=1e-12)
        assert a.S == pytest.approx(b.S, rel=1e-10)
        assert a.A_per_s == pytest.approx(b.A_per_s, rel=1e-10)
