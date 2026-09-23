"""M5 gate: UTA moments against the committed Fe XVII clean-room FAC array.

``tests/data/fe17_tr_fac_cleanroom.json`` stores the E1 lines of
2p⁵3s → 2p⁶ (2 lines) and 2p⁵3d → 2p⁶ (3 lines) together with FAC's
reduced matrix elements; the line-strength convention is
``S = rme²/2`` (verified exact in the 2026-07 A_r audit — see
``tests/test_ci_line_strengths.py`` and the fixture provenance).

This file pins three separate things (review 2026-09-19, finding M5):

1. **The strength-weighted reference.**  ``uta_moments`` on the stored
   lines is exact by construction (weighted centroid/variance) and the
   total is the sum rule ``Σ S = Σ rme²/2``; the reference values are
   hard-coded below so a fixture or arithmetic drift fails loudly.
2. **The exact reduction path.**  ``uta_simple_from_levels(strengths=…)``
   reproduces that reference to 1e-12 for each array and for the merged
   list.
3. **The weights-only (Unsöld) path is a degeneracy-weighted estimate,
   NOT the strength-weighted moment.**  On this array it sits −1.20 % in
   the 3d centroid and +68.6 % in σ (merged: −4.62 %, +98.2 % in σ, and
   a total weight 92.9x the true Σ S).

   Those numbers are **unchanged by construction** and are *not* a
   weighting bug: all three 2p⁵3d → 2p⁶ lines share J_u = 1, J_l = 0
   (g_u = 3, g_l = 1), so *every* (2J+1)-weight scheme gives them equal
   weight, while their true strengths differ by a factor 240
   (0.000508 / 0.031412 / 0.121760).  No function of (E, 2J+1) alone can
   recover a strength-weighted centroid.  Do not "fix" the proxy by
   reweighting: supply ``strengths=`` (or call ``uta_moments`` on a real
   line list) when the true array moments are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomx.spectrum import Spectrum
from atomx.spectrum.uta import add_uta_array_to_spectrum
from atomx.uta import uta_moments, uta_simple_from_levels

FIXTURE = Path(__file__).parent / "data" / "fe17_tr_fac_cleanroom.json"

#: Strength-weighted reference from the 5 stored FAC lines (S = rme²/2).
REFERENCE = {
    "2p5 3d1": {"n": 3, "S": 0.15368066023194,
                "centroid": 823.690434303247, "std": 5.930550271421},
    "2p5 3s1": {"n": 2, "S": 0.0077676874448597,
                "centroid": 732.080025813821, "std": 6.061559307602},
    "merged": {"n": 5, "S": 0.1614483476768,
               "centroid": 819.282826358777, "std": 20.484207852717},
}

#: Weights-only (Unsöld) proxy from the same level lists, g = 2J+1 = 3.
PROXY = {
    "2p5 3d1": {"total": 9.0, "centroid": 813.794986333333,
                "std": 9.999820653792},
    "merged": {"total": 15.0, "centroid": 781.413796800000,
               "std": 40.592354590921},
}


def _fac_lines():
    """[(config, E_upper_eV, two_J, S=rme²/2)] from the committed fixture."""
    data = json.loads(FIXTURE.read_text())
    return [(ln["config"], ln["E_upper_eV"], ln["two_J"], ln["rme"] ** 2 / 2.0)
            for ln in data["lines"]]


def _selector(name):
    if name == "merged":
        return lambda cfg: True
    return lambda cfg, name=name: cfg == name


@pytest.fixture(scope="module")
def fac_lines():
    return _fac_lines()


@pytest.mark.external_fac
def test_stored_fac_reference_moments(fac_lines):
    """uta_moments on the stored lines == the hard-coded reference.

    Pins the fixture content *and* the weighted-moment arithmetic at
    once: any drift in either shows up here (rel 1e-9)."""
    for name, ref in REFERENCE.items():
        sel = _selector(name)
        m = uta_moments([(E, S) for (cfg, E, _tj, S) in fac_lines if sel(cfg)])
        assert m.n_lines == ref["n"]
        assert m.total_strength == pytest.approx(ref["S"], rel=1e-9)
        assert m.centroid_eV == pytest.approx(ref["centroid"], rel=1e-9)
        assert m.std_eV == pytest.approx(ref["std"], rel=1e-9)


@pytest.mark.external_fac
def test_total_strength_is_half_sum_rme_squared(fac_lines):
    """Σ S over the stored lines == Σ rme²/2 exactly (the sum rule the
    CI/FAC chain also pins; see test_ci_line_strengths.py)."""
    S_direct = sum(S for (_c, _E, _tj, S) in fac_lines)
    m = uta_moments([(E, S) for (_c, E, _tj, S) in fac_lines])
    assert S_direct == pytest.approx(REFERENCE["merged"]["S"], rel=1e-12)
    assert m.total_strength == pytest.approx(S_direct, rel=1e-12)


@pytest.mark.external_fac
def test_strengths_reduction_is_exact(fac_lines):
    """The ``strengths=`` reduction (true per-pair S) == uta_moments.

    This is the supported way to get strength-weighted UTA moments from
    level lists: pass the real line strengths for the allowed pairs and
    leave the forbidden ones out."""
    for name, ref in REFERENCE.items():
        sel = _selector(name)
        upper, g_upper, strengths = [], [], {}
        for (cfg, E, tj, S) in fac_lines:
            if not sel(cfg):
                continue
            strengths[(len(upper), 0)] = S
            upper.append(E)
            g_upper.append(tj + 1)
        m = uta_simple_from_levels(
            upper, [0.0], g_upper=g_upper, g_lower=[1], strengths=strengths)
        assert m.n_lines == ref["n"]
        assert m.total_strength == pytest.approx(ref["S"], rel=1e-12)
        assert m.centroid_eV == pytest.approx(ref["centroid"], rel=1e-12)
        assert m.std_eV == pytest.approx(ref["std"], rel=1e-12)


@pytest.mark.external_fac
def test_weights_only_path_is_unsold_not_strength_weighted(fac_lines):
    """The g_u·g_l proxy reproduces the *degeneracy*-weighted moments and
    deviates from the strength-weighted reference by the documented,
    unchanged-by-construction amounts (3d: −1.20 % centroid, +68.6 % σ;
    merged: −4.62 %, +98.2 %, weight/S = 92.9x).

    A future reader must not "fix" the proxy by reweighting — see the
    module docstring: the three 3d lines share (J_u, J_l) = (1, 0), so
    all (2J+1)-weight schemes give equal weights while the true
    strengths span a factor 240."""
    # --- 2p5 3d -> 2p6: three J=1 levels, g_u=3, g_l=1 ---
    up3d = [(E, tj + 1) for (cfg, E, tj, _S) in fac_lines if cfg == "2p5 3d1"]
    p3d = uta_simple_from_levels(
        [E for (E, _g) in up3d], [0.0],
        g_upper=[g for (_E, g) in up3d], g_lower=[1])
    assert p3d.total_strength == pytest.approx(PROXY["2p5 3d1"]["total"], rel=1e-12)
    assert p3d.centroid_eV == pytest.approx(PROXY["2p5 3d1"]["centroid"], rel=1e-12)
    assert p3d.std_eV == pytest.approx(PROXY["2p5 3d1"]["std"], rel=1e-12)
    dev_c = p3d.centroid_eV / REFERENCE["2p5 3d1"]["centroid"] - 1.0
    dev_s = p3d.std_eV / REFERENCE["2p5 3d1"]["std"] - 1.0
    assert dev_c == pytest.approx(-0.012014, abs=2e-6)   # measured −1.2014 %
    assert dev_s == pytest.approx(+0.686154, abs=2e-5)   # measured +68.6154 %

    # --- merged 2p5 3s + 2p5 3d ---
    up_all = [(E, tj + 1) for (_cfg, E, tj, _S) in fac_lines]
    pm = uta_simple_from_levels(
        [E for (E, _g) in up_all], [0.0],
        g_upper=[g for (_E, g) in up_all], g_lower=[1])
    assert pm.total_strength == pytest.approx(PROXY["merged"]["total"], rel=1e-12)
    assert pm.centroid_eV == pytest.approx(PROXY["merged"]["centroid"], rel=1e-12)
    assert pm.std_eV == pytest.approx(PROXY["merged"]["std"], rel=1e-12)
    dev_cm = pm.centroid_eV / REFERENCE["merged"]["centroid"] - 1.0
    assert dev_cm == pytest.approx(-0.046222, abs=2e-6)  # measured −4.6222 %
    assert pm.total_strength / REFERENCE["merged"]["S"] == pytest.approx(
        92.909, rel=1e-4)                                # measured 92.909x


def test_weights_only_matches_equal_weight_mean(fac_lines):
    """With all J_u equal the proxy centroid is just the arithmetic mean
    of the upper-level energies — the closed form the (2J+1) weights
    reduce to, independent of the strength information they cannot see."""
    up3d = [E for (cfg, E, _tj, _S) in fac_lines if cfg == "2p5 3d1"]
    p = uta_simple_from_levels(up3d, [0.0], g_upper=[3] * 3, g_lower=[1])
    assert p.centroid_eV == pytest.approx(sum(up3d) / 3.0, rel=1e-14)
    assert p.total_strength == pytest.approx(9.0, rel=1e-14)


def test_strengths_path_input_validation():
    """Only the pairs listed in ``strengths`` enter; bad keys raise."""
    with pytest.raises(IndexError, match=r"outside the 2×1 level grid"):
        uta_simple_from_levels([10.0, 11.0], [1.0], strengths={(2, 0): 1.0})
    # All strengths zero / below threshold -> no valid transitions.
    with pytest.raises(ValueError, match="positive-strength"):
        uta_simple_from_levels([10.0], [1.0], strengths={(0, 0): 0.0})
    # A forbidden pair is simply absent, not zero-weighted: the moments
    # then equal uta_moments on the surviving line only.
    m = uta_simple_from_levels([10.0, 11.0], [1.0], strengths={(1, 0): 2.0})
    assert m.n_lines == 1
    assert m.centroid_eV == pytest.approx(10.0)
    assert m.total_strength == pytest.approx(2.0)


@pytest.mark.external_fac
def test_spectrum_bridge_conserves_power(fac_lines):
    """The spectrum bridge emits total_strength × centroid_eV and the
    Gaussian carries that area (power conservation, verified in the
    review; gated here so the M5 chain has an end-to-end check)."""
    lines = [(E, S) for (_c, E, _tj, S) in fac_lines]
    m = uta_moments(lines)
    lam_nm = m.centroid_nm
    # Window ±0.25 nm ≈ ±6.6 bridge-σ (σ_λ = hc·σ_E/E² ≈ 0.038 nm), so the
    # Gaussian tail outside the grid is ~1e-11 of the area.
    lam_lo, lam_hi = lam_nm - 0.25, lam_nm + 0.25
    spec = Spectrum.linear(lam_lo, lam_hi, 4001)
    out = add_uta_array_to_spectrum(
        spec, lines, lam_min_nm=lam_lo, lam_max_nm=lam_hi)
    assert out is not None
    assert spec.line_count == 1
    assert spec.integrate() == pytest.approx(m.total_strength * m.centroid_eV,
                                             rel=1e-4)
