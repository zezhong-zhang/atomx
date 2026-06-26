"""Fe XVII 3C/3D oscillator-strength ratio vs Bernitt 2012 2σ band.

Reference: Bernitt et al. (2012) Nature 492, 225 (3C/3D EBIT measurement).
The 2σ band quoted in the HYP-175 re-dispatch is f(3C)/f(3D) = 3.7 ± 0.4,
matching the FAC and CHIANTI v10 theoretical line lists (FAC: 2.51/0.62 = 4.02;
CHIANTI v10: ≈ 2.46/0.61 = 4.03).

The gate uses the FAC-parity structure model: ground + 2p⁵3l plus the
2s-hole partner configurations, with ``radial_optimization='fac_frozen'`` to
match FAC's ``OptimizeRadial(['n2'])`` protocol. We confirm the 3C/3D pairings
by selecting the two highest-f-value E1 transitions from the ground state in
the 15 A 2p⁶ ¹S₀ → 2p⁵3d J=1 odd-parity window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Bernitt 2012 Nature 492, 225 — 2σ band quoted in the HYP-175 spec.
BERNITT_2012_F_RATIO_CENTER = 3.7
BERNITT_2012_F_RATIO_2SIGMA = 0.4
BERNITT_2012_F_RATIO_LOW = BERNITT_2012_F_RATIO_CENTER - BERNITT_2012_F_RATIO_2SIGMA
BERNITT_2012_F_RATIO_HIGH = BERNITT_2012_F_RATIO_CENTER + BERNITT_2012_F_RATIO_2SIGMA

# Fe XVII 3C and 3D are both 2p⁶ ¹S₀ → 2p⁵ 3d J=1 (odd parity) E1 lines.
# Reference wavelengths: 3C 15.014 Å, 3D 15.262 Å — both in [14.5, 15.5] Å.
FE17_3CD_WL_LOW_A = 14.5
FE17_3CD_WL_HIGH_A = 15.5


def _solve_fe17():
    """Run the canonical Fe XVII MCDHF solve and return (atom, transitions)."""
    from atomx import Atom

    configs = [
        "1s2 2s2 2p6",
        "1s2 2s2 2p5 3s1",
        "1s2 2s2 2p5 3p1",
        "1s2 2s2 2p5 3d1",
        "1s2 2s1 2p6 3s1",
        "1s2 2s1 2p6 3p1",
        "1s2 2s1 2p6 3d1",
    ]
    atom = Atom(
        Z=26,
        charge=16,
        configs=configs,
        j_resolved_csfs=True,
        breit=False,
        qed=False,
        radial_optimization="fac_frozen",
    )
    atom.solve()
    trs = atom.transitions(multipoles=["E1"])
    return atom, trs


def _ground_index(atom) -> int:
    """Return the index of the lowest-energy state (the 2p⁶ ¹S₀ ground)."""
    energies = atom._result.state_energies
    return int(min(range(len(energies)), key=lambda i: energies[i]))


def _from_ground(transitions, ground_label: str):
    """All E1 transitions originating at the ground level."""
    return [t for t in transitions if t.initial_label == ground_label]


def _pick_3c_3d(transitions_from_ground):
    """Return (f_3c, f_3d) from the FAC/Bernitt 3C/3D wavelength window.

    FAC and CHIANTI place both 3C and 3D in 14.5-15.5 Å.  Selecting the top
    two oscillator strengths globally can accidentally compare 3C against an
    unrelated high-energy J=1 odd line, so this gate first restricts to the
    physical 3C/3D window and then picks the two strongest lines there.
    """
    in_window = [
        t for t in transitions_from_ground
        if FE17_3CD_WL_LOW_A <= float(t.wavelength_angstrom) <= FE17_3CD_WL_HIGH_A
    ]
    by_f = sorted(in_window, key=lambda t: -float(t.f_value))
    if len(by_f) < 2:
        return None, None, None, None
    t_3c, t_3d = by_f[0], by_f[1]
    return (
        float(t_3c.f_value),
        float(t_3d.f_value),
        float(t_3c.wavelength_angstrom),
        float(t_3d.wavelength_angstrom),
    )


def _atomx_f_ratio():
    atom, trs = _solve_fe17()
    ground = f"State_{_ground_index(atom)}"
    gs_trs = _from_ground(trs, ground)
    f_3c, f_3d, wl_3c, wl_3d = _pick_3c_3d(gs_trs)
    return f_3c, f_3d, wl_3c, wl_3d


def _band_status() -> tuple[bool, float | None, str]:
    """Return (in_band, f_ratio, message).

    Falls back to ``False`` with a diagnostic message if either f-value is
    zero / missing.
    """
    f_3c, f_3d, wl_3c, wl_3d = _atomx_f_ratio()
    if f_3c is None or f_3d is None or wl_3c is None or wl_3d is None or f_3d <= 0.0:
        return False, None, (
            "Fe XVII ground state returned fewer than two E1 transitions in "
            f"the FAC 3C/3D wavelength window [{FE17_3CD_WL_LOW_A}, "
            f"{FE17_3CD_WL_HIGH_A}] Å (f_3c={f_3c}, f_3d={f_3d}). "
            "The J=1 odd 2p^5 3d manifold is still misplaced relative to FAC."
        )
    ratio = f_3c / f_3d
    in_band = BERNITT_2012_F_RATIO_LOW <= ratio <= BERNITT_2012_F_RATIO_HIGH
    wl_in_range = (
        FE17_3CD_WL_LOW_A <= wl_3c <= FE17_3CD_WL_HIGH_A
        and FE17_3CD_WL_LOW_A <= wl_3d <= FE17_3CD_WL_HIGH_A
    )
    msg = (
        f"atomx f(3C)={f_3c:.3f} @ {wl_3c:.3f} Å, "
        f"f(3D)={f_3d:.3f} @ {wl_3d:.3f} Å, "
        f"f(3C)/f(3D)={ratio:.3f}; "
        f"Bernitt 2σ band: [{BERNITT_2012_F_RATIO_LOW:.2f}, "
        f"{BERNITT_2012_F_RATIO_HIGH:.2f}]; "
        f"wavelengths in [14.5,15.5] Å range = {wl_in_range}."
    )
    return in_band and wl_in_range, ratio, msg


# Pre-evaluate the status once so failure messages include concrete values.
try:
    _IN_BAND, _F_RATIO, _STATUS_MSG = _band_status()
except Exception as exc:  # pragma: no cover - defensive
    _IN_BAND, _F_RATIO, _STATUS_MSG = False, None, f"atomx solve failed: {exc!r}"


def test_fe17_3c_3d_f_ratio_in_bernitt_2012_2sigma_band():
    """FAC-parity f(3C)/f(3D) must land inside Bernitt 2012's 2sigma band."""
    in_band, ratio, msg = _band_status()
    assert in_band, msg


def test_fe17_3c_3d_band_definition_matches_redispatch():
    """The 2σ band must match Pojo's HYP-175 re-dispatch (3.7 ± 0.4)."""
    assert BERNITT_2012_F_RATIO_CENTER == pytest.approx(3.7)
    assert BERNITT_2012_F_RATIO_2SIGMA == pytest.approx(0.4)
    assert BERNITT_2012_F_RATIO_LOW == pytest.approx(3.3)
    assert BERNITT_2012_F_RATIO_HIGH == pytest.approx(4.1)


@pytest.mark.parametrize("radial_optimization", ["fac_frozen", "fac_frozen_local"])
def test_fe17_fac_frozen_places_bright_3d_lines_in_fac_window(radial_optimization):
    """FAC-frozen radial modes should recover the Fe XVII 15 A line bundle.

    This is not the final Bernitt-ratio gate.  It pins the current FAC-parity
    progress: the bright ground-state E1 strength belongs in the 14.5-15.5 A
    2p^5 3d window and remains in the same total-strength scale as FAC.  After
    the FAC Babushkin E1 normalization fix, FAC-vector/AtomX-radial hybrids
    match the FAC fixture; the remaining production gap is CI mixing, which
    can over-concentrate the same basis strength in the top AtomX eigenline.
    The xfailed ratio gate above covers that unresolved Hamiltonian issue.
    """
    from atomx import Atom

    configs = [
        "1s2 2s2 2p6",
        "1s2 2s2 2p5 3s1",
        "1s2 2s2 2p5 3p1",
        "1s2 2s2 2p5 3d1",
    ]
    atom = Atom(
        Z=26,
        charge=16,
        configs=configs,
        j_resolved_csfs=True,
        breit=False,
        qed=False,
        radial_optimization=radial_optimization,
    )
    atom.solve()
    assert atom._result.converged

    ground = f"State_{_ground_index(atom)}"
    in_window = [
        t for t in _from_ground(atom.transitions(multipoles=["E1"]), ground)
        if FE17_3CD_WL_LOW_A <= float(t.wavelength_angstrom) <= FE17_3CD_WL_HIGH_A
    ]
    by_f = sorted(in_window, key=lambda t: -float(t.f_value))

    assert len(by_f) >= 3
    f0, f1, f2 = (float(t.f_value) for t in by_f[:3])
    assert f0 > 1.5
    assert f1 > 0.2
    # FAC has two bright 3C/3D lines plus a weak J=1 2p^5 3d line
    # (f ~= 0.01).  A previous progress guard required all three lines to be
    # bright, which incorrectly rewarded the over-mixed Hamiltonian this test
    # is meant to catch.
    assert 0.0 < f2 < 0.05
    assert 3.0 < f0 / f1 < 5.5
    # FAC fixture: 3C + 3D + weak J=1 2p^5 3d line = 3.142186529.
    # Keep this as a broad progress guard; the exact strength distribution is
    # still gated by the xfailed Bernitt/FAC ratio test until CI mixing closes.
    total_f = f0 + f1 + f2
    assert total_f == pytest.approx(3.142186529, rel=0.25)
