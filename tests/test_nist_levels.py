"""NIST ASD v5.12 gates.

These pin the two things the external comparison established that no
internal gate could see:

1. **The Breit interaction is a uniform scalar shift, not a CI operator.**
   ``MCDHFSolver.solve`` evaluates Breit + QED once on the converged
   Guerra-averaged orbitals and adds the total to *every* CI state, so no
   excitation energy moves.  Measured across the whole panel:
   ``MCDHF+B+QED`` is byte-identical to ``MCDHF`` on **all 14 systems** that
   have both rungs (see ``docs/VERIFICATION.md`` and
   ``benchmarks/results_nist_levels.csv``).  For Fe XXV, where the Breit
   interaction is a dominant term in the level structure, it contributes
   nothing.
2. **``G¹(1s,2p)``, hence the ¹P–³P splitting, is too large.**  The
   CI identity ``E(¹P) − E(³P) = 2G¹/3`` holds internally to 1 %, so the
   radial integral carries the error; it is invariant from 5 to 17 CSFs.

Both are strict ``xfail``s: they assert the *correct* behaviour and will
fail loudly (XPASS) the moment the defects are fixed, which is the point.

The data is fetched by ``benchmarks/fetch_nist_asd.py`` into
``benchmarks/nist_asd/`` and committed, so this runs offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))

from atomx import Atom, Corrections  # noqa: E402

CM1_PER_HA = 219474.6313632

pytestmark = pytest.mark.external_fac

_CACHE = BENCH / "nist_asd" / "levels" / "asd.006004"
_needs_data = pytest.mark.skipif(
    not _CACHE.exists(),
    reason=("NIST ASD v5.12 cache missing; run "
            "`python benchmarks/fetch_nist_asd.py` to populate "
            "benchmarks/nist_asd/"))


def _cv_excitations(*, breit: bool) -> tuple[np.ndarray, np.ndarray]:
    """C V ``["1s2", "1s1 2p1"]`` excitations (cm-1) and total 2J per level."""
    atom = Atom(
                Z=6,
                charge=4,
                config=["1s2", "1s1 2p1"],
                corrections=Corrections(breit=breit, qed=breit),
           )
    atom.solve(tol=1e-9, max_iter=80)
    res = atom._result
    energies = np.asarray(res.state_energies)
    return (energies - energies.min()) * CM1_PER_HA, np.asarray(res.state_J2)


@_needs_data
def test_breit_shifts_excitation_energies():
    """``breit=True`` must change the C V excitation spectrum, not just totals.

    Was a strict xfail while Breit + QED were one uniform scalar; it XPASSed
    once the corrections gained differential structure, which is the signal
    the gate was designed to give.  C V ``MCDHF+B+QED`` rms improves from
    11199.048 to 10651.963 cm-1.

    Note this gate does NOT certify the Breit *operator*: see
    ``test_breit_operator_matches_closed_shell_scalar``, which measures the
    ``breit_mode="operator"`` path directly and currently fails.
    """

    """``breit=True`` must change the C V excitation spectrum, not just totals."""
    without, _ = _cv_excitations(breit=False)
    with_breit, _ = _cv_excitations(breit=True)
    assert without.shape == with_breit.shape
    # A real CI Breit operator moves the 3P/1P levels differentially.
    assert np.max(np.abs(with_breit - without)) > 100.0, (
        f"largest excitation-energy change from Breit+QED is "
        f"{np.max(np.abs(with_breit - without)):.3f} cm-1")


@_needs_data
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The 1s2p 1P-3P splitting is 1.44x too large and does not respond "
        "to the active space: measured 40481.6 / 40480.5 / 40969.7 / 40481.6 "
        "/ 40968.6 / 40965.7 cm-1 for 5 / 7 / 9 / 9 / 15 / 17 CSFs against "
        "NIST's observed 28172.3 cm-1.  Since the CI identity "
        "E(1P)-E(3P) = 2G^1(1s,2p)/3 holds internally to 1 %, the error is "
        "in the radial G^1 (atomx 0.2762 Ha vs NIST-implied 0.1925 Ha; the "
        "hydrogenic value at Z=6 is 0.2304).  _gk itself is exonerated - it "
        "matches an independent numerical quadrature to 2.4e-4."
    ),
)
def test_cv_1p_3p_splitting_matches_nist():
    """C V ¹P–³P term-average splitting against the observed NIST value."""
    excitations, two_j = _cv_excitations(breit=False)
    order = np.argsort(excitations)
    excitations, two_j = excitations[order], two_j[order]
    p3_zero = excitations[two_j == 0][0]
    p3_two = excitations[two_j == 4][0]
    j1 = np.sort(excitations[two_j == 2])          # lower ³P₁, upper ¹P₁
    assert j1.size == 2, f"expected two J=1 levels, got {j1.size}"
    splitting = j1[1] - (p3_zero + j1[0] + p3_two) / 3.0
    # Observed NIST ASD v5.12: 3P*0/1/2 = 2455191.58/2455179.08/2455314.87,
    # 1P*1 = 2483400.84 cm-1 relative to the 1s2 ground state.
    nist_splitting = 2483400.84 - (2455179.08 + 2455191.58 + 2455314.87) / 3.0
    assert splitting == pytest.approx(nist_splitting, rel=0.02), (
        f"1P-3P splitting {splitting:.1f} cm-1 vs NIST {nist_splitting:.1f}")


@_needs_data
def test_nist_cache_parses_the_he_like_carbon_panel():
    """The committed NIST cache must parse and carry the observed C V levels.

    Guards the acquisition layer: a truncated or tag-mangled file would
    otherwise be compared against silently.  The naive reader this replaces
    recovered 1 row out of 159.
    """
    import nist_asd as nist

    ion = nist.load("C V", with_lines=False)
    assert len(ion.levels) > 100
    observed = [l for l in ion.levels if l.energy_cm1 is not None]
    assert len(observed) > 100
    values = {round(l.energy_cm1, 2) for l in observed}
    assert 2455179.08 in values      # 1s1 2p1 3P*1
    assert 2483400.84 in values      # 1s1 2p1 1P*1


@_needs_data
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The SCALAR Breit is ~3x too large, not the operator.  Measured "
        "externally: DBSR_HF (``dbsr_hf an=Z term=AV mbreit=1``, the "
        "reference implementation this repo is modelled on) gives Ne "
        "1s2 2s2 2p6 Breit = 0.016643629692418 Ha, while "
        "``corrections.breit.compute_breit_energy`` -- the scalar behind "
        "``energy_components['E_Breit']``, and the default "
        "``breit_mode='scalar'`` -- reports 0.050449 Ha, a factor 3.03.  "
        "The per-(2J, parity) GAUNT operator "
        "(src/atomx/methods/mcdhf/breit_ci.py) gives 0.017340 Ha, 4.2% "
        "above DBSR_HF, which is the documented size of its omitted "
        "retardation term and therefore CORRECT.  Consequence: the three "
        "tests that failed while the operator was briefly the default "
        "(+1.126 Ha Co, 0.0053 Ha C) were pinned to the wrong scalar, and "
        "the default was flipped back to that wrong scalar.  This gate "
        "asserts the scalar against DBSR_HF; when it passes, the operator "
        "should be promoted to the default, the three pins re-measured, and "
        "``tests/test_breit_operator.py`` (claimed by the breit_ci "
        "docstring but ABSENT from the tree) written.  Split by v-channel "
        "(monkeypatching _breit_smu_coefficients to keep only v==L, or only "
        "v!=L): the scalar's v=L piece is He 0.00019153 / Ne 0.05040612, "
        "against the breit_ci Gaunt operator's Ne 0.017340 -- a 2.91x "
        "disagreement between two implementations of the SAME Gaunt "
        "coefficient, with the operator's value the DBSR_HF-consistent one.  "
        "And the scalar's v!=L (retardation) piece is only 0.00000001 (He) "
        "and 0.00004247 (Ne), i.e. 0.003% and 0.08% where retardation must be "
        "~4% (it is exactly what makes the Gaunt-only operator sit 4.2% above "
        "DBSR_HF).  So there are TWO independent coefficient errors, not one: "
        "v=L ~3x too large and v!=L ~50x too small.  The factor is "
        "STRUCTURE-INDEPENDENT, which rules out pair counting: He 1s2 (a "
        "single subshell, so no off-diagonal pair exists to double-count) "
        "gives DBSR_HF 0.0000637774 vs atomx 0.000192, ratio 3.01, against "
        "Ne's 3.03.  A constant ~3x across a one-subshell and a "
        "three-subshell atom points at a multiplicative angular or "
        "normalisation factor, not at the i/j loop."
    ),
)
def test_breit_scalar_matches_dbsr_hf():
    """The scalar Breit must reproduce DBSR_HF's Ne value (0.016644 Ha)."""
    from atomx.methods.mcdhf.config import enumerate_jj_csfs_many
    from atomx.methods.mcdhf.solver import MCDHFSolver

    csfs = enumerate_jj_csfs_many(["1s2 2s2 2p6"])
    solver = MCDHFSolver(Z=10, csfs=csfs, ns=113, ks=9,
                         include_breit=True, include_qed=False,
                         breit_mode="scalar")
    result = solver.solve(max_iter=30, scf_tol=1e-9)
    e_breit = result.energy_components["E_Breit"]
    dbsr_hf = 0.016643629692418
    assert e_breit == pytest.approx(dbsr_hf, rel=0.05), (
        f"scalar E_Breit = {e_breit:.6f} Ha vs DBSR_HF {dbsr_hf:.6f} Ha "
        f"(ratio {e_breit / dbsr_hf:.3f})")


@_needs_data
@pytest.mark.xfail(
    strict=True,
    reason=(
        "MCDHF is 45x worse than DHF for the simplest atom in the panel "
        "(NIST rms 2132 vs 47.5 cm-1 over 9 levels), because the Guerra "
        "average is applied to MUTUALLY EXCLUSIVE one-electron "
        "configurations as if they were coexisting fractional occupations: "
        "for ['1s1','2s1','2p1','3s1','3p1','3d1'] every orbital gets "
        "w = 1/9 and each orbital's mean field is built from the other eight "
        "at 1/9 each, though a one-electron atom has no electron-electron "
        "interaction to average.  Signature: the s orbitals come out exact "
        "(1s -0.500007, 2s -0.125002, 3s -0.055556 vs hydrogenic to 1e-6) "
        "while every l>0 orbital is far too DEEP -- 2p -0.361624, 3d -0.163935 "
        "against -0.125 and -0.055556, i.e. the 2p implies Z_eff ~ 1.7 in "
        "NEUTRAL hydrogen.  _collapsed_targets is empty here, so this is not "
        "the R7 seam.  The l>0 selectivity is unexplained: the occupations "
        "are uniform, so it must come from the angular structure of the "
        "direct/exchange coefficients."
    ),
)
def test_h1_mcdhf_orbitals_are_screened():
    """One-electron atoms: a virtual must not be attracted more than the bare nucleus."""
    atom = Atom(
                Z=1,
                config=["1s1", "2s1", "2p1", "3s1", "3p1", "3d1"],
                corrections="none",
                correlation=None,
           )
    atom.solve(tol=1e-9, max_iter=60)
    eps = {k: float(v) for k, v in atom._result.orbital_energies.items()}
    # s orbitals are already correct; assert them so a regression there shows.
    assert eps["1s1/2"] == pytest.approx(-0.5, abs=1e-3)
    assert eps["2s1/2"] == pytest.approx(-0.125, abs=1e-3)
    # l > 0 must not be deeper than hydrogenic.
    assert eps["2p1/2"] > -0.125 - 1e-3, (
        f"2p1/2 = {eps['2p1/2']:.6f} Ha is deeper than the hydrogenic "
        f"-0.125 Ha, i.e. it feels a spurious attraction")


@_needs_data
@pytest.mark.xfail(
    strict=True,
    reason=(
        "MBPT(2) blows up for Ar I.  Measured through "
        "benchmarks/verify_nist_levels.py: the Ar I panel row (model "
        "['1s2 2s2 2p6 3s2 3p6', '... 3p5 4s1', '... 3p5 4p1']) gives "
        "E2 = -1.057787 Ha and an rms of 217223 cm-1 over 15 observed NIST "
        "levels.  The RMS is the solid evidence, not the E2 magnitude: DHF "
        "gives 7185 cm-1 and MCDHF 5687 on the same panel, so MBPT(2) makes "
        "Ar I 30x worse than the rung below it.  Caveat on this gate's bound: "
        "|E2| < 0.35 Ha was chosen from 'argon's whole correlation energy is "
        "~0.7 Ha', but MP2 legitimately recovers 90-110% of a system's "
        "correlation energy, so -1.06 Ha is only ~1.5x the total, not "
        "obviously impossible.  Tighten or replace the bound once the cause "
        "is known.  MBPT(2) is also NOT uniformly the best rung: it wins "
        "on He I (1704 cm-1) and C V (10549) but is worse than DHF on Fe XXV "
        "(105716 vs 99382).  The cause is the VIRTUAL SPACE, measured by "
        "varying mbpt2_energy's controls on the same SCF result: default "
        "-1.057787, e_virt_max=5.0 -> -0.366318, e_virt_max=1.0 -> -0.124591, "
        "kappa_max=2 -> -0.826768, kappa_max=1 -> -0.317550, n_virt_max=3 -> "
        "-0.000001.  Do NOT 'fix' this by capping the virtual space as the "
        "default: measured across the panel, capping destroys values that are "
        "independently validated -- He I E2 None -0.036968 / e_virt_max=5.0 "
        "-0.023151 / 1.0 -0.003107 (the uncapped -0.036968 is the literature "
        "MP2/CBS value pinned by tests/test_mbpt2.py), C V -0.042856 / "
        "-0.005199 / -0.001760, Ne IX -0.044130 / -0.003491 / -0.002319, "
        "Fe XXV -0.045294 / -0.003397 / -0.003227.  The uncapped B-spline "
        "virtual space is a COMPLETE set and converges the MP2 sum toward the "
        "CBS limit; truncating it merely shrinks the correlation energy.  So "
        "Ar I's -1.058 Ha is a system-specific anomaly (candidate: a "
        "near-degeneracy in its 3p/3d virtual manifold), not a generic "
        "high-virtual artefact, and it needs its own diagnosis."
    ),
)
def test_ar_mbpt2_correlation_energy_is_physical():
    """Ar I MBPT(2) must give a sane second-order correlation energy."""
    from atomx.methods import DiracFock, mbpt2_energy

    result = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6").solve(
        DiracFock(breit=False, qed=False))
    e2 = float(mbpt2_energy(result).correlation_energy)
    # Argon's total correlation energy is ~ -0.7 Ha; a second-order term
    # larger than a third of that is not a converged perturbation series.
    assert abs(e2) < 0.35, (
        f"Ar I MBPT(2) E2 = {e2:.6f} Ha is not a physical second-order term")
