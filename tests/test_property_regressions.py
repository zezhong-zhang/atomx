"""Regression gates for two confirmed physics defects (review 2026-09-19).

Defect 1 — single-configuration (UTA) Einstein A used the *lower* state's
degeneracy.  ``_compute_single_config_transitions`` keeps only pairs with
``dE = E_f − E_i > 0``, so ``i`` is the lower state, but
``_build_single_orbital_result`` handed ``2 j_i + 1`` to
``einstein_A_eL`` instead of the spontaneous-emission weight
``2 j_f + 1``.  For the two H-like Ly-α fine-structure components that
inflated ``A(1s → 2p3/2)`` by exactly ``(2 j_u + 1)/(2 j_l + 1) = 2``
(1.253677e9 instead of 6.268384e8 s⁻¹) while leaving the
``A(1s → 2p1/2)`` row untouched, contradicting the ``gf`` reported in
the *same* row through ``A = 6.6703e15 · gf / (g_u · λ²[Å])``.  The
MCDHF (``calculator.py``) and CI (``ci_lines.py``) engines were already
correct, which is the inconsistency these gates pin down.

Defect 2 — the normal mass shift had the wrong sign
(``E_NMS_Ha = −kinetic / mass_me``).  ``H_NMS = +(m_e/M)·T_e`` raises a
bound level, so ``E_NMS`` is positive and the H→D 1s–2s NMS isotope
shift is ``+671.5554 GHz`` (exact reduced-mass value; measured
``+670.994 GHz``).  The pre-fix code returned ``−1.791704 THz`` for
H 1s and ``−671.5554 GHz`` for the 1s–2s shift.
"""

from __future__ import annotations

import re

import pytest

from atomx import Atom
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.properties.hyperfine import nuclear_moment
from atomx.properties.isotope_shift import normal_mass_shift
from atomx.properties.transitions.calculator import TransitionCalculator

#: NIST/FAC Einstein A of the H-like 2p → 1s Ly-α lines, in s⁻¹.
LY_ALPHA_A_NIST_PER_S = 6.2649e8

#: A-coefficient ⇄ weighted oscillator strength identity (λ in Å):
#: ``A = 6.6703e15 · gf / (g_upper · λ²)``.
A_FROM_GF_CONSTANT = 6.6703e15

#: Exact reduced-mass H→D 1s–2s normal-mass-shift, in GHz (ν_D − ν_H).
HD_1S2S_NMS_GHZ = 671.5554


def _twice_j(label: str) -> int:
    """Twice the total angular momentum of an atomx subshell label.

    ``"2p3/2" -> 3``, ``"1s1/2" -> 1``.  Atomx formats every label as
    ``f"{n}{l}{2j}/2"`` (see ``calculator._subshell_label``).
    """
    match = re.search(r"(\d+)/2$", label)
    if match is None:
        raise ValueError(f"cannot parse 2j from subshell label {label!r}")
    return int(match.group(1))


def _upper_degeneracy(label: str) -> int:
    """``g_upper = 2 j_upper + 1`` for an upper-state label."""
    return _twice_j(label) + 1


class _SolvedAtomView:
    """Minimal ``Atom`` view over an already-solved SCF result.

    :class:`TransitionCalculator` consumes only ``converged``, ``Z``,
    ``_mcdhf_result`` and ``_result`` from the atom.  Exposing a raw
    :class:`DFSolver` result through those four attributes lets the
    single-configuration path run on hydrogenic *virtual* orbitals, which
    the ``Atom`` configuration parser cannot express (zero occupation).
    """

    def __init__(self, result, Z: int):
        self._result = result
        self.converged = True
        self.Z = Z
        self._mcdhf_result = None


@pytest.fixture(scope="module")
def hydrogenic_ly_alpha_rows():
    """H-like 1s → 2p rows through the single-configuration (UTA) path.

    The 2p1/2 and 2p3/2 subshells carry occupation 0, so the DHF solve
    leaves them at the exact Dirac-hydrogenic energies
    (``−0.1250021`` / ``−0.1250004`` Ha) and the line strengths at
    ``S = 1.10982`` / ``2.21963`` — the "exact hydrogenic orbitals
    through the real code path" numbers of the 2026-09 review.
    """
    shells = [
        Subshell(n=1, kappa=-1, occupation=1),
        Subshell(n=2, kappa=+1, occupation=0),   # virtual 2p1/2
        Subshell(n=2, kappa=-2, occupation=0),   # virtual 2p3/2
    ]
    solver = DFSolver(
        Z=1, subshells=shells, ns=80, ks=9, r_min=1e-5, exchange="hf",
    )
    result = solver.solve(max_iter=80, tol=1e-10)
    calc = TransitionCalculator(_SolvedAtomView(result, Z=1))
    rows = calc.compute_all(["E1"], gauge="length")
    return {
        row.final_label: row
        for row in rows
        if row.initial_label == "1s1/2" and row.final_label.startswith("2p")
    }


@pytest.fixture(scope="module")
def neon_public_rows():
    """Ne ``Atom`` → ``transitions(['E1'])`` rows (default engine)."""
    atom = Atom(Z=10).solve()
    assert atom.converged
    return atom.transitions(["E1"])


@pytest.fixture(scope="module")
def hydrogenic_1s_2s_levels():
    """Solved one-electron H 1s and 2s level results (DFSolver)."""
    levels = {}
    for n in (1, 2):
        solver = DFSolver(
            Z=1, subshells=[Subshell(n=n, kappa=-1, occupation=1)],
            ns=80, ks=9, r_min=1e-5, exchange="hf",
        )
        levels[n] = solver.solve(max_iter=80, tol=1e-10)
    return levels


class TestEinsteinAUpperStateDegeneracy:
    """Defect 1: A must use ``2 j_upper + 1`` (gf keeps the lower state)."""

    def test_hydrogenic_ly_alpha_components_are_degenerate(
        self, hydrogenic_ly_alpha_rows,
    ):
        """Both Ly-α fine-structure components have A = 6.2684e8 s⁻¹."""
        rows = hydrogenic_ly_alpha_rows
        assert set(rows) == {"2p1/2", "2p3/2"}

        a_32 = rows["2p3/2"].A_rate_per_s
        a_12 = rows["2p1/2"].A_rate_per_s
        # Pre-fix: 1.253677e9 vs 6.268313e8, ratio 2.0000.
        assert pytest.approx(a_12, rel=1e-3) == a_32
        for label, row in rows.items():
            assert pytest.approx(
                LY_ALPHA_A_NIST_PER_S, rel=0.005,
            ) == row.A_rate_per_s, label

    def test_neon_public_api_pair_ratio_is_unity(self, neon_public_rows):
        """``Atom(Z=10).solve().transitions(['E1'])`` 1s → 2p pair.

        Pre-fix the ratio was 1.993 (both 1s → 2p3/2 and 2s → 2p3/2
        used the lower-state ``2j+1``).
        """
        pair = {
            row.final_label: row
            for row in neon_public_rows
            if row.initial_label == "1s1/2" and row.final_label.startswith("2p")
        }
        assert set(pair) == {"2p1/2", "2p3/2"}

        a_32 = pair["2p3/2"].A_rate_per_s
        a_12 = pair["2p1/2"].A_rate_per_s
        ratio = a_32 / a_12
        assert pytest.approx(1.0, rel=0.01) == ratio

        # The remaining 0.33 % is genuine fine structure: A ∝ ω³·S/(2j_u+1).
        # Pre-fix this exact identity was off by the factor (2j_u+1)/(2j_l+1)=2.
        omega_ratio = (
            pair["2p3/2"].energy_diff_hartree / pair["2p1/2"].energy_diff_hartree
        )
        s_ratio = pair["2p3/2"].line_strength / pair["2p1/2"].line_strength
        expected = 0.5 * omega_ratio ** 3 * s_ratio
        assert pytest.approx(expected, rel=1e-6) == ratio

    def test_a_matches_reported_gf_single_config(self, hydrogenic_ly_alpha_rows):
        """A = 6.6703e15·gf/(g_u·λ²) within the single-configuration rows."""
        for label, row in hydrogenic_ly_alpha_rows.items():
            g_upper = _upper_degeneracy(row.final_label)
            expected = (
                A_FROM_GF_CONSTANT * row.gf_value
                / (g_upper * row.wavelength_angstrom ** 2)
            )
            # Pre-fix the 2p3/2 row was 2× its own gf-implied rate.
            assert pytest.approx(expected, rel=1e-3) == row.A_rate_per_s, label

    def test_a_matches_reported_gf_public_api(self, neon_public_rows):
        """Same identity on every public-API E1 row (all four Ne lines)."""
        assert len(neon_public_rows) >= 4
        for row in neon_public_rows:
            g_upper = _upper_degeneracy(row.final_label)
            expected = (
                A_FROM_GF_CONSTANT * row.gf_value
                / (g_upper * row.wavelength_angstrom ** 2)
            )
            assert pytest.approx(expected, rel=1e-3) == row.A_rate_per_s, (
                row.initial_label, row.final_label,
            )


class TestNormalMassShiftSign:
    """Defect 2: ``E_NMS = +(m_e/M)·⟨T_e⟩`` raises a bound level."""

    def test_bound_level_shift_is_positive(self, hydrogenic_1s_2s_levels):
        """E_NMS(H 1s) = +1.7917 THz (was −1.7917 THz)."""
        ms = hydrogenic_1s_2s_levels[1].mass_shift_NMS(isotope=(1, 1))
        assert ms.kinetic_energy_au > 0
        assert ms.E_NMS_Ha > 0
        assert pytest.approx(1.7917, abs=0.001) == ms.E_NMS_MHz / 1e6

    def test_hd_1s_2s_shift_magnitude_analytic(self):
        """H→D 1s–2s NMS shift (ν_D − ν_H) = +671.5554 GHz (exact μ value).

        Measured shift is +670.994 GHz; the 0.08 % residual is the
        QED/nuclear-size part, so the 0.5 % gate pins the mass-shift
        term itself.  Pre-fix this came out at −671.5554 GHz.
        """
        mass_h = nuclear_moment(1, 1).mass_me
        mass_d = nuclear_moment(1, 2).mass_me

        def nu_nms_mhz(mass: float) -> float:
            """1s → 2s NMS transition shift, ν = E(2s) − E(1s) in MHz."""
            return (
                normal_mass_shift(-0.125, mass).E_NMS_MHz
                - normal_mass_shift(-0.5, mass).E_NMS_MHz
            )

        delta_ghz = (nu_nms_mhz(mass_d) - nu_nms_mhz(mass_h)) / 1e3
        assert delta_ghz > 0  # heavier isotope is blue-shifted
        assert pytest.approx(HD_1S2S_NMS_GHZ, rel=0.005) == delta_ghz

    def test_hd_1s_2s_shift_solved_levels(self, hydrogenic_1s_2s_levels):
        """Same shift built from solved 1s/2s ``mass_shift_NMS`` levels."""
        levels = hydrogenic_1s_2s_levels

        def level_shift_mhz(n: int) -> float:
            res = levels[n]
            return (
                res.mass_shift_NMS(isotope=(1, 2)).E_NMS_MHz
                - res.mass_shift_NMS(isotope=(1, 1)).E_NMS_MHz
            )

        delta_ghz = (level_shift_mhz(2) - level_shift_mhz(1)) / 1e3
        assert delta_ghz > 0
        assert pytest.approx(HD_1S2S_NMS_GHZ, rel=0.005) == delta_ghz
