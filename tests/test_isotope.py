"""Tests for the M1.5 isotope-shift machinery.

Covers:
* Normal mass shift via the virial theorem.
* Field-shift factor F_a (Seltzer) vs literature H 1s value.
* Combined isotope shift between two isotopes.
* SCFResult convenience methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.hyperfine import nuclear_moment
from atomx.isotope import (
    FieldShiftParameter,
    IsotopeShiftResult,
    MassShift,
    SpecificMassShift,
    field_shift_factor_per_orbital,
    normal_mass_shift,
    seltzer_F_Ha_per_Bohr2,
    specific_mass_shift,
)
from atomx.scf.solver import DFSolver, Subshell


# ---------------------------------------------------------------------------
# Shared H fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_1s_result():
    shells = [Subshell(n=1, kappa=-1, occupation=1)]
    solver = DFSolver(
        Z=1, subshells=shells, ns=80, ks=9, r_min=1e-5, exchange="hf",
    )
    return solver, solver.solve(max_iter=80, tol=1e-10)


# ---------------------------------------------------------------------------
# Normal mass shift
# ---------------------------------------------------------------------------

class TestNormalMassShift:
    def test_virial_theorem_for_hydrogen(self, h_1s_result):
        """For H 1s, virial theorem gives T = -E_total ≈ 0.5 Ha."""
        _, result = h_1s_result
        ms = result.mass_shift_NMS()
        assert ms.kinetic_energy_au == pytest.approx(0.5, rel=1e-3)
        assert ms.method == "virial"

    def test_hydrogen_mass_shift_magnitude(self, h_1s_result):
        """E_NMS = -T/M ≈ -0.5/1836 ≈ -2.72e-4 Ha ≈ -1.79 THz for H-1.

        Equivalent to m_e/m_p × 13.6 eV ≈ 7.4 meV ≈ 1.79 THz.
        """
        _, result = h_1s_result
        ms = result.mass_shift_NMS()
        expected_Ha = -0.5 / 1836.15
        assert ms.E_NMS_Ha == pytest.approx(expected_Ha, rel=1e-3)
        # -1.7917 THz = -1.7917e6 MHz
        assert ms.E_NMS_MHz / 1e6 == pytest.approx(-1.7917, abs=0.001)

    def test_nms_linear_in_inverse_mass(self, h_1s_result):
        """Doubling M halves NMS in magnitude."""
        _, result = h_1s_result
        ms_H  = result.mass_shift_NMS(isotope=(1, 1))
        ms_D  = result.mass_shift_NMS(isotope=(1, 2))
        # Deuteron is ~2× heavier, so |NMS_D| ≈ |NMS_H| / 2
        ratio = abs(ms_D.E_NMS_Ha / ms_H.E_NMS_Ha)
        assert 0.48 < ratio < 0.52

    def test_explicit_kinetic_energy(self):
        """Passing T explicitly should bypass the virial approximation."""
        ms = normal_mass_shift(
            total_energy=-1.0, mass_me=1000.0, kinetic_energy=2.0,
        )
        assert ms.method == "explicit"
        assert ms.E_NMS_Ha == pytest.approx(-0.002, rel=1e-12)

    def test_invalid_mass_raises(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            normal_mass_shift(total_energy=-1.0, mass_me=0.0)


# ---------------------------------------------------------------------------
# Specific mass shift (user-supplied K_SMS)
# ---------------------------------------------------------------------------

class TestSpecificMassShift:
    def test_basic_positive_K(self):
        sms = specific_mass_shift(K_SMS_au=0.1, mass_me=1836.15)
        assert sms.K_SMS_au == 0.1
        # E_SMS = K_SMS / M = 0.1 / 1836.15 ≈ 5.45e-5 Ha
        assert sms.E_SMS_Ha == pytest.approx(5.45e-5, rel=1e-3)

    def test_linear_in_K(self):
        s1 = specific_mass_shift(K_SMS_au=0.05, mass_me=1836.15)
        s2 = specific_mass_shift(K_SMS_au=0.10, mass_me=1836.15)
        assert s2.E_SMS_Ha == pytest.approx(2.0 * s1.E_SMS_Ha, rel=1e-14)

    def test_inverse_in_mass(self):
        s1 = specific_mass_shift(K_SMS_au=0.1, mass_me=1000.0)
        s2 = specific_mass_shift(K_SMS_au=0.1, mass_me=2000.0)
        assert s2.E_SMS_Ha == pytest.approx(0.5 * s1.E_SMS_Ha, rel=1e-14)

    def test_sign_follows_K(self):
        s_pos = specific_mass_shift(K_SMS_au=+0.1, mass_me=1836.15)
        s_neg = specific_mass_shift(K_SMS_au=-0.1, mass_me=1836.15)
        assert s_pos.E_SMS_Ha > 0
        assert s_neg.E_SMS_Ha < 0
        assert s_pos.E_SMS_Ha == pytest.approx(-s_neg.E_SMS_Ha, rel=1e-14)

    def test_rejects_bad_mass(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            specific_mass_shift(K_SMS_au=0.1, mass_me=0.0)


class TestSCFSpecificMassShift:
    def test_result_mass_shift_SMS(self, h_1s_result):
        _, result = h_1s_result
        sms = result.mass_shift_SMS(K_SMS_au=0.05, isotope=(1, 1))
        assert isinstance(sms, SpecificMassShift)
        assert sms.E_SMS_Ha > 0

    def test_isotope_shift_includes_sms(self, h_1s_result):
        _, result = h_1s_result
        # With K_SMS = 0.05, H−D SMS = 0.05 × (1/M_H − 1/M_D)
        #   M_H ≈ 1836, M_D ≈ 3672  →  ΔSMS ≈ +0.0895 THz
        iso = result.isotope_shift((1, 1), (1, 2), K_SMS_au=0.05)
        assert iso.E_SMS_MHz > 0
        # 0.0895 THz = 89500 MHz  — check ballpark
        assert iso.E_SMS_MHz / 1e6 == pytest.approx(0.0895, abs=0.005)

    def test_isotope_shift_no_sms_by_default(self, h_1s_result):
        _, result = h_1s_result
        iso = result.isotope_shift((1, 1), (1, 2))
        assert iso.E_SMS_MHz == 0.0


class TestNMSExplicitKineticEnergy:
    def test_explicit_T_overrides_virial(self, h_1s_result):
        _, result = h_1s_result
        # Force T = 0.3 (not the virial value 0.5)
        ms = result.mass_shift_NMS(kinetic_energy=0.3)
        assert ms.method == "explicit"
        assert ms.kinetic_energy_au == 0.3

    def test_default_uses_virial(self, h_1s_result):
        _, result = h_1s_result
        ms = result.mass_shift_NMS()
        assert ms.method == "virial"

    def test_isotope_shift_honors_explicit_T(self, h_1s_result):
        _, result = h_1s_result
        iso1 = result.isotope_shift((1, 1), (1, 2))
        iso2 = result.isotope_shift((1, 1), (1, 2), kinetic_energy=0.6)
        # Larger T → larger NMS magnitude
        assert abs(iso2.E_NMS_MHz) > abs(iso1.E_NMS_MHz)


# ---------------------------------------------------------------------------
# Field shift
# ---------------------------------------------------------------------------

class TestFieldShift:
    def test_seltzer_h_1s_matches_literature(self, h_1s_result):
        """F(H 1s) = (2π/3) × 1 × |ψ(0)|² × HARTREE_MHZ / Bohr²[fm²].

        Using hydrogenic |ψ(0)|² = 1/π:
            F_Ha = (2π/3) × (1/π) = 2/3  Ha/Bohr²

        Converting: F_MHz_per_fm² ≈ 1.57 MHz/fm²  (well-known result).
        """
        _, result = h_1s_result
        fs = result.field_shift_parameter()
        F_1s = fs.F_MHz_per_fm2["1s1/2"]
        assert F_1s == pytest.approx(1.57, abs=0.02)

    def test_seltzer_helper_scales_with_Z(self):
        """F ∝ Z at fixed |ψ(0)|²."""
        psi0_sq = 0.1
        F1 = seltzer_F_Ha_per_Bohr2(psi0_sq, Z=1)
        F10 = seltzer_F_Ha_per_Bohr2(psi0_sq, Z=10)
        assert F10 == pytest.approx(10 * F1, rel=1e-14)

    def test_field_shift_parameter_structure(self, h_1s_result):
        """Returns the expected dataclass with consistent dicts."""
        _, result = h_1s_result
        fs = result.field_shift_parameter()
        assert isinstance(fs, FieldShiftParameter)
        # 1s in both dicts
        assert "1s1/2" in fs.F_MHz_per_fm2
        assert "1s1/2" in fs.F_Ha_per_Bohr2
        # total_F_MHz_per_fm2 ≥ single-orbital value (with occ=1 both equal)
        assert fs.total_F_MHz_per_fm2 > 0

    def test_field_shift_orbital_restriction(self, h_1s_result):
        """Passing `orbitals` kwarg filters the returned dict."""
        _, result = h_1s_result
        fs = result.field_shift_parameter(orbitals=["1s1/2"])
        assert list(fs.F_MHz_per_fm2.keys()) == ["1s1/2"]


# ---------------------------------------------------------------------------
# Combined isotope shift
# ---------------------------------------------------------------------------

class TestIsotopeShift:
    def test_h_vs_d_sign_and_ballpark(self, h_1s_result):
        """H → D: heavier deuteron → smaller |NMS|, so ΔNMS = NMS(H)−NMS(D) < 0.

        Expected ΔNMS ≈ -0.5 × (1/M_H - 1/M_D) × HARTREE_MHZ ≈ -0.895 THz
        ≈ -8.954e5 MHz.
        """
        _, result = h_1s_result
        iso = result.isotope_shift((1, 1), (1, 2))
        assert iso.E_NMS_MHz < 0
        # -0.8954 THz = -8.954e5 MHz
        assert iso.E_NMS_MHz / 1e6 == pytest.approx(-0.8954, abs=0.001)

    def test_h_vs_d_dr2(self, h_1s_result):
        """δ⟨r²⟩ = r_rms(H)² − r_rms(D)² < 0 since proton is smaller."""
        _, result = h_1s_result
        iso = result.isotope_shift((1, 1), (1, 2))
        assert iso.delta_r2_fm2 < 0
        # 0.8783² − 2.1421² ≈ 0.772 − 4.589 ≈ -3.82 fm²
        assert iso.delta_r2_fm2 == pytest.approx(-3.82, abs=0.01)

    def test_field_shift_contribution_small_for_h(self, h_1s_result):
        """FS for 1s in H vs D is tiny (~6 MHz) compared to NMS (~-900 GHz)."""
        _, result = h_1s_result
        iso = result.isotope_shift((1, 1), (1, 2))
        # F ≈ 1.57 MHz/fm², δr² ≈ -3.8 fm², occ=1 → FS ≈ -6 MHz
        assert abs(iso.E_FS_MHz) < 20
        # NMS is ~8e5 MHz, so NMS >> FS by >1e5
        assert abs(iso.E_NMS_MHz) > 10000 * abs(iso.E_FS_MHz)

    def test_total_equals_sum_of_pieces(self, h_1s_result):
        _, result = h_1s_result
        iso = result.isotope_shift((1, 1), (1, 2))
        assert iso.E_total_MHz == pytest.approx(
            iso.E_NMS_MHz + iso.E_FS_MHz + iso.E_SMS_MHz, rel=1e-12,
        )

    def test_same_isotope_gives_zero_shift(self, h_1s_result):
        _, result = h_1s_result
        iso = result.isotope_shift((1, 1), (1, 1))
        assert iso.E_NMS_MHz == pytest.approx(0.0, abs=1e-10)
        assert iso.E_FS_MHz == pytest.approx(0.0, abs=1e-10)
        assert iso.E_total_MHz == pytest.approx(0.0, abs=1e-10)

    def test_different_elements_raise(self, h_1s_result):
        _, result = h_1s_result
        with pytest.raises(ValueError, match="same Z"):
            result.isotope_shift((1, 1), (2, 3))

    def test_orbital_breakdown_matches_total(self, h_1s_result):
        """Sum of per-orbital FS equals total FS (for H with occ=1)."""
        _, result = h_1s_result
        iso = result.isotope_shift((1, 1), (1, 2))
        # For H with 1 electron in 1s, sum of one-orbital = total
        orbital_sum = sum(iso.orbital_FS_MHz.values())
        # Allow a factor of occ difference: total uses occ-weighted F, orbital
        # dict is per-orbital un-weighted.  For H (occ=1), they should be equal.
        assert orbital_sum == pytest.approx(iso.E_FS_MHz, rel=1e-12)
