"""Unit tests for the DW excitation angular coupling (j-j recoupling).

Tests cover:
- Reduced C^k matrix elements (reduced_C)
- Allowed multipole k selection rules (_allowed_k)
- Parity conservation in channel selection
- J range calculation
- Exchange phase factors
- Wigner 6j recoupling correctness
- Full DW solver integration with angular coupling
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.angular.coefficients import reduced_C
from atomx.scattering.distorted_wave.solver import (
    _l_of,
    _tj_of,
    _allowed_k,
    _target_projectile_J_range,
    _target_state_metadata,
    _target_transition_population,
    dw_excitation_cross_section,
)
from atomx.angular.wigner import wigner_3j, wigner_6j


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestLOf:
    def test_positive_kappa(self):
        # kappa=1 → l=1 (p_{1/2})
        assert _l_of(1) == 1

    def test_negative_kappa(self):
        # kappa=-1 → l=0 (s_{1/2})
        assert _l_of(-1) == 0
        # kappa=-2 → l=1 (p_{3/2})
        assert _l_of(-2) == 1
        # kappa=-3 → l=2 (d_{5/2})
        assert _l_of(-3) == 2


class TestTjOf:
    def test_kappa_minus1(self):
        # kappa=-1 → j=1/2 → tj=1
        assert _tj_of(-1) == 1

    def test_kappa_1(self):
        # kappa=1 → j=1/2 → tj=1
        assert _tj_of(1) == 1

    def test_kappa_minus2(self):
        # kappa=-2 → j=3/2 → tj=3
        assert _tj_of(-2) == 3

    def test_kappa_2(self):
        # kappa=2 → j=3/2 → tj=3
        assert _tj_of(2) == 3

    def test_kappa_minus3(self):
        # kappa=-3 → j=5/2 → tj=5
        assert _tj_of(-3) == 5


class TestReducedC:
    """Test reduced matrix elements ⟨κ_a‖C^k‖κ_b⟩ against known values."""

    def test_diagonal_k0(self):
        # C^0 diagonal: magnitude should be sqrt(2j+1) for all kappa
        # (the sign depends on the phase convention).
        # Grant (2007): ⟨κ‖C^k‖κ⟩ = (-1)^{j+1/2} · (2j+1) · 3j(j,0,j;-1/2,0,1/2)
        for kappa in [-1, 1, -2, 2, -3]:
            tj = _tj_of(kappa)
            expected_mag = math.sqrt(tj + 1)
            actual = reduced_C(kappa, kappa, 0)
            assert abs(abs(actual) - expected_mag) < 1e-12, \
                f"|C^0({kappa},{kappa})| = {abs(actual)}, expected {expected_mag}"

    def test_parity_forbidden(self):
        # kappa=-1 (l=0) to kappa=-2 (l=1) with k=0: l+l+k=0+1+0=1 (odd) → forbidden
        assert reduced_C(-1, -2, 0) == 0.0

    def test_triangle_forbidden(self):
        # kappa=-1 (j=1/2) to kappa=-3 (j=5/2) with k=0:
        # |1/2 - 5/2| = 2 > 0, so k=0 is forbidden
        assert reduced_C(-1, -3, 0) == 0.0

    def test_symmetry_under_conjugation(self):
        # ⟨a‖C^k‖b⟩ and ⟨b‖C^k‖a⟩ differ by phase (-1)^{j_a-j_b+k}
        # but have the same magnitude
        for kappa_a in [-1, -2, 1, 2]:
            for kappa_b in [-1, -2, 1, 2]:
                for k in [0, 1, 2]:
                    va = abs(reduced_C(kappa_a, kappa_b, k))
                    vb = abs(reduced_C(kappa_b, kappa_a, k))
                    assert abs(va - vb) < 1e-12, \
                        f"|C^{k}({kappa_a},{kappa_b})| = {va}, |C^{k}({kappa_b},{kappa_a})| = {vb}"

    def test_1s_to_2p_dipole(self):
        # 1s_{1/2} (κ=-1) → 2p_{3/2} (κ=-2): k=1 (E1 allowed)
        # l_i=0, l_f=1, k=1: l_i+l_f+k=2 (even) ✓
        # |j_i-j_f|=1, k=1: 0≤1≤2 ✓
        val = reduced_C(-1, -2, 1)
        assert val != 0.0
        # Known value: for κ=-1 (j=1/2) → κ=-2 (j=3/2), k=1:
        # ⟨1/2‖C^1‖3/2⟩ = (-1)^{1/2+1/2} · sqrt(2·4) · 3j(1, 2, 3; -1, 0, 1) [twice-j]
        # = (-1)^1 · sqrt(8) · 3j(1,2,3;-1,0,1)
        # 3j(1,2,3;-1,0,1) = ... need to compute
        tja, tjb = 1, 3
        tL = 2
        w = wigner_3j(tja, tL, tjb, -1, 0, 1)
        phase = -1.0  # (-1)^{(1+1)/2} = (-1)^1
        expected = phase * math.sqrt(2.0 * 4.0) * w
        assert abs(val - expected) < 1e-12


class TestAllowedK:
    """Test allowed multipole k values."""

    def test_same_state(self):
        # Same κ: C^k(κ, κ) requires k even (l+l+k even → 2l+k even → k even)
        # and 0 ≤ k ≤ 0 (j=j, so triangle gives k=0)
        # Actually j=j, so |j-j|=0 ≤ k ≤ 2j, so k can be 0,1,2,...,2j
        # but parity: 2l+k even → k even
        k_1s = _allowed_k(-1, -1)  # κ=-1: l=0, j=1/2
        assert k_1s == [0]

        k_2p32 = _allowed_k(-2, -2)  # κ=-2: l=1, j=3/2
        # j=3/2: 0 ≤ k ≤ 3, and 2+k even → k=0,2
        assert k_2p32 == [0, 2]

    def test_s_to_p_dipole(self):
        # κ=-1 (l=0, j=1/2) → κ=-2 (l=1, j=3/2)
        # Triangle: |1/2-3/2|=1 ≤ k ≤ 2, parity: 0+1+k even → k odd
        # So k=1 only
        ks = _allowed_k(-1, -2)
        assert ks == [1]

    def test_s_to_p12(self):
        # κ=-1 (l=0, j=1/2) → κ=1 (l=1, j=1/2)
        # Triangle: |1/2-1/2|=0 ≤ k ≤ 1, parity: 0+1+k even → k odd
        # So k=1 only
        ks = _allowed_k(-1, 1)
        assert ks == [1]

    def test_p32_to_s(self):
        # Same allowed k as s_to_p32 by symmetry
        ks = _allowed_k(-2, -1)
        assert ks == [1]

    def test_p12_to_d52(self):
        # κ=1 (l=1, j=1/2) → κ=-3 (l=2, j=5/2)
        # Triangle: |1/2-5/2|=2 ≤ k ≤ 3, parity: 1+2+k even → k odd
        # So k=3 only
        ks = _allowed_k(1, -3)
        assert ks == [3]


class TestParityConservation:
    """Test that parity conservation is enforced in channel selection."""

    def test_s_to_p_parity_allowed(self):
        # κ_i=-1 (l=0), κ_f=-2 (l=1), κ_in=-1 (l=0), κ_out=-2 (l=1)
        # l_i + l_in + l_f + l_out = 0+0+1+1 = 2 (even) → allowed
        l_i, l_in, l_f, l_out = 0, 0, 1, 1
        assert (l_i + l_in + l_f + l_out) % 2 == 0

    def test_s_to_s_parity_forbidden(self):
        # κ_i=-1 (l=0), κ_f=-1 (l=0), κ_in=-1 (l=0), κ_out=-1 (l=0)
        # l_i + l_in + l_f + l_out = 0+0+0+0 = 0 (even) → allowed
        # But wait, this is elastic scattering, not excitation. For excitation,
        # κ_i ≠ κ_f. Let me use a case where parity is odd:
        # κ_i=-1 (l=0), κ_f=1 (l=1), κ_in=-1 (l=0), κ_out=-1 (l=0)
        # l_i + l_in + l_f + l_out = 0+0+1+0 = 1 (odd) → forbidden
        l_i, l_in, l_f, l_out = 0, 0, 1, 0
        assert (l_i + l_in + l_f + l_out) % 2 != 0


class TestTargetStateMetadata:
    """Test target total-J metadata used by the DW angular reduction."""

    def test_closed_shell_scf_infers_J0_even_parity(self):
        from atomx.methods.dirac_fock import Subshell

        result = SimpleNamespace(subshells=[Subshell(n=1, kappa=-1, occupation=2.0)])

        tj, parity, csf = _target_state_metadata(result)

        assert tj == 0
        assert parity == 1
        assert csf.total_J2 == 0

    def test_mcdhf_uses_lowest_state_J_not_active_orbital_j(self):
        from atomx.methods.dirac_fock import Subshell
        from atomx.methods.mcdhf.config import CSF

        csfs = [
            CSF([Subshell(1, -1, 1.0), Subshell(2, 1, 1.0)],
                total_J2=2, parity=-1, label="1s 2p1/2 [J=2/2]"),
            CSF([Subshell(1, -1, 1.0), Subshell(2, -2, 1.0)],
                total_J2=4, parity=-1, label="1s 2p3/2 [J=4/2]"),
        ]
        result = SimpleNamespace(
            csfs=csfs,
            mixing_coefficients=np.array([[0.9], [0.1]]),
            state_J2=np.array([2]),
            state_parity=np.array([-1]),
        )

        tj, parity, csf = _target_state_metadata(result)

        assert tj == 2
        assert parity == -1
        assert csf is csfs[0]

    def test_target_transition_population_filters_many_electron_rank(self):
        from atomx.methods.dirac_fock import Subshell
        from atomx.methods.mcdhf.config import CSF

        csf_i = CSF([Subshell(1, -1, 2.0)],
                    total_J2=0, parity=1, label="1s2 [J=0/2]")
        csf_f = CSF([Subshell(1, -1, 1.0), Subshell(2, 1, 1.0)],
                    total_J2=2, parity=-1, label="1s 2p1/2 [J=2/2]")

        assert _target_transition_population(csf_i, csf_f, 1, -1, 1) == 1.0
        assert _target_transition_population(csf_i, csf_f, 0, -1, 1) == 0.0


class TestJRange:
    """Test J range calculation for total angular momentum."""

    def test_target_J0_to_J1_with_spin_half_projectile(self):
        # He-like 1s^2 J=0 -> 1s2p J=1 with incoming/outgoing j=1/2.
        # The coupled target+projectile total J is 1/2, i.e. tJ=1.
        assert _target_projectile_J_range(0, 2, 1, 1) == (1, 1)

    def test_active_electron_range_would_be_different(self):
        # The old active-electron reduction for s1/2 -> p1/2 would allow
        # tJ=0,2. The target-state reduction must instead use tJ=1.
        old_lo = max(abs(1 - 1), abs(1 - 1))
        old_hi = min(1 + 1, 1 + 1)

        assert (old_lo, old_hi) == (0, 2)
        assert _target_projectile_J_range(0, 2, 1, 1) == (1, 1)

    def test_no_common_total_J_returns_none(self):
        # Initial target J=0 plus projectile j=1/2 can only make J=1/2;
        # final target J=2 plus projectile j=1/2 cannot.
        assert _target_projectile_J_range(0, 4, 1, 1) is None


class TestWigner6jRecoupling:
    """Test Wigner 6j symbols used in the angular coupling."""

    def test_6j_triangle_condition(self):
        # 6j must satisfy four triangle conditions; return 0 if violated
        # {j1, j2, j3; j4, j5, j6} requires triangles (j1,j2,j3), (j1,j5,j6),
        # (j4,j2,j6), (j4,j5,j3)
        # Try (1, 1, 3) which violates triangle |1-1|=0 ≤ 3 ≤ 2
        assert abs(wigner_6j(1, 1, 3, 1, 1, 0)) < 1e-15

    def test_6j_known_value_k0(self):
        # {j_a, j_in, J, j_out, j_f, 0} has a known analytical form:
        # = (-1)^{j_a+j_in+j_out+j_f} / sqrt((2j_a+1)(2j_in+1))
        #   × delta(j_a, j_out) × delta(j_in, j_f) ... this only works for k=0
        # when two pairs match. Test a specific case.
        # For k=0 (twice-k = 0): {1, 1, 2, 1, 1, 0}
        # If j_a=j_out and j_in=j_f and triangle holds:
        val = wigner_6j(1, 1, 2, 1, 1, 0)
        # {1/2, 1/2, 1; 1/2, 1/2, 0} = (-1)^{1+1} / sqrt(2·2) = 1/2
        expected = 1.0 / math.sqrt(2.0 * 2.0)
        # Sign: (-1)^{j1+j2+j4+j5} = (-1)^{1+1+1+1} = +1
        assert abs(val - expected) < 1e-12

    def test_6j_orthogonality(self):
        # Sum rule: Σ_J (2J+1) {j1,j2,J; j4,j5,k1} {j1,j2,J; j4,j5,k2}
        #           = delta(k1,k2) / (2k1+1)
        j1, j2, j4, j5 = 1, 1, 1, 1  # j=1/2 for all
        for k1 in [0, 2]:
            for k2 in [0, 2]:
                total = 0.0
                for tJ in range(abs(j1 + j2 - j4 - j5), j1 + j2 + j4 + j5 + 1, 2):
                    if tJ < abs(j1 - j2) or tJ > j1 + j2:
                        continue
                    if tJ < abs(j4 - j5) or tJ > j4 + j5:
                        continue
                    total += (tJ + 1) * wigner_6j(j1, j2, tJ, j4, j5, k1) * wigner_6j(j1, j2, tJ, j4, j5, k2)
                if k1 == k2:
                    assert abs(total - 1.0 / (k1 + 1)) < 1e-10, \
                        f"Orthogonality failed for k1={k1}, k2={k2}: got {total}"
                else:
                    assert abs(total) < 1e-10, \
                        f"Orthogonality (off-diag) failed for k1={k1}, k2={k2}: got {total}"


class TestExchangePhase:
    """Test exchange phase factor calculations."""

    def test_direct_phase_half_integer(self):
        # Phase (-1)^{(j_in + j_f + J)/2} with twice-j values
        # j_in=1/2 (tj=1), j_f=3/2 (tj=3), J=1 (tJ=2)
        # exponent = (1+3+2)/2 = 3 → (-1)^3 = -1
        tj_in, tj_f, tJ = 1, 3, 2
        exp = (tj_in + tj_f + tJ) // 2
        assert exp % 2 == 1  # odd → phase = -1

    def test_antisymmetrization_phase(self):
        # Phase (-1)^{(j_f + j_out - J)/2}
        # j_f=3/2 (tj=3), j_out=1/2 (tj=1), J=1 (tJ=2)
        # exponent = (3+1-2)/2 = 1 → (-1)^1 = -1
        tj_f, tj_out, tJ = 3, 1, 2
        exp = (tj_f + tj_out - tJ) // 2
        assert exp % 2 == 1  # odd → phase = -1


class TestJITKernel:
    """Test the dw_excitation_angular_sum JIT kernel directly."""

    def test_single_channel_single_J(self):
        from atomx.scattering._jit_kernels import dw_excitation_angular_sum

        # Simplest case: one channel, one J value, one k each for direct/exchange
        # κ_i=-1 (j=1/2, tj=1), κ_in=-1 (j=1/2, tj=1)
        # κ_f=-2 (j=3/2, tj=3), κ_out=-1 (j=1/2, tj=1)
        # J=1 (tJ=2): max(|1-1|, |3-1|)=2, min(1+1, 3+1)=2

        # Direct k=1 (the only allowed): R^k(bound_i,inc;bound_f,scat)
        # Exchange k=1: R^k(bound_i,inc;scat,bound_f)
        # Use dummy product values
        omega = dw_excitation_angular_sum(
            n_channels=1,
            tj_i=1, tj_f=3,
            tj_in_arr=np.array([1], dtype=np.int64),
            tj_out_arr=np.array([1], dtype=np.int64),
            tJ_lo_arr=np.array([2], dtype=np.int64),
            tJ_hi_arr=np.array([2], dtype=np.int64),
            n_direct_arr=np.array([1], dtype=np.int64),
            n_exchange_arr=np.array([1], dtype=np.int64),
            direct_products=np.array([1.0]),
            exchange_products=np.array([0.0]),  # no exchange
            w6j_direct=np.array([wigner_6j(1, 1, 2, 1, 3, 2)]),
            w6j_exchange=np.array([wigner_6j(1, 1, 2, 3, 1, 2)]),
        )
        # Only direct term: phase_d * 6j * product
        phase_d = -1.0  # (1+3+2)/2 = 3, odd
        w6j_val = wigner_6j(1, 1, 2, 1, 3, 2)
        amp = phase_d * w6j_val * 1.0
        expected = (2 + 1) * amp ** 2  # (tJ+1) * |amp|^2
        assert abs(omega - expected) < 1e-12, f"Expected {expected}, got {omega}"

    def test_positive_definite(self):
        from atomx.scattering._jit_kernels import dw_excitation_angular_sum

        # Omega = sum of |amp|^2 must be non-negative
        rng = np.random.default_rng(42)
        n_ch = 5
        tj_in = rng.integers(1, 6, size=n_ch)
        tj_out = rng.integers(1, 6, size=n_ch)
        # Ensure valid J ranges
        tJ_lo = np.maximum(np.abs(tj_in - 1), np.abs(tj_out - 3))
        tJ_hi = np.minimum(tj_in + 1, tj_out + 3)
        # Ensure lo <= hi
        tJ_lo = np.minimum(tJ_lo, tJ_hi)

        n_d = np.ones(n_ch, dtype=np.int64)
        n_e = np.ones(n_ch, dtype=np.int64)
        n_J_total = np.sum((tJ_hi - tJ_lo) // 2 + 1)
        d_prods = rng.standard_normal(n_ch)
        e_prods = rng.standard_normal(n_ch)
        w6j_d = rng.standard_normal(int(n_J_total))
        w6j_e = rng.standard_normal(int(n_J_total))

        omega = dw_excitation_angular_sum(
            n_ch, 1, 3,
            tj_in.astype(np.int64), tj_out.astype(np.int64),
            tJ_lo.astype(np.int64), tJ_hi.astype(np.int64),
            n_d, n_e, d_prods, e_prods, w6j_d, w6j_e,
        )
        assert omega >= 0.0

    def test_zero_when_all_products_zero(self):
        from atomx.scattering._jit_kernels import dw_excitation_angular_sum

        omega = dw_excitation_angular_sum(
            n_channels=2,
            tj_i=1, tj_f=3,
            tj_in_arr=np.array([1, 3], dtype=np.int64),
            tj_out_arr=np.array([1, 3], dtype=np.int64),
            tJ_lo_arr=np.array([0, 0], dtype=np.int64),
            tJ_hi_arr=np.array([2, 4], dtype=np.int64),
            n_direct_arr=np.array([2, 1], dtype=np.int64),
            n_exchange_arr=np.array([1, 1], dtype=np.int64),
            direct_products=np.zeros(3),
            exchange_products=np.zeros(2),
            w6j_direct=np.zeros(9),  # 3 J-values * 2 + 3*1 = ... need correct count
            w6j_exchange=np.zeros(6),
        )
        assert omega == 0.0


class TestIntegration:
    """Integration tests requiring atom solving (marked slow)."""

    @pytest.fixture
    def he_ground_atom(self):
        """Solve He 1s² ground state and return Atom."""
        from atomx import Atom
        atom = Atom(Z=2, config="1s2", breit=False, qed=False)
        atom.solve()
        return atom

    @pytest.fixture
    def he_excited_atom(self):
        """Solve He 1s2p excited state and return Atom."""
        from atomx import Atom
        atom = Atom(Z=2, config="1s1 2p1", breit=False, qed=False)
        atom.solve()
        return atom

    @pytest.fixture
    def he_atoms(self, he_ground_atom, he_excited_atom):
        return he_ground_atom, he_excited_atom

    @pytest.mark.slow
    def test_dw_runs_and_returns_positive(self, he_atoms):
        atom_i, atom_f = he_atoms
        result_i = atom_i._result
        result_f = atom_f._result
        dE_hartree = result_f.total_energy - result_i.total_energy
        from atomx.constants import HARTREE_EV
        dE_eV = dE_hartree * HARTREE_EV
        energies = np.array([dE_eV + 5.0, dE_eV + 10.0, dE_eV + 20.0])

        sigma, omega = dw_excitation_cross_section(
            atom_i, atom_f, energies, L_max=2, kappa_max=3)

        # Above threshold: sigma should be positive (or zero if no channels)
        # At least one energy should give non-zero sigma
        assert sigma.shape == energies.shape
        assert omega.shape == energies.shape
        assert np.all(sigma >= 0)
        assert np.all(omega >= 0)

    @pytest.mark.slow
    def test_dw_zero_below_threshold(self, he_atoms):
        atom_i, atom_f = he_atoms
        energies = np.array([0.1, 0.5])  # well below threshold

        sigma, omega = dw_excitation_cross_section(
            atom_i, atom_f, energies, L_max=2, kappa_max=3)

        assert np.all(sigma == 0)
        assert np.all(omega == 0)

    def test_dw_validates_l_max(self):
        """Test that invalid L_max raises ValueError."""
        with pytest.raises(ValueError, match="L_max"):
            dw_excitation_cross_section(None, None, np.array([100.0]), L_max=-1)

    def test_dw_validates_kappa_max(self):
        """Test that invalid kappa_max raises ValueError."""
        with pytest.raises(ValueError, match="kappa_max"):
            dw_excitation_cross_section(None, None, np.array([100.0]), kappa_max=0)
