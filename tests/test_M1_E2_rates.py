"""End-to-end regression tests for M1 and E2 transition rates.

Validates the full ``rates × multipole × angular`` pipeline for the
magnetic-dipole (L=1) and electric-quadrupole (L=2) cases against
published hydrogenic reference values.

Covers HYP-144 — closes the M1 / E2 verification gate in
``benchmarks/REPORT.md §7.5``. Extended under HYP-229 (T2) to add
angular-factor algebraic tests, E2 gauge-ratio verification at L=2,
additional hydrogenic j-channels, and helium-like M1/E2 benchmarks.

References
----------
* Drake, G. W. F. (1971), *Phys. Rev. A* **3**, 908 — H-like
  ``2s_{1/2} -> 1s_{1/2}`` M1 rate
  ``A(M1) ≈ 2.5e-6 · Z^10 s^-1`` (leading-order Z-scaling, accurate
  to a few % up through Z ~ 30; deviates by ~5–10 % at Z ~ 50 from
  relativistic / retardation corrections beyond the leading term).
* Garstang, R. H. (1957), *Mon. Not. R. Astron. Soc.* **117**, 393 /
  NIST ASD — H I ``3d_{5/2} -> 1s_{1/2}`` E2 rate
  ``A(E2) = 5.96e2 s^-1``, scales as Z^6 for the leading hydrogenic
  contribution.
* Grant, I. P. (2007), *Relativistic Quantum Theory of Atoms and
  Molecules*, §8.4 — multipole rate formulas and gauge conventions.
* Crespo López-Urrutia, J. R., Beiersdorfer, P., Widmann, K., &
  Savin, D. W. (1998), *Phys. Rev. A* **58**, 238 — measured lifetime
  of the He-like O⁶⁺ 1s2s ³S₁ level, 956 (+5/−4) μs, i.e.
  ``A(2^3S_1 -> 1^1S_0, M1) = 1046 (+4/−5) s^-1``.  This is the anchor
  used for the He-like M1 benchmark below; the Lin-Johnson-Dalgarno
  value of 7.0e6 s⁻¹ quoted here previously is inconsistent with that
  measurement by ~700x and is no longer used.

Bug context: pre-fix the M1 pipeline carried a stray /c in ``radial_mL``
plus a missing /(2L+1) in ``angular_mL``, producing rates that were
~ c^2 / (2L+1)^2 = ~ 2000× too small at L=1. The fix here matches the
Grant (2007) Eq. 8.4.17 / FAC ``TRRateMultipole`` convention.
"""

import functools

import numpy as np
import pytest

from atomx.angular.wigner import wigner_3j
from atomx.constants import ALPHA, AU_RATE_PER_S
from atomx.numerics.analytic.hydrogenic_dirac import hydrogenic_dirac_wavefunction
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import (
    build_dirac_matrices,
    solve_dirac,
)
from atomx.numerics.bspline.knots import make_knots
from atomx.properties.transitions.angular import angular_eL, angular_mL
from atomx.properties.transitions.multipole import (
    radial_eL,
    radial_eL_velocity,
    radial_mL,
)
from atomx.properties.transitions.rates import einstein_A_eL, einstein_A_mL

#: Measured M1 rate of the He-like intercombination line
#: (Crespo López-Urrutia et al., Phys. Rev. A 58, 238 (1998)):
#: lifetime 956 (+5/-4) us -> A(O6+ 1s2s 3S1 -> 1s^2 1S0) = 1046 (+4/-5) s^-1.
O6PLUS_M1_RATE_PER_S = 1046.0
#: The same measured rate taken to Z=10 by the M1 Z^10 law,
#: 1046 * (10/8)^10 = 9.74e3 s^-1 (clean-room FAC 1.1.5 gives 9458.4).
NE9_M1_RATE_FROM_O6PLUS = O6PLUS_M1_RATE_PER_S * (10.0 / 8.0) ** 10


def _solve_hlike(Z, kappa, n_idx, ns=80, ks=9, r_max=None, kb=True):
    """Solve the Dirac equation for a hydrogenic ion and return the
    ``n_idx``-th bound (P, Q) spinor plus the bspline grid data."""
    if r_max is None:
        r_max = max(60.0 / Z, 5.0)
    t = make_knots(ns, ks, 0.005 / Z, r_max)
    data = prepare_bspline_data(t, ns, ks, kinetic_balance=kb)
    H, S = build_dirac_matrices(data["ns"], kappa, Z, data)
    res = solve_dirac(
        H, S, data["ns"], kappa,
        nsp=data.get("nsp", data["ns"]),
        nsq=data.get("nsq", data["ns"]),
    )
    P, Q, E = res.get_orbital(n_idx)
    return P, Q, E, data


def _hlike_A_M1_2s_1s(Z, ns=80, r_max=None):
    """Full pipeline: A(M1, H-like 2s_{1/2} -> 1s_{1/2}) at given Z."""
    P_2s, Q_2s, E_2s, data = _solve_hlike(Z, -1, 1, ns=ns, r_max=r_max)
    P_1s, Q_1s, E_1s, _ = _solve_hlike(Z, -1, 0, ns=ns, r_max=r_max)
    omega = E_2s - E_1s
    R = radial_mL(P_2s, Q_2s, P_1s, Q_1s, data, L=1)
    ang = angular_mL(-1, -1, L=1)
    S = (ang * R) ** 2
    return einstein_A_mL(omega, S, L=1, tj_i=1)


def _hlike_A_E2_3d52_1s(Z, ns=100, r_max=None):
    """Full pipeline: A(E2, H-like 3d_{5/2} -> 1s_{1/2}) at given Z."""
    if r_max is None:
        r_max = max(120.0 / Z, 8.0)
    P_3d, Q_3d, E_3d, data = _solve_hlike(Z, -3, 0, ns=ns, r_max=r_max)
    P_1s, Q_1s, E_1s, _ = _solve_hlike(Z, -1, 0, ns=ns, r_max=r_max)
    omega = E_3d - E_1s
    R = radial_eL(P_3d, Q_3d, P_1s, Q_1s, data, L=2)
    ang = angular_eL(-3, -1, L=2)
    S = (ang * R) ** 2
    return einstein_A_eL(omega, S, L=2, tj_i=5)


# ----------------------------------------------------------------------
# M1: H-like 2s_{1/2} -> 1s_{1/2}, Drake (1971) reference  A = 2.5e-6 Z^10
# ----------------------------------------------------------------------

@pytest.mark.parametrize("Z,A_ref", [
    (1, 2.5e-6 * 1**10),
    (2, 2.5e-6 * 2**10),
    (10, 2.5e-6 * 10**10),
])
def test_M1_hlike_2s_to_1s_drake(Z, A_ref):
    """Hydrogenic M1 ``2s_{1/2} -> 1s_{1/2}`` matches Drake's leading
    ``2.5e-6 Z^10 s^-1`` to within 1 % across Z = 1, 2, 10.

    This pins both the magnetic prefactor and the
    ``(kappa_i+kappa_f)/(2L+1)`` algebraic factor — pre-fix the rate was
    ~ 2000× too small at all Z.
    """
    A = _hlike_A_M1_2s_1s(Z)
    assert pytest.approx(A_ref, rel=0.01) == A, (
        f"M1 Z={Z}: atomx={A:.4e}, Drake={A_ref:.4e}, "
        f"ratio={A/A_ref:.4f}"
    )


def test_M1_hlike_2s_to_1s_Z26_moderate_relativistic():
    """At Z = 26 (Fe XXVI) the leading ``2.5e-6 Z^10`` Drake scaling
    starts to drift from the exact relativistic rate by a few %; we
    accept 5 % here as the documented retardation envelope.  The
    ``Z = 1, 2, 10`` tests above pin the prefactor; this one keeps an
    eye on the moderate-Z regime."""
    A = _hlike_A_M1_2s_1s(26)
    A_ref = 2.5e-6 * 26**10
    assert pytest.approx(A_ref, rel=0.05) == A, (
        f"M1 Z=26: atomx={A:.4e}, Drake-Z^10={A_ref:.4e}, "
        f"ratio={A/A_ref:.4f}"
    )


# ----------------------------------------------------------------------
# E2: H-like 3d_{5/2} -> 1s_{1/2}, Garstang (1957) reference
#   A = 5.96e2 s^-1 at Z = 1, scaling as Z^6.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("Z,A_ref", [
    (1, 5.96e2 * 1**6),
    (10, 5.96e2 * 10**6),
    (26, 5.96e2 * 26**6),
])
def test_E2_hlike_3d52_to_1s_garstang(Z, A_ref):
    """Hydrogenic E2 ``3d_{5/2} -> 1s_{1/2}`` matches the Garstang /
    NIST hydrogenic value ``5.96e2 Z^6 s^-1`` to within 1 % across
    Z = 1, 10, 26.

    This pins the existing electric-quadrupole prefactor
    (Grant Eq. 8.4.10 at L=2) and confirms the E2 path is unaffected
    by the M1 fix."""
    A = _hlike_A_E2_3d52_1s(Z)
    assert pytest.approx(A_ref, rel=0.01) == A, (
        f"E2 Z={Z}: atomx={A:.4e}, Garstang Z^6={A_ref:.4e}, "
        f"ratio={A/A_ref:.4f}"
    )


# ----------------------------------------------------------------------
# Sanity: E1 baseline must not regress (the L=1 fix in REPORT §1.1).
# ----------------------------------------------------------------------

def test_E1_baseline_h_2p_to_1s_regression():
    """Closed-form check that the E1 H 2p_{1/2} -> 1s_{1/2} rate
    using the canonical line strength S = 1.110 still returns
    6.265e8 s^-1.  Guards against accidental coupling of the M1 fix
    into the E1 path."""
    A = einstein_A_eL(omega=3.0/8.0, line_strength=1.110, L=1, tj_i=1)
    assert pytest.approx(6.265e8, rel=2e-3) == A


# ======================================================================
# HYP-229 (T2): M1 / E2 prefactor verification extensions
# ======================================================================

# ----- Helpers -----

def _base_angular(tj_i, tL, tj_f, threej_m1=-1, threej_0=0, threej_p1=1):
    """Compute the shared ``phase * sqrt((tj_i+1)(tj_f+1)) * 3j`` factor
    used by both angular_eL and angular_mL (before the kappa-factor)."""
    phase_exp = (tj_f + 1) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0
    threej = wigner_3j(tj_i, tL, tj_f, threej_m1, threej_0, threej_p1)
    return phase * np.sqrt((tj_i + 1.0) * (tj_f + 1.0)) * threej


def _hlike_E2_gauge_ratio(Z, k_up, k_lo, n_up, n_lo):
    """Full rate-level gauge ratio A_velocity / A_length for an E2
    hydrogenic transition."""
    P_lo, Q_lo, bsp, E_lo = hydrogenic_dirac_wavefunction(n_lo, k_lo, Z)
    P_up, Q_up, _, E_up = hydrogenic_dirac_wavefunction(n_up, k_up, Z)
    omega = abs(E_up - E_lo)
    rme_l = radial_eL(P_up, Q_up, P_lo, Q_lo, bsp, L=2)
    rme_v = radial_eL_velocity(P_up, Q_up, P_lo, Q_lo, bsp, L=2,
                                kappa_i=k_up, kappa_f=k_lo, omega=omega)
    ang = angular_eL(k_up, k_lo, 2)
    tj_up = 2 * abs(k_up) - 1
    S_l = (ang * rme_l) ** 2
    S_v = (ang * rme_v) ** 2
    A_l = einstein_A_eL(omega, S_l, 2, tj_up)
    A_v = einstein_A_eL(omega, S_v, 2, tj_up)
    return A_l, A_v


# ======================================================================
# 1. M1 angular prefactor algebraic tests
# ======================================================================

class TestM1AngularPrefactor:
    """Verify the ``(kappa_i + kappa_f) / (2L+1)`` kappa-factor in
    :func:`angular_mL` against hand-assembled 3j-symbol values.

    The M1 angular factor is (Grant 2007 Eq. 8.4.17):

        angular_mL = base_angular × (κ_i + κ_f) / (2L+1)

    where ``base_angular`` is the same phase × sqrt × 3j structure shared
    with :func:`angular_eL`.
    """

    def test_M1_2s_to_1s_kappa_factor(self):
        """M1 2s₁/₂→1s₁/₂ (κ=-1,-1): kappa_factor = -2/3."""
        tj_i, tj_f, tL = 1, 1, 2  # 2×j=1, 2×L=2
        base = _base_angular(tj_i, tL, tj_f)
        kappa_factor = (-1 + (-1)) / 3.0
        expected = base * kappa_factor
        actual = angular_mL(-1, -1, 1)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_M1_p32_to_p12_kappa_factor(self):
        """M1 2p₃/₂→2p₁/₂ (κ=-2,1): kappa_factor = -1/3."""
        tj_i, tj_f, tL = 3, 1, 2
        base = _base_angular(tj_i, tL, tj_f)
        kappa_factor = (-2 + 1) / 3.0
        expected = base * kappa_factor
        actual = angular_mL(-2, 1, 1)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_M1_p12_to_p32_kappa_factor(self):
        """M1 2p₁/₂→2p₃/₂ (κ=1,-2): kappa_factor = -1/3, sign flip from tj_f."""
        tj_i, tj_f, tL = 1, 3, 2
        base = _base_angular(tj_i, tL, tj_f)
        kappa_factor = (1 + (-2)) / 3.0
        expected = base * kappa_factor
        actual = angular_mL(1, -2, 1)
        assert actual == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("L,ki,kf", [
        (1, -1, -1),   # 2s→1s
        (1, -2, 1),    # 2p3/2→2p1/2
        (1, -3, 2),    # 3d5/2→3d3/2
        (2, -1, -1),   # M2 2s→1s (parity: 0+0+2=2 even → M forbidden)
    ])
    def test_M1_selection_rules_consistent(self, L, ki, kf):
        """angular_mL returns zero when selection rules forbid the transition."""
        from atomx.properties.transitions.angular import selection_rules
        allowed = selection_rules(ki, kf, L, 'M')
        val = angular_mL(ki, kf, L)
        if not allowed:
            assert val == pytest.approx(0.0, abs=1e-15), (
                f"M({ki},{kf},L={L}) should be zero (forbidden)")
        # else: an allowed transition may legitimately have a vanishing 3j
        # factor, so there is nothing to assert on this branch.  (This was
        # ``assert val != 0.0 or True``, which asserted nothing.)


# ======================================================================
# 2. E2 angular factor verification
# ======================================================================

class TestE2AngularFactor:
    """Verify E2 (L=2) angular factors against hand-computed 3j values."""

    def test_E2_3d52_to_1s(self):
        """E2 3d₅/₂→1s₁/₂ (κ=-3,-1): angular factor from 3j(5,4,1)."""
        tj_i, tj_f, tL = 5, 1, 4
        base = _base_angular(tj_i, tL, tj_f)
        actual = angular_eL(-3, -1, 2)
        assert actual == pytest.approx(base, abs=1e-12)

    def test_E2_3d32_to_1s(self):
        """E2 3d₃/₂→1s₁/₂ (κ=2,-1): angular factor from 3j(3,4,1)."""
        tj_i, tj_f, tL = 3, 1, 4
        base = _base_angular(tj_i, tL, tj_f)
        actual = angular_eL(2, -1, 2)
        assert actual == pytest.approx(base, abs=1e-12)

    def test_E2_branching_ratio_3d_channels(self):
        """The ratio |ang(3d₃/₂→1s)|² / |ang(3d₅/₂→1s)|² = 2/3.

        This is a well-known hydrogenic result: the two j-channels of an
        nl→n'l' E2 transition carry angular strengths in the ratio
        (2j_small+1)/(2j_large+1) = 4/6 = 2/3 when fine-structure
        splitting is negligible (Z→0 limit).
        """
        ang_52 = angular_eL(-3, -1, 2)
        ang_32 = angular_eL(2, -1, 2)
        ratio = (ang_32 / ang_52) ** 2
        assert ratio == pytest.approx(2.0 / 3.0, rel=1e-10)


# ======================================================================
# 3. E2 rate prefactor at L=2
# ======================================================================

class TestE2RatePrefactor:
    """Verify the L=2 electric-multipole rate prefactor (Grant 8.4.10).

    A(E2) = 2 α⁵ ω⁵ (L+1)(2L+1) / [L (2L+1)!!²] × S / (2j+1)
          = (α⁵ ω⁵ / 15) × S / (2j+1)    at L=2
    """

    def test_E2_prefactor_analytic(self):
        """Hand-computed A(E2) matches einstein_A_eL at L=2."""
        omega = 0.125  # arbitrary, Hartree
        S = 2.5
        tj = 5  # 2j = 5 (j = 5/2)

        # Manual: 5!! = 1·3·5 = 15
        dbl_fact = 15.0
        L = 2
        prefactor_au = (2.0 * ALPHA ** 5 * omega ** 5
                        * (L + 1) * (2 * L + 1)
                        / (L * dbl_fact ** 2))
        A_manual = prefactor_au * S / (tj + 1) * AU_RATE_PER_S

        A_code = einstein_A_eL(omega, S, L, tj)
        assert A_code == pytest.approx(A_manual, rel=1e-13)

    def test_ML_uses_same_prefactor_as_EL(self):
        """einstein_A_mL delegates to einstein_A_eL (same prefactor)."""
        omega, S, L, tj = 0.375, 1.0, 1, 1
        assert einstein_A_mL(omega, S, L, tj) == einstein_A_eL(omega, S, L, tj)


# ======================================================================
# 4. E2 gauge agreement at rate level (L=2)
# ======================================================================

class TestE2GaugeAgreement:
    """For exact hydrogenic eigenstates, the full-order velocity gauge
    (Grant §8.4) must reproduce the length-gauge rate to high precision.
    This extends the RME-level checks in ``test_velocity_gauge_full_order``
    to the full A-coefficient at L=2."""

    @pytest.mark.parametrize("Z", [1, 2, 5])
    def test_E2_gauge_ratio_3d52_to_1s(self, Z):
        """E2 3d₅/₂→1s₁/₂ gauge ratio ≈ 1 (hydrogenic exact)."""
        A_l, A_v = _hlike_E2_gauge_ratio(Z, -3, -1, 3, 1)
        assert A_v == pytest.approx(A_l, rel=1e-2), (
            f"E2 gauge drift at Z={Z}: A_v/A_l = {A_v/A_l:.6f}")

    @pytest.mark.parametrize("Z", [1, 2])
    def test_E2_gauge_ratio_3d32_to_1s(self, Z):
        """E2 3d₃/₂→1s₁/₂ gauge ratio ≈ 1 (hydrogenic exact)."""
        A_l, A_v = _hlike_E2_gauge_ratio(Z, 2, -1, 3, 1)
        assert A_v == pytest.approx(A_l, rel=1e-2), (
            f"E2 gauge drift at Z={Z}: A_v/A_l = {A_v/A_l:.6f}")


# ======================================================================
# 5. Additional hydrogenic M1: fine-structure 2p₃/₂ → 2p₁/₂
# ======================================================================

class TestM1FineStructure:
    """M1 fine-structure transition 2p₃/₂ → 2p₁/₂ in hydrogenic ions.

    This transition has κ_i = -2, κ_f = 1, giving kappa_factor = -1/3
    (different from the 2s→1s value of -2/3), providing an independent
    cross-check on the kappa-factor implementation.

    The rate scales approximately as Z^12 because ω ∝ Z⁴ (fine-structure
    splitting) and A ∝ ω³ × S, with S roughly Z-independent at leading
    order for M1 within the same shell.
    """

    # Audit S4 finding 24: ``test_M1_fs_rate_positive`` (Z=2, 5, 10) asserted
    # only ``A > 0`` — which the pre-fix M1 rate, 2000x too small (see the
    # module docstring), also satisfied.  Deleted rather than re-asserted:
    # the absolute scale of this channel's machinery is pinned by
    # ``test_M1_hlike_2s_to_1s_drake`` (Drake 1971, B-class) and by
    # ``test_M1_fs_Z12_scaling`` below, whose 5x Z-ratio test no scale
    # error can fake.

    def test_M1_fs_Z12_scaling(self):
        """Rate scales as ~Z^12 for the fine-structure transition."""
        rates = {}
        for Z in [5, 10]:
            P_lo, Q_lo, bsp, E_lo = hydrogenic_dirac_wavefunction(2, 1, Z)
            P_up, Q_up, _, E_up = hydrogenic_dirac_wavefunction(2, -2, Z)
            omega = abs(E_up - E_lo)
            R = radial_mL(P_up, Q_up, P_lo, Q_lo, bsp, L=1)
            ang = angular_mL(-2, 1, L=1)
            S = (ang * R) ** 2
            rates[Z] = einstein_A_mL(omega, S, L=1, tj_i=3)

        ratio = rates[10] / rates[5]
        z12_ratio = (10.0 / 5.0) ** 12
        assert ratio == pytest.approx(z12_ratio, rel=0.05), (
            f"M1 fs Z-scaling: ratio={ratio:.2f}, expected Z^12={z12_ratio:.2f}")


# ======================================================================
# 6. Additional hydrogenic E2: 3d₃/₂ → 1s₁/₂ channel
# ======================================================================

class TestE2AdditionalChannels:
    """E2 3d₃/₂ → 1s₁/₂: the second j-channel of the 3d→1s E2 transition.

    At Z=1 the fine-structure splitting is negligible, so both j-channels
    produce nearly equal A-coefficients (the 2/3 reduction in angular
    strength for j=3/2 is compensated by the 3/2 increase in the
    degeneracy factor 1/(2j_i+1): (2/3) × (6/4) = 1).
    """

    def test_E2_3d32_to_1s_Z1(self):
        """E2 3d₃/₂→1s₁/₂ at Z=1: A ≈ 5.94×10² s⁻¹, close to the
        Garstang 3d₅/₂ value of 5.96×10² s⁻¹."""
        P_3d, Q_3d, data, E_3d = hydrogenic_dirac_wavefunction(3, 2, 1)
        P_1s, Q_1s, _, E_1s = hydrogenic_dirac_wavefunction(1, -1, 1)
        omega = E_3d - E_1s
        R = radial_eL(P_3d, Q_3d, P_1s, Q_1s, data, L=2)
        ang = angular_eL(2, -1, L=2)
        S = (ang * R) ** 2
        A = einstein_A_eL(omega, S, L=2, tj_i=3)
        assert pytest.approx(5.94e2, rel=0.01) == A, (
            f"E2 3d3/2→1s Z=1: A={A:.4e}")

    def test_E2_3d32_garstang_Z6_scaling(self):
        """E2 3d₃/₂→1s₁/₂ scales as Z⁶ (same as 3d₅/₂ channel)."""
        A_ref = 5.94e2  # approximate Z=1 value from the test above
        for Z in [10, 26]:
            P_3d, Q_3d, data, E_3d = hydrogenic_dirac_wavefunction(3, 2, Z)
            P_1s, Q_1s, _, E_1s = hydrogenic_dirac_wavefunction(1, -1, Z)
            omega = E_3d - E_1s
            R = radial_eL(P_3d, Q_3d, P_1s, Q_1s, data, L=2)
            ang = angular_eL(2, -1, L=2)
            S = (ang * R) ** 2
            A = einstein_A_eL(omega, S, L=2, tj_i=3)
            A_expected = A_ref * Z ** 6
            assert pytest.approx(A_expected, rel=0.01) == A, (
                f"E2 3d3/2 Z={Z}: A={A:.4e}, expected={A_expected:.4e}")


# ======================================================================
# 7. Helium-like M1 benchmark
# ======================================================================

class TestHeliumLikeM1:
    """He-like 1s2s → 1s² M1 benchmark via MCDHF.

    THE ANCHOR (re-derived 2026-09-25).  The measured rate for this
    transition in the isoelectronic neighbour O⁶⁺ is
    **A(1s2s ³S₁ → 1s² ¹S₀) = 1046 (+4/−5) s⁻¹**, from the measured
    lifetime 956 (+5/−4) μs (Crespo López-Urrutia, Beiersdorfer, Widmann
    & Savin, *Phys. Rev. A* **58**, 238 (1998)).  The M1 Z¹⁰ law puts that
    at **9.74e3 s⁻¹ for Ne IX** (Z=10); an independent clean-room
    FAC 1.1.5 run of the same config set gives **9458.4 s⁻¹** at
    903.91 eV, and FAC's own M1 kernel reproduces the H-like
    2s₁/₂ → 1s₁/₂ rate 2.4943e4 s⁻¹ (= Drake 1971's 2.5e-6 Z¹⁰) to
    0.03 %, so the ~9.5e3 scale is not in doubt.

    The class used to cite "Lin, Johnson & Dalgarno (1977) … A ≈ 7.0e6 s⁻¹
    at Z=10".  That number is inconsistent with the 1998 measurement by
    ~700x and is not used as an anchor here; the docstring that carried it
    also claimed the tracer "has been at 3.94e6 all along", which the
    measurements below contradict.

    WHAT ATOMX GIVES.  The energy-selected tracer is 148x the anchor at
    Z=10 (1.3997e6 vs 9.74e3), 355x at Z=8 (3.7178e5 vs the *measured*
    1046) and 2441x at Z=6 (9.169e4 vs FAC 37.55), with the wrong
    Z-dependence (atomx Ne/O = 3.77 against 9.31 from Z¹⁰ and 10.9 from
    FAC).  The one-electron limit is sound — atomx reproduces Drake's
    H-like 2s → 1s M1 rate to 0.2 % — so the defect is in the many-body M1
    line strength of the jj-coupled CSF basis, not in the operator.

    HISTORY.  The 3.94e6 this class pinned until 2026-09-25 (and the 6.19e6
    before it) were outputs of the pre-``04cd274`` ``build_ci_matrix``,
    which never evaluated ``<CSF_r|H|CSF_s>`` at all (configuration-average
    diagonal plus ad-hoc J patches).  Re-pinning to 1.3997e6 would be
    pinning the current, equally unvalidated output.  Both external gates
    below are therefore ``xfail(strict=True)``: the He-like M1 rate is not
    currently computable by atomx, and this is the loud record of that.

    MECHANISM (measured, read-only diagnosis 2026-09-25).  The M1
    amplitude is an (αZ)²-order cancellation residue of two ~2e-3 radial
    integrals — ∫P_2s Q_1s r dr = −2.102121e-3 vs ∫Q_2s P_1s r dr =
    +1.937630e-3.  The (1s, 2s) spinors the MCDHF hands to the kernel
    cancel only to 7.83 % (residual −1.644909e-4) where the exact
    hydrogenic Z=10 pair cancels to 0.598 % (−1.222498e-5, which through
    the *same* kernel gives 2.5101e4 s⁻¹ = Drake's 2.5e-6 Z¹⁰), and the
    residual is Z-independent (−2.217e-4 / −1.737e-4 / −1.645e-4 at
    Z=6/8/10) where the physics requires R ∝ Z².  Two further defects sit
    in the same path: ``calculator.py:693`` gives the
    closed-shell-J=0 → one-electron-promoted channel a recoupling factor
    √(2/3) where an independent m-scheme CG/3j check gives exactly 1 (1.5x
    in the rate), and ``angular/csf.py:170-183`` builds the population
    factor from the *initial* occupation, so the two-electron change
    2s² → 1s2s gets a spurious 1.704971e6 s⁻¹ "M1" line at 1004.43 eV.
    """

    @pytest.mark.slow
    @pytest.mark.xfail(strict=True, reason=(
        "Ne IX 2^3S1 -> 1^1S0 M1: atomx gives 1.3997e6 s^-1 for the "
        "energy-selected tracer (906.004 eV) against the external anchor "
        "9.74e3 s^-1 (measured O6+ rate 1046 (+4/-5) s^-1, lifetime "
        "956 (+5/-4) us, Crespo Lopez-Urrutia et al., PRA 58, 238 (1998), "
        "scaled by the M1 Z^10 law; an independent clean-room FAC 1.1.5 run "
        "gives 9458.4 s^-1 at 903.91 eV).  148x too large, and the "
        "Z-dependence is wrong too (atomx Ne/O = 3.77 vs 9.31 for Z^10 and "
        "10.9 for FAC).  The one-electron M1 limit is correct (atomx "
        "reproduces Drake's H-like 2s1/2->1s1/2 2.5e-6 Z^10 to 0.2 %), so "
        "the defect is the many-body M1 line strength of the jj-coupled "
        "CSF basis, not the operator or prefactor.  The 3.94e6 previously "
        "pinned here was the output of the pre-04cd274 configuration-"
        "average CI matrix and is NOT restored (it is 400x the anchor)."))
    def test_helike_M1_Z10_vs_measured_anchor(self):
        """Ne IX tracer within 15 % of the measured-sequence M1 rate.

        The transition is selected by energy (903.91 eV per FAC; the
        measurement-derived value is ~904 eV), not by ``max(A)``.
        """
        he = _he_like_ne_m1()
        trs = [t for t in he.transitions(multipoles=["M1"])
               if t.A_rate_per_s > 1e4]
        assert trs, "Expected at least one strong M1 transition"

        tracer = min(trs, key=lambda t: abs(t.energy_eV - 903.91))
        assert abs(tracer.energy_eV - 903.91) < 10.0, (
            f"no M1 transition near the 2³S₁ → 1¹S₀ energy: nearest is "
            f"{tracer.energy_eV:.1f} eV")
        assert pytest.approx(NE9_M1_RATE_FROM_O6PLUS, rel=0.15) == (
            tracer.A_rate_per_s), (
            f"Ne IX M1 tracer A = {tracer.A_rate_per_s:.4e} s^-1 vs the "
            f"measured-sequence anchor {NE9_M1_RATE_FROM_O6PLUS:.4e} s^-1 "
            f"({tracer.A_rate_per_s / NE9_M1_RATE_FROM_O6PLUS:.1f}x)")

    @pytest.mark.slow
    @pytest.mark.xfail(strict=True, reason=(
        "O VII 2^3S1 -> 1^1S0 M1, the one point in this isoelectronic "
        "sequence with a direct measurement: A = 1046 (+4/-5) s^-1 "
        "(lifetime 956 (+5/-4) us, Crespo Lopez-Urrutia et al., PRA 58, 238 "
        "(1998)).  atomx gives 3.7178e5 s^-1 for the energy-selected tracer "
        "at 561.851 eV (FAC 1.1.5: 870.09 s^-1 at 559.94 eV) — 355x too "
        "large.  Same root cause as the Z=10 case: the many-body M1 line "
        "strength of the jj-coupled CSF basis (the one-electron M1 kernel "
        "is validated against Drake 1971)."))
    def test_helike_M1_O7_vs_measured_o6plus(self):
        """O VII tracer against the *measured* O⁶⁺ M1 rate (no Z-scaling)."""
        he = _he_like_o_m1()
        trs = [t for t in he.transitions(multipoles=["M1"])
               if t.A_rate_per_s > 1e2]
        assert trs, "Expected at least one M1 transition"

        tracer = min(trs, key=lambda t: abs(t.energy_eV - 559.94))
        assert abs(tracer.energy_eV - 559.94) < 10.0, (
            f"no M1 transition near the 2³S₁ → 1¹S₀ energy: nearest is "
            f"{tracer.energy_eV:.1f} eV")
        assert pytest.approx(O6PLUS_M1_RATE_PER_S, rel=0.15) == (
            tracer.A_rate_per_s), (
            f"O VII M1 tracer A = {tracer.A_rate_per_s:.4e} s^-1 vs the "
            f"measured O6+ {O6PLUS_M1_RATE_PER_S:.1f} s^-1 "
            f"({tracer.A_rate_per_s / O6PLUS_M1_RATE_PER_S:.0f}x)")


@functools.lru_cache(maxsize=1)
def _he_like_ne_m1():
    """Solved He-like Ne (Z=10, charge=8) for the M1 gate."""
    from atomx import Atom, Basis

    he = Atom(
              Z=10,
              charge=8,
              config=["1s2", "1s1 2s1", "2s2", "2p2"],
              basis=Basis(ns=60, ks=9),
         )
    he.solve(max_iter=80, tol=1e-4)
    if not he.converged:
        pytest.skip("He-like Z=10 MCDHF did not converge")
    return he


@functools.lru_cache(maxsize=1)
def _he_like_o_m1():
    """Solved He-like O (Z=8, charge=6) — the measured point of the sequence."""
    from atomx import Atom, Basis

    he = Atom(
              Z=8,
              charge=6,
              config=["1s2", "1s1 2s1", "2s2", "2p2"],
              basis=Basis(ns=60, ks=9),
         )
    he.solve(max_iter=80, tol=1e-4)
    if not he.converged:
        pytest.skip("He-like Z=8 MCDHF did not converge")
    return he


# ======================================================================
# 8. Helium-like E2 benchmark
# ======================================================================

class TestHeliumLikeE2:
    """He-like 1s3d → 1s² E2 benchmark via MCDHF.

    Uses configs ``["1s2", "1s1 3d1"]`` to create the 3d excited state.
    The dominant E2 transitions (ground → 3d states) are compared against
    the hydrogenic E2 rate at the same Z, adjusted for screening.

    Limitation: the jj-coupled MCDHF states do not correspond to clean
    LS terms (³D₂ vs ¹D₂), so the benchmark verifies the total E2 rate
    is of the correct order and that the gauge ratio is within 10 % of
    unity for the strong channels (the old one-sided ``> 0.9`` accepted a
    velocity rate 1000x the length rate).
    """

    @pytest.mark.slow
    def test_helike_E2_Z10_order_of_magnitude(self):
        """He-like E2 at Z=10: dominant transition A ~ 10⁸–10⁹ s⁻¹.

        The hydrogenic E2 3d→1s rate at Z=10 is ~6×10⁸ s⁻¹; the He-like
        rate includes screening corrections and multi-electron coupling.
        """
        from atomx import Atom, Basis
        he = Atom(Z=10, charge=8, config=["1s2", "1s1 3d1"], basis=Basis(ns=80, ks=9))
        he.solve(max_iter=80, tol=1e-4)
        if not he.converged:
            pytest.skip("He-like Z=10 MCDHF did not converge")

        trs = he.transitions(multipoles=["E2"])
        strong = [t for t in trs if t.A_rate_per_s > 1e6]
        assert len(strong) >= 1, (
            f"Expected strong E2 transitions, got {len(trs)} total, "
            f"max A={max((t.A_rate_per_s for t in trs), default=0):.4e}")

        A_max = max(t.A_rate_per_s for t in strong)
        assert 1e7 < A_max < 5e9, (
            f"He-like E2 Z=10: A_max={A_max:.4e}, expected O(10⁸–10⁹)")

    @pytest.mark.slow
    def test_helike_E2_Z10_gauge_ratio(self):
        """He-like E2 at Z=10: gauge ratio for the dominant transition
        must be within 10 % of unity (single-config MCDHF basis quality
        check).

        ``gauge_ratio`` is ``A_velocity / A_length``
        (``src/atomx/properties/transitions/calculator.py:103``); the old
        gate ``> 0.90`` accepted a velocity rate 1000x the length rate.
        """
        from atomx import Atom, Basis
        he = Atom(Z=10, charge=8, config=["1s2", "1s1 3d1"], basis=Basis(ns=80, ks=9))
        he.solve(max_iter=80, tol=1e-4)
        if not he.converged:
            pytest.skip("He-like Z=10 MCDHF did not converge")

        trs = he.transitions(multipoles=["E2"], gauge="both")
        strong = [t for t in trs if t.A_rate_per_s > 1e6 and t.gauge_ratio is not None]
        assert len(strong) >= 1

        for t in strong:
            assert abs(t.gauge_ratio - 1.0) < 0.10, (
                f"E2 gauge ratio {t.gauge_ratio:.4f} more than 10 % from "
                f"unity for {t.initial_label}->{t.final_label}")
