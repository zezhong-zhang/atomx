"""Length / velocity gauge invariance as a structure-quality diagnostic.

For exact eigenstates of a common Dirac-Coulomb Hamiltonian the Babushkin
(length) and Coulomb (velocity) reduced matrix elements of the dipole
operator agree exactly; for any approximate basis their ratio is the
FAC-style gauge-invariance diagnostic (Gu 2008, *Can. J. Phys.* 86, 675,
§3.2; Grant 2007, §8.4). This module exercises the new
``gauge='both'`` path on :class:`atomx.properties.transitions.TransitionCalculator`
and gates the hydrogenic ratio at numerical precision.

The full (κ_i, κ_f)-dependent velocity-gauge reduction for the *other*
E1 channels (p_{3/2}↔s_{1/2}, p↔d, d↔f) and for all higher multipoles
is tracked as the T1 follow-up — see ``radial_eL_velocity`` for the
current scope.
"""

from __future__ import annotations

import pytest

from atomx.methods.hydrogenic_reference import hydrogenic_dirac_wavefunction
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity
from atomx.properties.transitions.rates import einstein_A_eL


def _hydrogenic_e1_gauge_ratio(n_lo: int, k_lo: int,
                               n_up: int, k_up: int, Z: int,
                               *, ns: int = 120, ks: int = 9) -> tuple[float, float, float]:
    """Compute (A_length, A_velocity, |A_v/A_l - 1|) for a hydrogenic E1.

    Uses :func:`hydrogenic_dirac_wavefunction` to obtain numerically exact
    Dirac-Coulomb orbitals on a shared B-spline basis (so both transition
    legs are eigenstates of the *same* Hamiltonian), then calls
    :mod:`atomx.properties.transitions` directly. This isolates the gauge
    diagnostic from any SCF averaging / configuration-mixing residuals.
    """
    P_lo, Q_lo, bsp, E_lo = hydrogenic_dirac_wavefunction(n_lo, k_lo, Z, ns=ns, ks=ks)
    P_up, Q_up, _, E_up = hydrogenic_dirac_wavefunction(n_up, k_up, Z, ns=ns, ks=ks)

    omega = abs(E_up - E_lo)
    rme_l = radial_eL(P_up, Q_up, P_lo, Q_lo, bsp, 1,
                      kappa_i=k_up, kappa_f=k_lo, omega=-omega)
    rme_v = radial_eL_velocity(P_up, Q_up, P_lo, Q_lo, bsp, 1,
                                kappa_i=k_up, kappa_f=k_lo, omega=-omega)
    ang = angular_eL(k_up, k_lo, 1)
    tj_up = 2 * abs(k_up) - 1

    S_l = (ang * rme_l) ** 2
    S_v = (ang * rme_v) ** 2
    A_l = einstein_A_eL(omega, S_l, 1, tj_up)
    A_v = einstein_A_eL(omega, S_v, 1, tj_up)
    return A_l, A_v, abs(A_v / A_l - 1.0)


# ---------------------------------------------------------------------------
# 1. Hydrogenic — gauge invariance at numerical precision (<= 1e-9).
# ---------------------------------------------------------------------------
# These exercise the canonical s_{1/2} ↔ p_{1/2} E1 channel for which
# ``radial_eL_velocity`` implements the FAC-normalised derivative-free
# form.  Measured at ns=120/ks=9: 2.08e-11 (Z=1), 3.84e-12 (Z=2),
# 1.09e-12 (Z=3), 3.37e-12 (Z=5), 3.55e-11 (Z=8); the gate is 1e-9.

class TestHydrogenicGaugeInvariance:

    def test_H_2p12_to_1s_at_numerical_precision(self):
        """H 2p_{1/2} → 1s_{1/2}: gauge ratio at numerical precision.

        Measured A_l = 6.268313e8 s^-1 (5.4e-4 above NIST) and
        |A_v/A_l - 1| = 2.08e-11.  The old gates (1e-2 on NIST, 1e-3 on
        the ratio) were 7-8 orders looser than the theorem holds.
        """
        A_l, A_v, dev = _hydrogenic_e1_gauge_ratio(1, -1, 2, 1, Z=1)
        assert A_l > 0 and A_v > 0
        # NIST: A(H 2p -> 1s) = 6.2649e8 s^-1 — sanity-check the
        # length-gauge value before comparing the two gauges.
        assert abs(A_l - 6.2649e8) / 6.2649e8 < 1e-3
        assert dev < 1e-9, f"hydrogenic gauge deviation {dev:.3e} exceeds 1e-9"

    def test_He_plus_2p12_to_1s_at_numerical_precision(self):
        """He+ (Z=2) hydrogenic: gauge ratio at numerical precision."""
        _, _, dev = _hydrogenic_e1_gauge_ratio(1, -1, 2, 1, Z=2)
        assert dev < 1e-9

    @pytest.mark.parametrize("Z", [3, 5, 8])
    def test_higher_Z_hydrogenic(self, Z):
        """Hydrogenic 2p_{1/2} → 1s for low-to-moderate Z: <= 1e-9.

        Measured 1.09e-12 (Z=3), 3.37e-12 (Z=5), 3.55e-11 (Z=8) at
        ns=120/ks=9, so the gate is ~30x the worst point.  (The old bound
        was 5e-3, and its docstring guessed "~few×10⁻⁴" — both far above
        what the derivative-free Grant reduction actually achieves here,
        and far below the Gu 2008 ground-state E1 threshold of 0.10.)
        """
        _, _, dev = _hydrogenic_e1_gauge_ratio(1, -1, 2, 1, Z=Z)
        assert dev < 1e-9


# ---------------------------------------------------------------------------
# 2. End-to-end API: TransitionCalculator must populate both gauges.
# ---------------------------------------------------------------------------

class TestGaugeBothApi:

    def test_gauge_both_populates_A_length_and_A_velocity(self):
        """``gauge='both'`` populates ``A_length`` and ``A_velocity`` fields."""
        from atomx import Atom

        # Default engine (shooting); the gauge machinery runs on grid
        # orbitals through the workspace seam and its quality is gated
        # against bspline in test_transitions_engine_uniform.py.
        ne = Atom(Z=10)
        ne.solve(max_iter=50, tol=1e-4)
        if not ne.converged:
            pytest.skip("Ne single-config DHF did not converge")

        results = ne.transitions(multipoles=["E1"], gauge="both")
        assert len(results) > 0, "expected at least one E1 transition for Ne"
        for tr in results:
            assert tr.A_length is not None, "gauge='both' must populate A_length"
            assert tr.A_velocity is not None, "gauge='both' must populate A_velocity"
            # A_rate_per_s stays length-gauge by default contract — the
            # "Don't change the default A-rate value" hard rule.
            assert tr.A_rate_per_s == tr.A_length
            assert tr.gauge == "length"
            assert tr.gauge_ratio is not None

        # The 2p_{1/2} → 2s channel is the well-behaved s ↔ p_{1/2} case
        # this PR scopes; assert it is at the spec's excited-state
        # threshold (≤ 0.20) for neutral Ne.
        good = [t for t in results
                if t.initial_label.startswith("2p1/2")
                and t.final_label.startswith("2s1/2")]
        if good:
            tr = good[0]
            dev = abs(tr.A_velocity / tr.A_length - 1.0)
            assert dev < 0.20, (
                f"Ne 2p_{{1/2}} → 2s_{{1/2}} gauge ratio dev {dev:.3f} > 0.20"
            )

    def test_invalid_gauge_raises(self):
        """A bad gauge string must raise ``ValueError`` early."""
        from atomx import Atom, Basis

        h = Atom(Z=1, config="1s1", basis=Basis(ns=40, ks=7))
        h.solve(max_iter=30, tol=1e-4)
        if not h.converged:
            pytest.skip("H did not converge")
        with pytest.raises(ValueError):
            h.transitions(multipoles=["E1"], gauge="banana")


# ---------------------------------------------------------------------------
# 3. Many-electron MCDHF reference: He-like ground-state E1.
# ---------------------------------------------------------------------------

class TestHeLikeGroundTransitionGaugeRatio:
    """1s²→1s2p E1 in He-like ions through the MCDHF eigenstates.

    Per Gu 2008 §3.2, the gauge ratio for ground-state E1 transitions in
    a well-converged calculation should satisfy ``|A_v/A_l - 1| ≤ 0.10``.

    The velocity-gauge radial integral now uses the full Grant §8.4
    derivative-free reduction (T1 / HYP-190), giving exact gauge invariance
    for hydrogenic eigenstates.  For MCDHF states the orbitals are not exact
    eigenstates of a common local potential, so residual gauge discrepancy
    reflects the wavefunction quality.
    """

    # Was a strict xfail (2026-09-25): measured |A_v/A_l - 1| = 0.234 at
    # this test's parameters (0.389 with the protocol of
    # docs/archive/2026-09/CORRELATION_SD_RESEARCH_2026-09-20.md section 3.7), blamed on
    # the He 2p radial function diving to a bare-nucleus hydrogenic
    # solution.  Re-measured 2026-09-23 after the parent-core
    # spectator-row reform (docs/test_audit/PHYSICS_hlike_2p_spectator.md):
    # 0.0868, below the Gu 2008 §3.2 threshold of 0.10 — the 2p spectator
    # now sees its own parent core (1s1) instead of the capped reference,
    # which is the same defect, not a separate orbital problem.  The
    # threshold stays at the literature value; 0.0868 sits at 87 % of it.
    def test_he_like_resonance_e1_z2(self):
        from atomx import Atom, Basis

        he = Atom(Z=2, config=["1s2", "1s1 2p1"], basis=Basis(ns=60, ks=9))
        he.solve(max_iter=60, tol=1e-5)
        if not he.converged:
            pytest.skip("He-like Z=2 MCDHF did not converge")

        results = he.transitions(multipoles=["E1"], gauge="both")
        e1s = [t for t in results if t.A_length is not None
               and t.A_velocity is not None and t.A_length > 0]
        if not e1s:
            pytest.skip("No E1 transitions resolved for He-like Z=2")

        strongest = max(e1s, key=lambda t: t.A_length)
        ratio_dev = abs(strongest.A_velocity / strongest.A_length - 1.0)
        # HYP-1689: after the HYP-190 velocity-gauge fix this passed
        # deterministically at ratio_dev = 0.0833 (main @ 6172d1e).  The
        # 2026-09-23 parent-core spectator-row reform then moved it to
        # 0.0868 (see the note above the method): still below the Gu 2008
        # §3.2 ground-state E1 threshold of 0.10.  The gate is kept at the
        # literature value rather than tightened: 0.0868 sits at 87% of the
        # threshold, leaving too little headroom for cross-platform
        # (numba/BLAS) jitter at a tighter bound.
        assert ratio_dev < 0.10, (
            f"He-like Z=2: strongest E1 |A_v/A_l - 1| = {ratio_dev:.3f}"
        )

    def test_he_like_resonance_e1_z10(self):
        from atomx import Atom, Basis

        he = Atom(Z=10, config=["1s2", "1s1 2p1"], basis=Basis(ns=60, ks=9))
        he.solve(max_iter=60, tol=1e-5)
        if not he.converged:
            pytest.skip("He-like Z=10 MCDHF did not converge")

        results = he.transitions(multipoles=["E1"], gauge="both")
        e1s = [t for t in results if t.A_length is not None
               and t.A_velocity is not None and t.A_length > 0]
        if not e1s:
            pytest.skip("No E1 transitions resolved for He-like Z=10")

        strongest = max(e1s, key=lambda t: t.A_length)
        ratio_dev = abs(strongest.A_velocity / strongest.A_length - 1.0)
        # The module docstring's threshold: Gu 2008 §3.2 ground-state E1,
        # |A_v/A_l - 1| <= 0.10.  Keeping 0.20 here (the old value) let a
        # 2x regression through silently.  Measured at this test's
        # parameters: 0.0369 (2.7x margin), so the literature threshold is
        # reachable for Z=10 even though the Z=2 case above is not.
        assert ratio_dev < 0.10, (
            f"He-like Z=10: strongest E1 |A_v/A_l - 1| = {ratio_dev:.4f} "
            f"exceeds the Gu 2008 ground-state E1 threshold 0.10"
        )


# ---------------------------------------------------------------------------
# 4. Negative test: an *unconverged* atom must still produce a sane API.
# ---------------------------------------------------------------------------

def test_gauge_both_unaffected_by_no_kappa_argument():
    """Length-gauge default still returns sane ``A_length`` / ``A_velocity``."""
    # Even with the default (length) gauge, the calculator should not
    # populate the velocity field — that's the "don't pay for what you
    # don't ask for" contract.
    from atomx import Atom

    h = Atom(Z=1, config="1s1")  # default engine (shooting)
    h.solve(max_iter=30, tol=1e-4)
    if not h.converged:
        pytest.skip("H did not converge")

    transitions = h.transitions(multipoles=["E1"], gauge="length")
    for tr in transitions:
        # In length-only mode, A_length is populated but A_velocity is None
        # (no extra integral is performed).
        assert tr.A_velocity is None
        assert tr.A_length is not None
        assert tr.gauge_ratio is None
