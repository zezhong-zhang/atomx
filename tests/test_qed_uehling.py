"""Uehling vacuum-polarisation kernel: external-anchored regression tests.

These exist because the shipped kernel was ``K_1(2cr)`` instead of the
Uehling integral ``I(2cr)`` — a different special function, larger at
*every* radius by a factor rising to 160 at small ``x``.  Every gate here
compares against a value that does **not** come from this package:

* ``I(x)`` against ``scipy.integrate.quad`` of the defining integral;
* the closed-form small-``x`` structure (logarithmic divergence);
* the H 1s vacuum-polarisation shift against the published
  ``Zα`` expansion ``F_Ueh = -4/15 + (5π/48)Zα - (2/15)(Zα)²L + …``
  (Yerokhin & Shabaev, J. Phys. Chem. Ref. Data 44, 033103 (2015),
  Eqs. 20-21) — the ``-4/15`` limit in particular is violated by a
  factor 3.9 by the old kernel;
* the U^91+ 1s shift against an independent evaluation of the same
  integral.

See ``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding C1.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.constants import ALPHA
from atomx.corrections.qed import (
    _uehling_I_exact,
    _uehling_kernel,
    compute_vp_energy,
    uehling_potential,
)

HARTREE_EV = 27.211386245988
HARTREE_MHZ = 6.579683920502e9

#: H 1s vacuum polarisation, dominant (leading-order) term
#: ``mc²(α/π)(Zα)⁴(-4/15)``.  The full value including the (Zα) and
#: (Zα)² terms is ≈ -215.1 MHz; the old K_1 kernel gave -844.4 MHz.
_H_1S_VP_MHZ = -215.1


class TestKernelAgainstQuadrature:
    """``I(x)`` must equal its defining integral, not ``K_1(x)``."""

    @pytest.mark.parametrize(
        "x, expected",
        [
            (0.01, 3.899512e00),
            (0.5, 4.843297e-01),
            (1.0, 1.779336e-01),
            (5.0, 6.902014e-04),
            (20.0, 3.693416e-11),
        ],
    )
    def test_tabulated_values(self, x, expected):
        assert _uehling_kernel(x) == pytest.approx(expected, rel=1e-6)

    def test_spline_interpolation_is_machine_accurate(self):
        """Kernel table vs ``_uehling_I_exact`` — an INTERPOLATION check.

        C-degenerate by construction: ``_uehling_kernel`` is a
        ``CubicSpline`` over 900 nodes of ``_uehling_I_exact``
        (``qed.py:77-98``), so comparing the two measures interpolation
        error only (exactly 0 at the nodes) and no longer validates the
        kernel.  The physics gate is :meth:`test_tabulated_values`, whose
        five literals are the defining integral
        ``∫₁^∞ e^{−xt}(1 + 1/2t²)√(t²−1)/t² dt`` (independently
        reproduced by ``scipy.integrate.quad`` to ≤ 1.2e-7), and
        :meth:`test_kernel_is_not_bessel_k1`.
        """
        xs = np.logspace(-9, np.log10(80.0), 60)
        worst = max(
            abs(_uehling_kernel(x) - _uehling_I_exact(x)) / _uehling_I_exact(x)
            for x in xs
        )
        assert worst < 1e-8, f"kernel table deviates from quad by {worst:.2e}"

    def test_kernel_is_not_bessel_k1(self):
        """Regression guard: K_1 must not reappear as the kernel."""
        from scipy.special import k1

        # At x = 1 the two differ by 3.4x; at x = 0.01 by 26x.
        assert _uehling_kernel(1.0) == pytest.approx(0.1779336, rel=1e-5)
        assert _uehling_kernel(1.0) < 0.5 * k1(1.0)
        assert _uehling_kernel(0.01) < 0.05 * k1(0.01)

    def test_small_x_is_logarithmic(self):
        """``I ~ log(1/x)`` for x -> 0 (K_1 would diverge as 1/x)."""
        small = [_uehling_kernel(x) for x in (1e-6, 1e-7, 1e-8)]
        # One decade down in x raises I by exactly ln(10).
        deltas = np.diff(small)
        assert all(abs(d - np.log(10.0)) < 1e-5 for d in deltas), deltas

    def test_monotone_decreasing(self):
        xs = np.logspace(-6, 2, 400)
        vals = np.asarray([_uehling_kernel(x) for x in xs])
        assert np.all(np.diff(vals) < 0.0)


class TestPotential:
    def test_attractive_and_short_range(self):
        r = np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
        V = uehling_potential(r, Z=10)
        assert np.all(V[:-1] < 0.0), "Uehling potential must be attractive"
        assert abs(V[-1]) < 1e-30, "potential must die beyond the Compton scale"

    def test_scalar_and_array_agree(self):
        for r in (1e-4, 1e-3, 0.01, 0.1):
            assert uehling_potential(r, 10) == pytest.approx(
                float(uehling_potential(np.array([r]), 10)[0]), rel=1e-14
            )

    def test_zero_radius_is_zero(self):
        assert uehling_potential(0.0, 10) == 0.0
        assert uehling_potential(np.array([0.0, 1e-3]), 10)[0] == 0.0

    def test_scales_linearly_in_Z(self):
        r = np.array([1e-3, 1e-2])
        ratio = uehling_potential(r, 20) / uehling_potential(r, 10)
        assert np.allclose(ratio, 2.0, rtol=1e-14)


class TestPhysicalAnchors:
    """End-to-end shifts against published values."""

    def _hydrogenlike_1s(self, Z):
        from atomx.numerics.analytic.hydrogenic_dirac import (
            hydrogenic_dirac_wavefunction,
        )

        P, Q, bsp_data, _E = hydrogenic_dirac_wavefunction(
            n=1, kappa=-1, Z=Z, ns=60, ks=9
        )
        return [(P, Q, 1.0, -1, "1s1/2")], bsp_data

    def test_hydrogen_1s_vp_matches_published_expansion(self):
        """H 1s VP: the -4/15 limit plus O(Zα) corrections -> -215.1 MHz.

        Published expansion (Yerokhin & Shabaev 2015, Eqs. 20-21):
        ``F_Ueh,pnt = -4/15 + (5π/48)Zα - (2/15)(Zα)²L + …``.  The old
        K_1 kernel gave -844.4 MHz, i.e. F = -1.039, which violates the
        exact -4/15 = -0.2667 limit by 3.9x.
        """
        orbitals, bsp_data = self._hydrogenlike_1s(1)
        E = compute_vp_energy(orbitals, bsp_data, Z=1)
        assert pytest.approx(_H_1S_VP_MHZ, rel=2e-3) == E * HARTREE_MHZ, (
            f"H 1s E_VP = {E * HARTREE_MHZ:.1f} MHz, expected ≈ -215.1 MHz"
        )
        # The leading-order coefficient must not be violated: the exact
        # -4/15 term gives -217.0 MHz for H 1s, and the O(Zα) corrections
        # move it by <1 %.  The old K_1 kernel sat at -844.4 MHz, i.e.
        # 3.9x the leading term — a 290 % violation.
        F_leading = -4.0 / 15.0
        E_leading_mhz = ALPHA**4 / (np.pi * ALPHA) * F_leading * HARTREE_MHZ
        assert pytest.approx(-217.0, rel=1e-3) == E_leading_mhz
        assert pytest.approx(E_leading_mhz * 0.9912, rel=2e-3) == E * HARTREE_MHZ

    def test_uranium_1s_vp(self):
        """U^91+ 1s VP ≈ -97.9 eV (independent evaluation of the same integral).

        The self-energy for the same orbital is +355 eV; the old kernel
        gave -517.5 eV here, which alone exceeded the measured total 1s
        Lamb shift of 460.2 ± 4.6 eV once combined.
        """
        orbitals, bsp_data = self._hydrogenlike_1s(92)
        E = compute_vp_energy(orbitals, bsp_data, Z=92)
        assert pytest.approx(-97.9, rel=5e-3) == E * HARTREE_EV

    @pytest.mark.snapshot
    def test_vp_monotone_in_Z(self):
        """|E_VP| must grow with Z (it scales as Z⁴ at fixed shape).

        The monotonicity is first-principles: the Uehling potential is
        ∝ Z (exact, ``qed.py`` prefactor) and the 1s density shrinks as
        1/Z³, so each successive Z must increase |E_VP|.

        The ratio below is a *frozen* pin, not a first-principles bound:
        the ratio is ``92⁴·f(92α)/f(α)`` and at Zα = 0.67 the published
        two-term expansion ``F = -4/15 + (5π/48)Zα`` (used at :126) gives
        1.27e7, 8.6x below what the full hydrogenic integral gives — the
        asymptotic series is not converged there, so no closed form is
        available at the required accuracy.  Measured 1.099859e8 with
        the fixed ``I(2cr)`` kernel; the pre-fix ``K₁(2cr)`` kernel gives
        4.3e8, outside ``rel=0.02``.  The band this replaced was
        ``1e6 < ratio < 1e9``, which both kernels satisfied.
        """
        vals = []
        for Z in (1, 10, 26, 54, 92):
            orbitals, bsp_data = self._hydrogenlike_1s(Z)
            vals.append(abs(compute_vp_energy(orbitals, bsp_data, Z=Z)))
        assert np.all(np.diff(vals) > 0.0), vals
        assert vals[-1] / vals[0] == pytest.approx(1.0998587063393934e8,
                                                  rel=0.02)
