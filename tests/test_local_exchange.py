"""Unit tests for the local-exchange functionals.

All tests are pure-numpy: they construct a synthetic density, compute
V_x, and check against the closed-form value. No SCF.
"""

import numpy as np
import pytest

from atomx.physics.local_exchange import (
    fac_gu_exchange,
    get_hxs,
    kohn_sham_exchange,
    slater_exchange,
)


class TestSlater:
    def test_reproduces_alpha_two_thirds(self):
        r = np.linspace(0.01, 10.0, 512)
        rho = np.exp(-r ** 2)
        V = slater_exchange(rho, r, alpha=2.0 / 3.0)
        expected = -(2.0 / 3.0) * (3.0 * rho / np.pi) ** (1.0 / 3.0)
        assert np.allclose(V, expected, rtol=1e-12)

    def test_zero_density_gives_zero_potential(self):
        r = np.linspace(0.01, 10.0, 512)
        V = slater_exchange(np.zeros_like(r), r, alpha=2.0 / 3.0)
        assert np.all(V == 0.0)


class TestKohnSham:
    def test_reproduces_alpha_one(self):
        r = np.linspace(0.01, 10.0, 512)
        rho = np.exp(-r ** 2)
        V = kohn_sham_exchange(rho, r)
        expected = -1.0 * (3.0 * rho / np.pi) ** (1.0 / 3.0)
        assert np.allclose(V, expected, rtol=1e-12)


class TestGetHXS:
    """get_hxs must be byte-equivalent to fac/radial.c:654."""

    def test_slater_lda_value_from_sentinel(self):
        # hxs <= -5 sentinel → (3/(4π²))^(1/3) ≈ 0.4236
        expected = (3.0 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
        assert get_hxs(Z=26, N=10, ihx=0.0, hxs=-5.0) == pytest.approx(
            expected, rel=1e-12
        )

    def test_default_fac_ihx_equals_minus_one_uses_k1_branch(self):
        # FAC defaults: hxs=1, ihx=-1 → |ihx|=1 ≥ 0.999 → k=1 only
        # From calibration.csv row (Z=26, N=10): α = 0.50445120
        a = get_hxs(Z=26, N=10, ihx=-1.0, hxs=1.0)
        assert a == pytest.approx(0.50445120, rel=1e-7)

    def test_ne_like_neon_matches_calibration(self):
        a = get_hxs(Z=10, N=10, ihx=-1.0, hxs=1.0)
        assert a == pytest.approx(0.51465105, rel=1e-7)

    def test_pb_ar_like_matches_calibration(self):
        a = get_hxs(Z=82, N=18, ihx=-1.0, hxs=1.0)
        assert a == pytest.approx(0.37229885, rel=1e-7)

    def test_ihx_zero_hits_k0_branch(self):
        # With hs < 0.001 → k=0 only
        # Hand-check: _hx0[0][8] + _hx1[0][8]·100 for Ne (n=10, idx=8)
        # = 0.4581007 + (-2.01426809e-05)·100 = 0.45608643
        a = get_hxs(Z=10, N=10, ihx=0.0, hxs=1.0)
        assert a == pytest.approx(0.45608643, rel=1e-6)

    def test_clips_below_0p1(self):
        # Crank Z² up until table result would go below 0.1
        # Result must clip at 0.1 (not go negative)
        a = get_hxs(Z=200, N=10, ihx=-1.0, hxs=1.0)
        assert a == pytest.approx(0.1, abs=1e-12)

    def test_clips_above_0p9(self):
        # Plain slater-override branch: hxs > 0, != 1 → r0 = hxs + ihx·Z²
        # Pick ihx = 0, hxs = 2.0: r0 = 2 → clip to 0.9
        a = get_hxs(Z=10, N=10, ihx=0.0, hxs=2.0)
        assert a == pytest.approx(0.9, abs=1e-12)

    def test_n_below_2_returns_half(self):
        a = get_hxs(Z=10, N=1, ihx=-1.0, hxs=1.0)
        assert a == 0.5

    def test_n_above_NHXN_uses_last_row(self):
        a_31 = get_hxs(Z=10, N=31, ihx=-1.0, hxs=1.0)
        a_50 = get_hxs(Z=10, N=50, ihx=-1.0, hxs=1.0)
        assert a_31 == a_50

    def test_intermediate_ihx_blends_the_two_branches(self):
        # hs = 0.5 → r1*0.5 + r0*0.5
        a_blend = get_hxs(Z=26, N=10, ihx=-0.5, hxs=1.0)
        a_k0 = get_hxs(Z=26, N=10, ihx=0.0, hxs=1.0)
        a_k1 = get_hxs(Z=26, N=10, ihx=-1.0, hxs=1.0)
        assert a_blend == pytest.approx(0.5 * (a_k0 + a_k1), rel=1e-10)


class TestFacGu:
    def test_hxs_zero_falls_through_to_slater(self):
        r = np.linspace(0.01, 10.0, 512)
        rho = np.exp(-r ** 2) / (4 * np.pi * r ** 2)
        V1 = fac_gu_exchange(
            rho, r, Z=10, N_total=10, ihx=-1.0, hxs=0.0,
            slater_alpha=2.0 / 3.0,
        )
        V2 = slater_exchange(rho, r, alpha=2.0 / 3.0)
        assert np.allclose(V1, V2, rtol=1e-12)

    def test_Z_dependence_is_monotonic(self):
        """α_xc decreases monotonically with Z² at fixed N (k=1 branch)."""
        r = np.linspace(0.01, 10.0, 256)
        rho = np.exp(-r ** 2) / (4 * np.pi * r ** 2)
        vals = []
        for Z in (10, 30, 54, 82):
            V = fac_gu_exchange(
                rho, r, Z=Z, N_total=10, ihx=-1.0, hxs=1.0,
            )
            vals.append(abs(V[100]))  # |V_x| scales with α
        # α_xc shrinks as Z grows → |V_x| shrinks
        assert all(vals[i] > vals[i + 1] for i in range(3))

    def test_returns_same_grid_shape(self):
        r = np.linspace(0.01, 10.0, 777)
        rho = np.exp(-r ** 2) / (4 * np.pi * r ** 2)
        V = fac_gu_exchange(rho, r, Z=10, N_total=10, ihx=-1.0, hxs=1.0)
        assert V.shape == r.shape
