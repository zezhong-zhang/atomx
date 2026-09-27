"""Self-energy table: external-anchored regression tests.

The table in :mod:`atomx.data.mohr_se` previously agreed with the published
hydrogenic self-energy function only at ``Z = 1`` and then diverged with the
wrong ``Z``-dependence — by 2.5x at ``Z = 10``, 6.1x at ``Z = 36`` and 14.7x
at ``Z = 92``.  Because the default ``qed_backend="auto"`` switched tables at
``Z = 37``, that also produced a 6.3x *discontinuity* in a physical
observable.

Every gate here compares against something that is not this package:

* FAC 1.1.x ``faclib/coulomb.c`` static arrays ``_qed_se0``/``_qed_se1``/
  ``_sd`` (the values are transcribed programmatically, not by hand);
* the published ``Zα`` expansion of the self-energy remainder
  ``F = L·A41 + A40 + (Zα)A50 + (Zα)²[L²A62 + L·A61 + G]`` with the
  coefficients of Yerokhin & Shabaev, J. Phys. Chem. Ref. Data 44, 033103
  (2015), Eqs. (7)-(19);
* the measured 1s Lamb shift of U^91+, 460.2 ± 4.6 eV
  (Gumberidze et al., PRL 94, 223001 (2005)) — the *self-energy alone*
  cannot exceed it.

See ``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` findings C2 and M1.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.constants import ALPHA
from atomx.corrections.qedmod.indelicato import _interpolate_indelicato_f
from atomx.corrections.qedmod.mohr import MohrBackend, _interpolate_se_f
from atomx.data.mohr_se import SD_ROWS, SD_TABLE, SD_Z_GRID, klamaq_f
from atomx.data.nist_bethe_log import BETHE_LOG_NIST

HARTREE_EV = 27.211386245988


def _f_expansion(Z, G=-30.0):
    """Published Zα expansion of ``F_SE,pnt`` (Yerokhin & Shabaev Eqs. 7-19).

    ``G`` is the unknown higher-order remainder; a *single* constant
    ``G ≈ -30`` reproduces the exact Z=1 and Z=2 values, which is the
    point of the check.
    """
    za = ALPHA * Z
    L = np.log(1.0 / za**2)
    a41, a40, a50, a62, a61 = (
        4.0 / 3.0,
        -(4.0 / 3.0) * 2.984128556 + 10.0 / 9.0,
        (139.0 / 32.0 - 2.0 * np.log(2.0)) * np.pi,
        -1.0,
        5.419373685,
    )
    return L * a41 + a40 + za * a50 + za**2 * (L * L * a62 + L * a61 + G)


def _se_1s_eV(Z, F):
    return HARTREE_EV * (ALPHA * Z) ** 4 / (np.pi * ALPHA) * F


class TestTableAgainstPublishedExpansion:
    def test_z1_and_z2_match_with_one_constant_remainder(self):
        """The expansion nails Z=1 and Z=2 with the same ``G``.

        The old table would have required ``G ≈ +9705`` at Z=2.
        """
        assert _interpolate_se_f(1, 1, -1) == pytest.approx(_f_expansion(1), rel=1e-3)
        assert _interpolate_se_f(2, 1, -1) == pytest.approx(_f_expansion(2), rel=1e-3)

    @pytest.mark.parametrize("Z, tol", [(5, 2e-2), (10, 2e-2)])
    def test_low_z_trend_follows_the_expansion(self, Z, tol):
        """Trend at low Z with ``G`` frozen at its Z=1 value.

        The expansion is only asymptotic in ``Zα``: freezing ``G`` costs
        <1 % at Z=10, 3.9 % at Z=18 and 11 % at Z=26, so the gate covers
        Z <= 10.  The discriminating power is in ``test_z1_and_z2...``:
        the old table would have needed ``G = +9705`` at Z=2.
        """
        assert _interpolate_se_f(Z, 1, -1) == pytest.approx(
            _f_expansion(Z), rel=tol
        )

    def test_table_values_are_fac(self):
        """Spot values against FAC ``_qed_se0 + _qed_se1``."""
        for Z, expected in [(1, 10.31679), (2, 8.52833), (10, 4.65413),
                            (54, 1.78189), (92, 1.47251)]:
            assert _interpolate_se_f(Z, 1, -1) == pytest.approx(expected, rel=5e-5)

    def test_falls_with_z_over_the_physical_range(self):
        """``F_SE`` for the 1s falls by ~6x from Z=1 to Z=54.

        It has a shallow minimum near Z~87 and rises again at the very
        top of the grid (the ``(Zα)²`` terms take over), which is a
        feature of the published function, not of this table — so the
        gate covers the range where the fall is the physics.
        """
        vals = [_interpolate_se_f(Z, 1, -1) for Z in range(1, 55)]
        assert all(b < a for a, b in zip(vals[:-1], vals[1:], strict=True)), (
            "F(1s) must fall with Z up to Z=54"
        )
        assert vals[0] / vals[-1] > 5.0
        assert _interpolate_se_f(92, 1, -1) < 2.0


class TestNoDiscontinuity:
    def test_auto_policy_is_continuous_at_z37(self):
        """The historical ``auto`` switch at Z=37 must not move the answer.

        Before the tables were unified, the 1s self-energy fell from
        79.15 eV to 14.13 eV between Z=36 and Z=37.
        """
        f36 = _interpolate_se_f(36, 1, -1)
        f37 = _interpolate_se_f(37, 1, -1)
        assert abs(f37 - f36) / f36 < 0.02, (f36, f37)
        # Energy-level continuity: the per-electron shift must change by
        # only the smooth O(Z^3) trend, not by a factor.
        e36 = abs(_se_1s_eV(36, f36))
        e37 = abs(_se_1s_eV(37, f37))
        assert 1.02 < e37 / e36 < 1.20, (e36, e37)

    def test_backends_agree_exactly(self):
        for Z in (1, 10, 36, 37, 54, 92, 120):
            for (n, k) in ((1, -1), (2, -1), (2, 1), (2, -2)):
                assert _interpolate_indelicato_f(Z, n, k) == _interpolate_se_f(
                    Z, n, k
                )

    def test_interpolation_is_smooth_in_z(self):
        """No jumps between adjacent integer Z anywhere on the grid."""
        prev = None
        for Z in range(1, 121):
            f = _interpolate_se_f(Z, 1, -1)
            if prev is not None:
                assert abs(f - prev) / prev < 0.25, Z
            prev = f


class TestCoverageAndLdependence:
    def test_d_and_f_orbitals_differ(self):
        """``l``/``j`` dependence is no longer discarded.

        The old ``F(1s)/n³`` fallback gave a ``4f5/2`` and a ``4d3/2``
        orbital *identical* shifts.
        """
        f_3d52 = _interpolate_se_f(92, 3, -3)
        f_3d32 = _interpolate_se_f(92, 3, 2)
        f_4d32 = _interpolate_se_f(92, 4, -2)
        assert f_3d52 != f_4d32
        assert f_3d52 != f_3d32
        # FAC's 3d+ is positive and 3d- is negative across the table.
        assert f_3d52 > 0.0 > f_3d32

    def test_n5_rows_are_tabulated(self):
        """n=5 is covered by FAC's ``_sd`` table, not the analytic fallback."""
        for key in ((5, -1), (5, 1), (5, -2)):
            assert key in SD_ROWS
            i = SD_ROWS.index(key)
            for z in SD_Z_GRID:
                assert np.isfinite(SD_TABLE[i][SD_Z_GRID.index(z)])

    def test_high_n_uses_bethe_log_with_l_dependence(self):
        """``n > 5`` falls back to FAC's Klamaq, which depends on ``l``."""
        assert klamaq_f(6, -1) != klamaq_f(6, -2)
        assert klamaq_f(6, -1) != klamaq_f(7, -1)
        # Klamaq is O(1), unlike the old F(1s)/n^3 (= 0.03..10).
        assert abs(klamaq_f(6, -1)) < 20.0

    def test_nist_rydberg_values_cover_high_n(self):
        assert len(BETHE_LOG_NIST) == 200
        assert sum(map(len, BETHE_LOG_NIST.values())) == 1572
        assert BETHE_LOG_NIST[9][0] == pytest.approx(2.72875117)
        assert BETHE_LOG_NIST[59][1] == pytest.approx(-0.0489983824)
        assert klamaq_f(9, -1) != klamaq_f(10, -1)

    def test_multidigit_orbital_label_uses_full_principal_number(self):
        result = MohrBackend().compute(
            [(np.zeros(1), np.zeros(1), 1.0, -1, "10s1/2")], None, 1)
        expected = ALPHA**3 / (np.pi * 10**3) * klamaq_f(10, -1)
        assert result.per_orbital["10s1/2"] == pytest.approx(expected)


class TestPhysicalAnchors:
    def test_uranium_self_energy_below_measured_lamb_shift(self):
        """The 1s self-energy of U^91+ cannot exceed the measured Lamb shift.

        Measured total 1s Lamb shift: 460.2 ± 4.6 eV.  The self-energy is
        its dominant part, so it must be a few hundred eV — not the
        5218 eV the old table produced (11.3x the measurement).
        """
        e_se = _se_1s_eV(92, _interpolate_se_f(92, 1, -1))
        assert e_se == pytest.approx(355.1, rel=5e-3)
        assert 0.5 * 460.2 < e_se < 460.2

    def test_hydrogen_1s_self_energy(self):
        """H 1s self-energy = 8396.5 MHz (dominant term of the 1S Lamb shift).

        The measured 1S Lamb shift is 8172.87 MHz = SE - 215.1 (VP) -
        6.6 (higher order).
        """
        mhz_per_ev = 2.417989242e14 / 1e6
        e_se = _se_1s_eV(1, _interpolate_se_f(1, 1, -1))
        assert e_se * mhz_per_ev == pytest.approx(8396.5, rel=2e-3)
        assert e_se * mhz_per_ev - 215.1 - 6.6 == pytest.approx(8172.87, rel=3e-3)
