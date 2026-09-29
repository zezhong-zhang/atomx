"""High-Z K-shell accuracy of the automatic B-spline grid.

The DBSR_HF-style knot sequence starts at ``hs = hi/Z`` with ``hi = 0.25``
and grows by ``1 + he`` per interval.  For a relativistic 1s the small-``r``
behaviour is ``r^gamma`` with ``gamma = sqrt(1 - (Z alpha)^2)``; at Z=92
``gamma = 0.741``, so the cusp below the first knot cannot be represented and
the error scales as ``hs^(2 gamma)``.  Measured against the exact
Dirac-Coulomb eigenvalue the default grid puts the hydrogen-like U^91+ 1s
**+97.7 eV** too high — the same order as the entire 1s self-energy (355 eV),
in the engine that carries every CI/MCDHF/transition/scattering calculation.

The grid shape was not reachable from the public API, so there was no way to
fix it.  ``Atom(grid_hi=..., grid_he=..., grid_hmax=...)`` now forwards to
:func:`atomx.numerics.bspline.knots.make_physics_knots`; ``hi=0.005,
he=0.10`` brings the same eigenvalue to **0.24 eV**.

See ``docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding C3.
"""

from __future__ import annotations

import math

import pytest

from atomx import Atom, Basis
from atomx.constants import C

HARTREE_EV = 27.211386245988


def _exact_dirac_coulomb(n, kappa, Z):
    gamma = math.sqrt(kappa * kappa - (Z / C) ** 2)
    N = math.sqrt((n - abs(kappa) + gamma) ** 2 + (Z / C) ** 2)
    return C * C * (math.sqrt(1.0 - (Z / C) ** 2 / N ** 2) - 1.0)


def _h_like_1s_eV(Z, **grid):
    atom = Atom(
                Z=Z,
                charge=Z - 1,
                config="1s1",
                engine="bspline",
                nuclear="point",
                corrections="none",
                basis=Basis(**grid),
           )
    atom.solve()
    exact = _exact_dirac_coulomb(1, -1, Z)
    return (atom.total_energy - exact) * HARTREE_EV


class TestDefaultGridErrorProfile:
    """Pin the *measured* error of the default grid against Z.

    The default is adequate at low Z and progressively useless at high Z:
    the K-shell error is 8e-5 eV at Z=10 but 1.6 eV at Z=54 and 98 eV at
    Z=92.  Recording the profile here means a change to the default knot
    sequence shows up in exactly one place, and that "the default is fine"
    cannot be assumed.
    """

    #: Z -> (measured 1s error in eV for the default physics grid).
    MEASURED = {
        1: 0.0,        # at the double-precision noise floor
        2: 6.08e-9,
        10: 8.045e-5,
        18: 2.5628e-3,
        26: 2.1606e-2,
        36: 0.14159,
        54: 1.6292,
        74: 14.640,
        92: 97.737,
    }

    @pytest.mark.snapshot
    @pytest.mark.parametrize("Z", sorted(MEASURED))
    def test_error_matches_the_recorded_profile(self, Z):
        """Upper bound only: the default grid must not get *worse*.

        The recorded values are atomx's own output (D-class: this asserts
        nothing external), so the test is marked ``snapshot``.  It is a
        one-sided bound ``err(Z) < MEASURED[Z] * 1.25`` — the two-sided 25 %
        band it replaces would also have accepted an *improvement*, and could
        not fail for any physics reason beyond the recorded value changing.
        """
        expected = self.MEASURED[Z]
        got = _h_like_1s_eV(Z)
        assert got < expected * 1.25 + 1e-9, (
            f"Z={Z}: default-grid 1s error is {got:.4g} eV, recorded "
            f"{expected:.4g} eV.  If the default grid changed on purpose, "
            f"re-measure the profile and update C3 of the review."
        )


class TestDefaultGridErrorIsBounded:
    """First-principles bound on the default grid (A-class).

    The reference is the exact Dirac–Coulomb eigenvalue written out in
    :func:`_exact_dirac_coulomb`, not an atomx value.  U^91+ 1s on the
    default grid is measured +97.737 eV too high (see the module docstring
    and finding C3); the bound below is the previously documented ceiling —
    it fails if the default grid ever degrades further, and it is the
    honest replacement for the deleted
    ``test_error_grows_strongly_with_z`` tautology (which asserted only that
    the test's own ``MEASURED`` dict was monotone and never called atomx).
    """

    def test_default_grid_u91_1s_error_stays_below_100_ev(self):
        err = _h_like_1s_eV(92)
        assert err < 100.0, (
            f"Z=92 default-grid 1s error {err:.3f} eV exceeds the documented "
            f"97.737 eV ceiling (exact Dirac eigenvalue reference)"
        )


class TestGridKnobs:
    def test_grid_hi_refines_the_k_shell(self):
        """``grid_hi`` is the knob the absence of which made C3 unfixable.

        Reference is the exact Dirac eigenvalue, so this is A-class.
        Measured: 97.737 eV -> 0.2416 eV (404x better); the bounds below are
        2x inside each measurement.
        """
        coarse = _h_like_1s_eV(92)
        fine = _h_like_1s_eV(92, grid_hi=0.005, grid_he=0.10)
        assert abs(fine) < 0.5, f"grid_hi=0.005 gives {fine:.3f} eV"
        assert abs(fine) < abs(coarse) / 200.0

    def test_convergence_is_monotone_in_grid_hi(self):
        errs = [abs(_h_like_1s_eV(92, grid_hi=h))
                for h in (0.25, 0.05, 0.02, 0.01)]
        assert all(b < a for a, b in zip(errs[:-1], errs[1:], strict=True)), errs

    def test_defaults_are_not_disturbed(self):
        """Passing None must reproduce the historical grid exactly."""
        base = _h_like_1s_eV(10)
        explicit = _h_like_1s_eV(10, grid_hi=None, grid_he=None, grid_hmax=None)
        assert base == explicit

    def test_grid_does_not_change_a_normal_atom(self):
        """A mid-Z closed shell is insensitive to the inner step at 1e-6."""
        kwargs = dict(Z=10, config="1s2 2s2 2p6", engine="bspline",
                      corrections="none")
        a = Atom(**kwargs)
        a.solve()
        b = Atom(**kwargs, basis=Basis(grid_hi=0.02, grid_he=0.10))
        b.solve()
        assert a.total_energy == pytest.approx(b.total_energy, rel=1e-6)
