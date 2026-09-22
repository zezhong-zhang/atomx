"""M1 tests: Dirac solver validated against hydrogenic analytical solutions.

Reference values are the exact point-Coulomb Dirac (Sommerfeld) eigenvalue,
**written out in this file** by :func:`_dirac_coulomb` — not read back from
``atomx.physics.dirac.hydrogenic_dirac_energy``.  The seven tests that used
to compare the B-spline solver against that function (a package value) now
compare it against the closed form; a scale error in the package function
that preserved ``n=1, kappa=-1, Z=1`` would previously have gone unnoticed.
"""

import math

import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.dirac import hydrogenic_dirac_energy

#: CODATA 2018 fine-structure constant, quoted here so the closed form below
#: depends on no atomx constant.  (atomx's own ``constants.C`` differs from
#: ``1/_ALPHA`` by 6.8e-10 relative, which propagates to <=2.4e-10 relative
#: in E at Z=92 — hence the ``rel=1e-9`` used when comparing the package
#: function to this closed form.)
_ALPHA = 1.0 / 137.035999084
_C = 1.0 / _ALPHA          # speed of light in Hartree a.u.


def _dirac_coulomb(n: int, kappa: int, Z: float) -> float:
    r"""Exact point-Coulomb Dirac eigenvalue (Sommerfeld), in Hartree.

    Derivation.  For a point nucleus the Dirac equation separates with

        gamma = sqrt(kappa^2 - (Z alpha)^2),          kappa = +/-(j + 1/2)
        N     = sqrt((n - |kappa| + gamma)^2 + (Z alpha)^2)

    and the bound-state energy is

        E(n, kappa) = m c^2 ( sqrt(1 - (Z alpha / N)^2) - 1 ).

    Two exact properties of this expression (both gated in
    ``tests/test_first_principles_invariants.py``, not duplicated here):

    * it depends on ``|kappa|`` only — ``E(n, kappa) == E(n, -kappa)``
      exactly, which is the Dirac degeneracy of 2s1/2 with 2p1/2;
    * for ``n = 1`` one has ``|kappa| = gamma = 1``, ``N = 1``, and the
      general formula collapses to ``E = m c^2 (sqrt(1 - (Z alpha)^2) - 1)``.
    """
    gamma = math.sqrt(kappa * kappa - (Z * _ALPHA) ** 2)
    N = math.sqrt((n - abs(kappa) + gamma) ** 2 + (Z * _ALPHA) ** 2)
    return _C * _C * (math.sqrt(1.0 - (Z * _ALPHA / N) ** 2) - 1.0)


class TestHydrogenicAnalytical:
    """Test closed-form Dirac energies.

    The closed form is the test's own; the comparison target is the package
    function ``hydrogenic_dirac_energy``, i.e. this class gates atomx.
    """

    def test_hydrogen_1s(self):
        """H 1s1/2: E = -0.50000665 Hartree (Dirac, not -0.5 Schrodinger).

        The textbook value is rounded to 9 digits; the closed form gives
        -0.5000066566, i.e. 1.3e-8 away, so rel=1e-7 (not 1e-6).
        """
        E = hydrogenic_dirac_energy(n=1, kappa=-1, Z=1)
        assert pytest.approx(-0.50000665, rel=1e-7) == E
        assert pytest.approx(_dirac_coulomb(n=1, kappa=-1, Z=1), rel=1e-9) == E

    def test_hydrogen_2s(self):
        E = hydrogenic_dirac_energy(n=2, kappa=-1, Z=1)
        assert pytest.approx(_dirac_coulomb(n=2, kappa=-1, Z=1), rel=1e-9) == E
        # Measured closed form: -0.1250020802 Ha.
        assert pytest.approx(-0.1250020802, rel=1e-9) == E
        assert hydrogenic_dirac_energy(n=1, kappa=-1, Z=1) < E

    def test_hydrogen_2p_fine_structure(self):
        """2p3/2 is less bound than 2p1/2 (fine structure).

        Closed forms: -0.1250004160 (2p3/2) vs -0.1250020802 (2p1/2); the
        splitting is (Z alpha)^4/32 = 5.3e-9 Ha.
        """
        E_2p12 = hydrogenic_dirac_energy(n=2, kappa=1, Z=1)
        E_2p32 = hydrogenic_dirac_energy(n=2, kappa=-2, Z=1)
        assert E_2p12 == pytest.approx(
            _dirac_coulomb(n=2, kappa=1, Z=1), rel=1e-9)
        assert E_2p32 == pytest.approx(
            _dirac_coulomb(n=2, kappa=-2, Z=1), rel=1e-9)
        assert E_2p32 == pytest.approx(-0.1250004160, rel=1e-9)
        assert E_2p32 > E_2p12

    def test_high_Z_uranium(self):
        """U91+ (Z=92) 1s1/2: deeply bound relativistic atom.

        Closed form (n=1 collapse): m c^2 (sqrt(1-(Z alpha)^2) - 1)
        = -4861.1979 Ha.
        """
        exact = _dirac_coulomb(n=1, kappa=-1, Z=92)
        assert exact == pytest.approx(-4861.1979, abs=2.0e-5)
        E = hydrogenic_dirac_energy(n=1, kappa=-1, Z=92)
        assert pytest.approx(exact, rel=1e-9) == E
        assert E < -4800

    def test_invalid_quantum_numbers(self):
        with pytest.raises(ValueError):
            hydrogenic_dirac_energy(n=1, kappa=0, Z=1)
        with pytest.raises(ValueError):
            hydrogenic_dirac_energy(n=0, kappa=-1, Z=1)
        with pytest.raises(ValueError):
            hydrogenic_dirac_energy(n=1, kappa=-2, Z=1)


class TestDiracSolver:
    """Test B-spline Dirac solver against analytical hydrogenic energies."""

    def _solve_hydrogenic(self, Z, kappa, ns=40, ks=9, r_min=None, r_max=50.0):
        """Helper: solve Dirac equation for hydrogenic ion."""
        if r_min is None:
            r_min = 0.1 / max(Z, 1)  # scale with Z to handle singularity
        t = make_knots(ns, ks, r_min, r_max)
        data = prepare_bspline_data(t, ns, ks)
        H, S = build_dirac_matrices(ns, kappa, Z, data)
        return solve_dirac(H, S, ns, kappa)

    def test_hydrogen_1s_bspline(self):
        """H 1s1/2 via B-spline solver matches the closed form."""
        result = self._solve_hydrogenic(Z=1, kappa=-1)
        E_closed = _dirac_coulomb(n=1, kappa=-1, Z=1)
        assert result.n_bound >= 1, "No bound states found"
        assert result.bound_energies[0] == pytest.approx(E_closed, rel=1e-6), \
            f"1s: num={result.bound_energies[0]:.8f}, closed={E_closed:.8f}"

    def test_hydrogen_2s_bspline(self):
        """H 2s1/2 via B-spline; closed form -0.1250020802 Ha."""
        result = self._solve_hydrogenic(Z=1, kappa=-1)
        E_closed = _dirac_coulomb(n=2, kappa=-1, Z=1)
        assert result.n_bound >= 2
        assert result.bound_energies[1] == pytest.approx(E_closed, rel=1e-4)

    def test_hydrogen_2p32_bspline(self):
        """H 2p3/2 (kappa=-2) via B-spline; closed form -0.1250004160 Ha."""
        result = self._solve_hydrogenic(Z=1, kappa=-2)
        E_closed = _dirac_coulomb(n=2, kappa=-2, Z=1)
        assert result.n_bound >= 1
        assert result.bound_energies[0] == pytest.approx(E_closed, rel=1e-4)

    def test_he_plus_1s(self):
        """He+ (Z=2) 1s1/2; closed form -2.0001065141 Ha."""
        result = self._solve_hydrogenic(Z=2, kappa=-1, r_max=30.0)
        E_closed = _dirac_coulomb(n=1, kappa=-1, Z=2)
        assert result.n_bound >= 1
        assert result.bound_energies[0] == pytest.approx(E_closed, rel=1e-5)

    def test_fe26_1s(self):
        """Fe25+ (Z=26) 1s1/2 — moderately relativistic; closed form
        -341.0978372 Ha."""
        result = self._solve_hydrogenic(Z=26, kappa=-1, ns=50, r_max=10.0)
        E_closed = _dirac_coulomb(n=1, kappa=-1, Z=26)
        assert result.n_bound >= 1
        assert result.bound_energies[0] == pytest.approx(E_closed, rel=1e-4), \
            f"Fe25+ 1s: num={result.bound_energies[0]:.4f}, closed={E_closed:.4f}"

    def test_fine_structure_splitting(self):
        """2p1/2 and 2p3/2 differ by fine-structure splitting.

        Note: kappa=+1 channel has a spurious state near -0.5 due to lack of
        kinetic balance (a known issue in B-spline Dirac without KB).
        The physical 2p1/2 state is at index [1], not [0].

        Closed forms: 2p1/2 -0.1250020802, 2p3/2 -0.1250004160 Ha.
        """
        result_p12 = self._solve_hydrogenic(Z=1, kappa=1)
        result_p32 = self._solve_hydrogenic(Z=1, kappa=-2)
        # kappa=+1: index 0 is spurious, index 1 is physical 2p1/2
        E_2p12 = result_p12.bound_energies[1]
        E_2p32 = result_p32.bound_energies[0]  # kappa=-2: index 0 is 2p3/2
        assert E_2p12 == pytest.approx(
            _dirac_coulomb(n=2, kappa=1, Z=1), rel=1e-3)
        assert E_2p32 == pytest.approx(
            _dirac_coulomb(n=2, kappa=-2, Z=1), rel=1e-3)
        # Fine structure: 2p3/2 is less bound (higher energy)
        assert E_2p32 > E_2p12

    def test_dirac_degeneracy(self):
        """In hydrogenic atoms, |kappa| determines energy (Dirac degeneracy).

        2s1/2 (kappa=-1, n=2) and 2p1/2 (kappa=+1, n=2) should be degenerate.
        """
        result_s12 = self._solve_hydrogenic(Z=1, kappa=-1)
        result_p12 = self._solve_hydrogenic(Z=1, kappa=1)
        E_2s12 = result_s12.bound_energies[1]  # 2s1/2
        E_2p12 = result_p12.bound_energies[1]  # 2p1/2 (skip spurious [0])
        assert E_2s12 == pytest.approx(E_2p12, rel=1e-4)
        # ...and both sit on the same closed form, which is exactly even in
        # kappa: _dirac_coulomb(2, +1, 1) == _dirac_coulomb(2, -1, 1).
        closed = _dirac_coulomb(n=2, kappa=-1, Z=1)
        assert _dirac_coulomb(n=2, kappa=1, Z=1) == closed
        assert E_2s12 == pytest.approx(closed, rel=1e-4)


class TestKineticBalance:
    """Test Restricted Kinetic Balance (RKB): ksp=ks-1 for P, ksq=ks for Q."""

    def _solve_kb(self, Z, kappa, ns=40, ks=9, r_min=None, r_max=50.0):
        if r_min is None:
            r_min = 0.1 / max(Z, 1)
        t = make_knots(ns, ks, r_min, r_max)
        data = prepare_bspline_data(t, ns, ks, kinetic_balance=True)
        ns_full = data["ns"]
        H, S = build_dirac_matrices(ns_full, kappa, Z, data)
        return solve_dirac(H, S, ns_full, kappa,
                           nsp=data["nsp"], nsq=data["nsq"])

    def test_h_1s_accuracy(self):
        """H 1s with RKB should match the closed form to 6+ digits."""
        result = self._solve_kb(Z=1, kappa=-1)
        E_closed = _dirac_coulomb(n=1, kappa=-1, Z=1)
        assert result.bound_energies[0] == pytest.approx(E_closed, rel=1e-6)

    def test_no_spurious_kappa_plus(self):
        """kappa=+1 with RKB: NO spurious state at -0.5."""
        result = self._solve_kb(Z=1, kappa=1)
        # First bound state should be 2p1/2 (closed form -0.1250020802),
        # NOT -0.5
        assert result.bound_energies[0] == pytest.approx(
            _dirac_coulomb(n=2, kappa=1, Z=1), rel=1e-3)

    def test_fine_structure_direct(self):
        """2p1/2 and 2p3/2 fine structure — no spurious state correction needed."""
        r_p12 = self._solve_kb(Z=1, kappa=1)
        r_p32 = self._solve_kb(Z=1, kappa=-2)
        # With RKB: kappa=+1 first state IS the physical 2p1/2 (index 0, not 1)
        E_2p12 = r_p12.bound_energies[0]
        E_2p32 = r_p32.bound_energies[0]
        assert E_2p32 > E_2p12  # 2p3/2 less bound
        assert E_2p12 == pytest.approx(
            _dirac_coulomb(n=2, kappa=1, Z=1), rel=1e-3)
        assert E_2p32 == pytest.approx(
            _dirac_coulomb(n=2, kappa=-2, Z=1), rel=1e-3)

    def test_he_plus_1s(self):
        """He+ 1s with RKB; closed form -2.0001065141 Ha."""
        result = self._solve_kb(Z=2, kappa=-1, r_max=30.0)
        E_closed = _dirac_coulomb(n=1, kappa=-1, Z=2)
        assert result.bound_energies[0] == pytest.approx(E_closed, rel=1e-5)
