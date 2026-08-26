"""Regression tests for Breit-Pauli hydrogenic correction factors."""

import pytest

from atomx.constants import ALPHA
from atomx.physics.breit_pauli import (
    darwin_shift,
    mass_velocity_shift,
    spin_orbit_shifts,
    spin_orbit_xi,
)


ALPHA_SQ = ALPHA * ALPHA


def test_spin_orbit_xi_hydrogen_2p_matches_cowan_factor():
    assert spin_orbit_xi(n=2, l=1, Z_eff=1.0) == pytest.approx(
        ALPHA_SQ / 48.0,
        rel=0.0,
        abs=1e-12,
    )


def test_hydrogen_2p_sommerfeld_splitting():
    xi = spin_orbit_xi(n=2, l=1, Z_eff=1.0)
    E_so_plus, E_so_minus = spin_orbit_shifts(l=1, xi=xi)

    assert E_so_plus - E_so_minus == pytest.approx(
        ALPHA_SQ / 32.0,
        rel=0.0,
        abs=1e-12,
    )


def test_hydrogen_2s12_2p12_breit_pauli_degeneracy():
    Z_eff = 1.0
    two_s12 = (
        mass_velocity_shift(n=2, l=0, Z_eff=Z_eff)
        + darwin_shift(n=2, l=0, Z_eff=Z_eff)
    )

    xi_2p = spin_orbit_xi(n=2, l=1, Z_eff=Z_eff)
    _, two_p12_so = spin_orbit_shifts(l=1, xi=xi_2p)
    two_p12 = (
        mass_velocity_shift(n=2, l=1, Z_eff=Z_eff)
        + darwin_shift(n=2, l=1, Z_eff=Z_eff)
        + two_p12_so
    )

    assert two_p12 - two_s12 == pytest.approx(0.0, rel=0.0, abs=1e-15)


def test_spin_orbit_xi_scales_as_z_four():
    xi_h = spin_orbit_xi(n=2, l=1, Z_eff=1.0)
    xi_ne = spin_orbit_xi(n=2, l=1, Z_eff=10.0)

    assert xi_ne == pytest.approx(xi_h * 10.0**4, rel=1e-14)


class TestBreitPauliSolve:
    """End-to-end regression tests for ``BreitPauli.solve()`` (HYP-1589).

    The NR base solver keys ``orbital_energies`` by jj subshell labels
    (``"1s1/2"``, ``"2p1/2"``); ``solve()`` must resolve base energies
    against those keys, never against synthesized NR labels (``"2p"``)
    with a silent zero fallback.
    """

    def test_hydrogen_2p1_sommerfeld_splitting_end_to_end(self):
        """H(Z=1) '2p1': no KeyError, two jj keys, splitting == α²/32."""
        from atomx.atom import Atom
        from atomx.methods.relativistic_hf import BreitPauli

        result = Atom(Z=1, config="2p1").solve(BreitPauli())

        assert set(result.orbital_energies) == {"2p1/2", "2p3/2"}
        splitting = (
            result.orbital_energies["2p3/2"]
            - result.orbital_energies["2p1/2"]
        )
        # Both levels share one NR base energy, so the splitting is the
        # exact analytic spin-orbit value α²Z⁴/32.
        assert splitting == pytest.approx(ALPHA_SQ / 32.0, rel=1e-9)

    def test_helium_1s2_single_corrected_key(self):
        """He '1s2': one output key per subshell, corrected in place."""
        from atomx.atom import Atom
        from atomx.methods.hartree_fock import HartreeFock
        from atomx.methods.relativistic_hf import BreitPauli

        result = Atom(Z=2, config="1s2").solve(BreitPauli())

        # Exactly one jj key — no stale "1s1/2" (uncorrected) next to a
        # synthesized NR-style "1s" key.
        assert set(result.orbital_energies) == {"1s1/2"}

        nr = Atom(Z=2, config="1s2").solve(HartreeFock())
        expected = (
            nr.orbital_energies["1s1/2"]
            + mass_velocity_shift(n=1, l=0, Z_eff=2.0)
            + darwin_shift(n=1, l=0, Z_eff=2.0)
        )
        assert result.orbital_energies["1s1/2"] == pytest.approx(
            expected, rel=1e-12
        )

    def test_jj_split_subshells_sharing_nl(self):
        """Two subshells with the same (n, l) but different kappa.

        The grouped shell must yield exactly two corrected jj keys (no
        KeyError, no leftover stale entries), and the fine-structure
        splitting stays the analytic α²Z⁴/32.
        """
        from atomx.methods.dirac_fock import Subshell
        from atomx.methods.relativistic_hf import BreitPauli

        subshells = [
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=1, occupation=2),   # 2p1/2
            Subshell(n=2, kappa=-2, occupation=4),  # 2p3/2
        ]
        result = BreitPauli().solve(Z=10, subshells=subshells)

        assert set(result.orbital_energies) == {"1s1/2", "2p1/2", "2p3/2"}
        splitting = (
            result.orbital_energies["2p3/2"]
            - result.orbital_energies["2p1/2"]
        )
        assert splitting == pytest.approx(ALPHA_SQ * 10.0**4 / 32.0, rel=1e-9)
