"""Silent-failure guards: no zero/sea/ceiling results reported as converged.

Before these guards the package could return a *plausible* number with
``converged=True`` in three distinct ways:

* an unbound (virtual) d/f subshell collapsed into the negative-energy
  continuum, giving a total energy of ``-2 m_e c²`` times the occupancy
  (about ``-3.8e4`` Ha) — e.g. ``Atom(Z=10, charge=-1, config=...3d1)``;
* a requested subshell with no bound state anywhere in the basis
  produced an all-zero orbital, so the "total energy" was ``0.0`` Ha and
  ``orbital_energies`` was empty;
* the shooting engine ran out of box and returned its own bracketing
  ceiling ``-Z/r_max`` (4 % error for H 6s, 31 % for 7s, and the whole
  Na I Rydberg series 9s..13s pinned to a single value).

See ``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding C4.
"""

from __future__ import annotations

import math

import pytest

from atomx import Atom, Basis
from atomx.constants import C
from atomx.methods.dirac_fock import UnboundSubshellError

HARTREE_EV = 27.211386245988


def _exact_dirac_coulomb(n, kappa, Z):
    """Exact point-nucleus Dirac-Coulomb eigenvalue, Hartree."""
    gamma = math.sqrt(kappa * kappa - (Z / C) ** 2)
    N = math.sqrt((n - abs(kappa) + gamma) ** 2 + (Z / C) ** 2)
    return C * C * (math.sqrt(1.0 - (Z / C) ** 2 / N ** 2) - 1.0)


class TestNegativeEnergyCollapse:
    """An unbound subshell must raise, not return a sea state."""

    @pytest.mark.parametrize(
        "Z, charge, config",
        [
            (10, -1, "1s2 2s2 2p6 3d1"),   # 3d is virtual in Ne
            (2, -1, "1s2 4f1"),             # 4f is virtual in He
            (18, 0, "1s2 2s2 2p6 3s2 3p6 3d1"),  # 3d virtual in Ar
        ],
    )
    def test_unbound_subshell_raises(self, Z, charge, config):
        atom = Atom(
                    Z=Z,
                    charge=charge,
                    config=config,
                    engine="bspline",
                    corrections="none",
               )
        with pytest.raises(UnboundSubshellError, match="no bound state"):
            atom.solve()

    def test_bound_d_orbitals_still_work(self):
        """The guard must not fire for genuinely occupied 3d shells."""
        for Z, charge, config in [
            (21, 0, "1s2 2s2 2p6 3s2 3p6 3d1 4s2"),   # Sc
            (26, 0, "1s2 2s2 2p6 3s2 3p6 3d6 4s2"),   # Fe
        ]:
            atom = Atom(
                        Z=Z,
                        charge=charge,
                        config=config,
                        engine="bspline",
                        corrections="none",
                   )
            atom.solve()
            assert atom.converged
            assert atom.orbital_energies, "bound 3d must produce orbitals"
            for e in atom.orbital_energies.values():
                assert e > -0.5 * C * C, "no state may fall below -m_e c^2"


class TestNoEmptyResults:
    def test_rydberg_state_outside_the_box_raises(self):
        """``Atom(Z=1, config="7s1", numerics="bspline")`` used to return
        ``total_energy == 0.0`` with ``orbital_energies == {}``."""
        atom = Atom(Z=1, config="7s1", engine="bspline", corrections="none")
        with pytest.raises(UnboundSubshellError):
            atom.solve()

    def test_explicit_r_max_makes_it_solvable(self):
        atom = Atom(
                    Z=1,
                    config="7s1",
                    engine="bspline",
                    basis=Basis(r_max=400.0),
                    corrections="none",
               )
        atom.solve()
        exact = _exact_dirac_coulomb(7, -1, 1)
        assert atom.total_energy == pytest.approx(exact, rel=1e-6)


class TestShootingBox:
    """The shooting engine must not return its bracketing ceiling."""

    @pytest.mark.parametrize("n", [5, 6, 7, 8])
    def test_hydrogen_rydberg_series_is_accurate(self, n):
        """The default box now auto-sizes to contain the state.

        Before: 5s had a 4.8e-6 error, 6s a 4 % error pinned exactly at
        ``-Z/r_max``, and 7s was 31 % wrong — all with ``converged=True``.
        """
        atom = Atom(
                    Z=1,
                    config=f"{n}s1",
                    engine="shooting",
                    exchange="slater",
                    corrections="none",
               )
        atom.solve()
        exact = _exact_dirac_coulomb(n, -1, 1)
        assert atom.total_energy == pytest.approx(exact, rel=1e-6), (
            f"{n}s1/2: {atom.total_energy:+.12f} vs exact {exact:+.12f}"
        )

    def test_explicit_small_box_raises(self):
        """An explicit too-small box is a hard error, not a wrong number."""
        atom = Atom(
                    Z=1,
                    config="8s1",
                    engine="shooting",
                    basis=Basis(r_max=75.0),
                    exchange="slater",
                    corrections="none",
               )
        with pytest.raises(RuntimeError, match="box too small"):
            atom.solve()

    def test_resolve_r_max_is_adaptive(self):
        from atomx.methods.dirac_fock import Subshell

        shallow = Atom(Z=1, config="6s1", engine="shooting")
        assert shallow._resolve_r_max([Subshell(6, -1, 1.0)]) >= 5.0 * 36
        # Ordinary configurations keep the historical 75 Bohr box.
        assert shallow._resolve_r_max([Subshell(2, -1, 2.0)]) == 75.0
        assert Atom(Z=92, engine="shooting")._resolve_r_max(
            [Subshell(1, -1, 2.0)]) == 75.0
        # An explicit value always wins.
        assert Atom(
                    Z=1,
                    basis=Basis(r_max=123.0),
               )._resolve_r_max([Subshell(9, -1, 1.0)]) == 123.0


class TestNegativeContinuumEdge:
    def test_solve_dirac_discards_the_sea(self):
        """Eigenpairs at or below ``-m_e c²`` are not orbitals."""
        import numpy as np

        from atomx.numerics.bspline.basis import prepare_bspline_data
        from atomx.numerics.bspline.dirac_assembly import (
            NEGATIVE_CONTINUUM_EDGE,
            build_dirac_matrices,
            solve_dirac,
        )
        from atomx.numerics.bspline.knots import make_physics_knots

        assert NEGATIVE_CONTINUUM_EDGE == -C * C
        t, ns = make_physics_knots(ks=9, Z=10, r_max=40.0)
        bd = prepare_bspline_data(t, ns, 9, kinetic_balance=True)
        nsp = bd.get("nsp", ns)
        nsq = bd.get("nsq", ns)
        H, S = build_dirac_matrices(ns, 3, 10, bd, None)
        res = solve_dirac(H, S, ns, 3, nsp=nsp, nsq=nsq)
        assert np.all(res.bound_energies > NEGATIVE_CONTINUUM_EDGE)
        assert np.all(res.bound_energies < 0.0)
        # Every returned "bound" energy must be a genuine bound state.
        for e in res.bound_energies:
            assert e > -C * C, f"{e} is in the negative-energy continuum"
