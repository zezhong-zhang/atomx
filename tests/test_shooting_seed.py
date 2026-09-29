"""Shooting-to-B-spline projection is an initial guess, not a new Hamiltonian."""

import numpy as np
import pytest

from atomx import Atom, Basis, NuclearModel
from atomx.config.jj_coupling import generate_jj_average
from atomx.methods.dirac_fock import DiracFock
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.shooting_seed import project_shooting_orbitals
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver


@pytest.mark.parametrize("z,configuration", [(2, "1s2"), (10, "1s2 2s2 2p6")])
def test_projected_seed_recovers_independent_bspline_solution(z, configuration):
    basis = Basis(r_max=40.0)
    knots, ns = make_physics_knots(
        ks=basis.ks,
        Z=z,
        nuclear="point",
        r_max=basis.r_max,
        hi=basis.grid_hi,
        he=basis.grid_he,
        hmax=basis.grid_hmax,
    )
    workspace = prepare_bspline_data(knots, ns, basis.ks, kinetic_balance=True)
    shells, _ = generate_jj_average(configuration)
    shooting = ShootingHFSolver(
        z,
        shells,
        n_grid=1600,
        r_max=max(basis.r_max, float(np.max(workspace["gr"]))),
        nuclear="point",
    ).solve(max_iter=100, tol=1e-10)
    assert shooting.converged
    seed, error = project_shooting_orbitals(shooting, workspace, z, nuclear="point")
    assert error < 1e-3
    options = dict(nuclear=NuclearModel.POINT, breit=False, qed=False)
    reference = Atom(z, config=configuration, engine="bspline", basis=basis).solve(
        DiracFock(**options), tol=1e-10, max_iter=150
    )
    seeded = Atom(z, config=configuration, engine="bspline", basis=basis).solve(
        DiracFock(**options, seed_orbitals=seed), tol=1e-10, max_iter=150
    )
    assert reference.converged and seeded.converged
    assert seeded.total_energy == pytest.approx(reference.total_energy, abs=1e-8, rel=0)
    assert seeded.orbital_energies == pytest.approx(
        reference.orbital_energies, abs=1e-7, rel=0
    )
    shooting.converged = False
    with pytest.raises(ValueError, match="converged"):
        project_shooting_orbitals(shooting, workspace, z)
