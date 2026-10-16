"""The same declared determinant budget reaches CI energies and their RDM."""

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.methods.dirac_fock import DFSolver
from atomx.methods.mcdhf.solver import MCDHFSolver
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.physics.csf.radial_density import CIRadialDensity


def model():
    csfs = [c for cfg in ("1s2", "2s2") for c in enumerate_jj_csfs(cfg)]
    seed = DFSolver(2, csfs[0].subshells, ns=24, ks=7)
    return csfs, seed


def test_budget_rejects_the_complete_rdm_space_instead_of_truncating():
    from atomx.angular.csf_hamiltonian import ExactCISpaceError

    csfs, seed = model()
    with pytest.raises(ExactCISpaceError, match="cap of 1"):
        CSFOrbitalFunctional(csfs, seed.bsp_data, 2, max_determinants=1)
    assert CIRadialDensity(csfs, max_determinants=2).max_determinants == 2


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_invalid_budgets_reject_before_angular_allocation(limit):
    csfs, _ = model()
    with pytest.raises(ValueError, match="positive integer"):
        CIRadialDensity(csfs, max_determinants=limit)


def test_explicit_budget_preserves_full_energy_and_signed_gradient():
    csfs, seed = model()
    kwargs = dict(nuclear_potential=seed._nuclear_potential)
    default = CSFOrbitalFunctional(csfs, seed.bsp_data, 2, **kwargs)
    bounded = CSFOrbitalFunctional(csfs, seed.bsp_data, 2, max_determinants=2, **kwargs)
    spectrum = solve_dirac(default.one_body[-1], default.metric[-1], default.ns, -1)
    orbitals = {key: (spectrum.coefficients[:default.ns, i],
                      spectrum.coefficients[default.ns:, i])
                for i, key in enumerate(default.keys)}
    e0, g0, c0 = default.evaluate(orbitals)
    e1, g1, c1 = bounded.evaluate(orbitals)
    assert e0 == e1
    np.testing.assert_array_equal(c0.state_energies, c1.state_energies)
    for key in default.keys:
        np.testing.assert_array_equal(g0[key], g1[key])


def test_public_solver_passes_budget_to_the_correlated_functional(monkeypatch):
    import atomx.methods.csf.variational as module

    seen = []
    real = module.CSFOrbitalFunctional

    class Capture(real):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs["max_determinants"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(module, "CSFOrbitalFunctional", Capture)
    csfs, _ = model()
    result = MCDHFSolver(2, csfs, ns=24, ks=7, use_physics_grid=False,
                         max_determinants=2).solve(max_iter=1, scf_tol=1e-5)
    assert seen == [2]
    assert len(result.state_energies) == 2
