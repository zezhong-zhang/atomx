import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import gammaln

from atomx.constants import ALPHA
from atomx.corrections.qedmod.mohr import MohrBackend
from atomx.physics.dirac import (
    hydrogenic_dirac_coulomb_expectation,
    hydrogenic_dirac_energy,
)


def test_weak_binding_does_not_cancel_against_rest_energy():
    n = 1000000
    assert hydrogenic_dirac_energy(n, -1, 1) == pytest.approx(
        -0.5 / n**2, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize("z", [1.0, 12.0, 92.0])
@pytest.mark.parametrize("n,kappa", [(1, -1), (2, 1), (2, -2), (4, 2), (4, -3)])
def test_coulomb_expectation_equals_energy_charge_derivative(z, n, kappa):
    step = z * 1e-4
    # Five-point derivative of the independent energy expression.
    derivative = (
        hydrogenic_dirac_energy(n, kappa, z - 2 * step)
        - 8 * hydrogenic_dirac_energy(n, kappa, z - step)
        + 8 * hydrogenic_dirac_energy(n, kappa, z + step)
        - hydrogenic_dirac_energy(n, kappa, z + 2 * step)
    ) / (12 * step)
    assert hydrogenic_dirac_coulomb_expectation(n, kappa, z) == pytest.approx(
        z * derivative, rel=2e-10
    )


@pytest.mark.parametrize("z", [1, 12, 92])
def test_unscreened_dirac_1s_recovers_tabulated_self_energy(z):
    gamma = np.sqrt(1 - (ALPHA * z) ** 2)
    r = np.geomspace(1e-12 / z, 40 / z, 20001)
    # Exact total radial density of point-Coulomb Dirac 1s.
    density = np.exp(
        (2 * gamma + 1) * np.log(2 * z)
        - gammaln(2 * gamma + 1)
        + 2 * gamma * np.log(r)
        - 2 * z * r
    )
    # Quadrature through a sampled radial workspace, independent of SCF.
    from atomx.numerics.radial_workspace import make_grid_workspace

    # Simpson weights on this nonuniform mesh; each value is its basis integral.
    h = np.diff(r)
    weights = np.zeros_like(r)
    a = h[::2]
    b = h[1::2]
    weights[:-2:2] += (a + b) / 6 * (2 - b / a)
    weights[1:-1:2] += (a + b) ** 3 / (6 * a * b)
    weights[2::2] += (a + b) / 6 * (2 - a / b)
    # Mohr accepts the standard grid representation payload.
    workspace = make_grid_workspace(r, weights)
    # Direct density integration also tests the analytic reference against waves.
    expectation = simpson(density * (-z / r), x=r)
    assert expectation == pytest.approx(
        hydrogenic_dirac_coulomb_expectation(1, -1, z), rel=2e-10
    )
    orbitals = [
        (
            np.sqrt(density * (1 + gamma) / 2),
            -np.sqrt(density * (1 - gamma) / 2),
            1.0,
            -1,
            "1s1/2",
        )
    ]
    backend = MohrBackend()
    assert (
        pytest.approx(backend.compute(orbitals, None, z).E_SE, rel=2e-10)
        == backend.compute(orbitals, workspace, z).E_SE
    )
