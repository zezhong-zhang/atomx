"""Fe K-shell RR comparison against a hydrogenic benchmark fallback."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.scattering.radiative_recombination import (
    RRResult,
    rr_cross_section,
    rr_rate_coefficient,
)


FE26_K_SHELL_RR_REFERENCE = np.array([
    9.48e-12,
    4.50e-11,
    5.96e-11,
    9.55e-11,
    9.96e-11,
])


@pytest.mark.slow
def test_fe26_k_shell_rr_hydrogenic_benchmark_within_15_percent():
    atom = Atom(Z=26, config="1s1", numerics="bspline", breit=False, qed=False)
    atom.solve()

    temperatures = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
    energies = np.geomspace(1.0, 2.0e5, 80)
    rr = rr_cross_section(
        atom._result,
        energies,
        capture_orbitals=["1s1/2"],
        g_initial=1,
        multipoles=["E1", "M1", "E2"],
    )
    rates = rr_rate_coefficient(rr, temperatures)
    np.testing.assert_allclose(rates, FE26_K_SHELL_RR_REFERENCE, rtol=0.15)

    extra = rr.sigma_by_multipole["M1"] + rr.sigma_by_multipole["E2"]
    extra_rates = rr_rate_coefficient(
        RRResult(energies, extra),
        temperatures,
    )
    ratio = extra_rates / rates
    assert ratio[0] < 0.01
    assert 0.01 <= ratio[2] <= 0.05
    assert 0.01 <= ratio[-1] <= 0.05
