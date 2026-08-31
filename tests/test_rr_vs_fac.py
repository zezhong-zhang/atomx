"""Fe K-shell RR comparison against a hydrogenic benchmark fallback.

Benchmark provenance (HYP-1645)
-------------------------------
The reference rate coefficients are the exact *nonrelativistic
hydrogenic* values: the Stobbe (1930) 1s photoionisation cross section
mapped through Milne detailed balance
(:func:`atomx.methods.hydrogenic_cross_sections.milne_rr_cross_section`)
and Maxwellian-averaged with
:func:`atomx.plasma.rates.maxwellian_rate` on the same energy grid as
the code under test.

They replace the original hard-coded constants
``[9.48e-12, 4.50e-11, 5.96e-11, 9.55e-11, 9.96e-11]`` cm^3/s, which
were outputs of the pre-#65 B-spline continuum path and are physically
incorrect.  That path used box-normalized pseudo-states without the
``sqrt(dn/dE)`` delta-energy density factor (plus a 2x angular
double-count), so its "cross sections" were per-pseudo-state line
strengths: understated by the pseudo-level spacing ``dn/dE`` (which
shrinks with energy), producing rate coefficients 10x-300x too large
that *increased* with temperature.  The corrected energy-normalized
calculation agrees with the exact hydrogenic benchmark to 5-13 % over
T = 100 eV - 10 keV and reproduces the physical decreasing temperature
trend; the residual at 100 eV is dominated by relativistic corrections
to the near-threshold K-shell bound-free cross section at Z = 26,
which the nonrelativistic benchmark does not include.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.methods.hydrogenic_cross_sections import milne_rr_cross_section
from atomx.plasma.rates import maxwellian_rate
from atomx.scattering.radiative_recombination import (
    RRResult,
    rr_cross_section,
    rr_rate_coefficient,
)


@pytest.mark.slow
def test_fe26_k_shell_rr_hydrogenic_benchmark_within_15_percent():
    atom = Atom(Z=26, config="1s1", numerics="bspline", breit=False, qed=False)
    atom.solve()

    temperatures = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
    # 200-point grid (up from 80): the delta-energy-normalized continuum
    # represents sigma_RR as a staircase over the pseudo-spectrum, so the
    # Maxwellian integral needs a grid finer than the pseudo-level spacing
    # to keep sampling error well below the relativistic-correction floor.
    energies = np.geomspace(1.0, 2.0e5, 200)
    rr = rr_cross_section(
        atom._result,
        energies,
        capture_orbitals=["1s1/2"],
        g_initial=1,
        multipoles=["E1", "M1", "E2"],
    )
    rates = rr_rate_coefficient(rr, temperatures)

    # Exact nonrelativistic hydrogenic reference: capture of a free
    # electron by bare Fe26+ (g_recombining = 1) into the 1s (j = 1/2,
    # g_recombined = 2) shell of Fe25+, Stobbe PI + Milne detailed
    # balance.  Numerically, on this grid:
    #   [5.08e-12, 2.23e-12, 1.53e-12, 5.63e-13, 3.35e-13] cm^3/s.
    sigma_ref_cm2 = milne_rr_cross_section(
        26, 1, energies / HARTREE_EV, g_recombined=2.0, g_recombining=1.0
    )
    reference = np.array([
        maxwellian_rate(sigma_ref_cm2, energies, float(T)) for T in temperatures
    ])
    np.testing.assert_allclose(rates, reference, rtol=0.15)

    # M1 + E2 (relativistic channels) share of the K-shell rate:
    # ~0.5-1 % below 1 keV, rising to ~3 % at 10 keV — consistent with
    # the expected O((alpha*Z)^2) ~= 3.6 % relativistic correction at
    # Z = 26.  Bounds bracket that with margin.
    extra = rr.sigma_by_multipole["M1"] + rr.sigma_by_multipole["E2"]
    extra_rates = rr_rate_coefficient(
        RRResult(energies, extra),
        temperatures,
    )
    ratio = extra_rates / rates
    assert ratio[0] < 0.01
    assert 0.005 <= ratio[2] <= 0.05
    assert 0.01 <= ratio[-1] <= 0.05
