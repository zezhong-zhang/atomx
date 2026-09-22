"""Fe K-shell RR comparison against a hydrogenic benchmark fallback.

Benchmark provenance (HYP-1645)
-------------------------------
The reference rate coefficients are the exact *nonrelativistic
hydrogenic* values: the Stobbe (1930) 1s photoionisation cross section
mapped through Milne detailed balance
(:func:`atomx.methods.hydrogenic_cross_sections.milne_rr_cross_section`)
and Maxwellian-averaged with
:func:`atomx.plasma.rates.maxwellian_rate` on the same energy grid as
the code under test.  The Milne prefactor of that chain is anchored
*absolutely* by the H 1s RR rate at 1e4 K (Storey & Hummer 1995) in
``test_hydrogenic_cross_sections``, so this benchmark is not
common-mode with the code under test.

Two layered defects were uncovered (see HYP-1645):

1. The original hard-coded constants
   ``[9.48e-12, 4.50e-11, 5.96e-11, 9.55e-11, 9.96e-11]`` cm^3/s were
   outputs of the pre-#65 B-spline continuum path, which used
   box-normalized pseudo-states without the ``sqrt(dn/dE)``
   delta-energy density factor (plus a 2x angular double-count) —
   per-pseudo-state line strengths whose Maxwellian average was
   10x-300x too large and *increased* with temperature.
2. A residual factor-2 error in the Milne degeneracy prefactor
   (introduced in ``74ff5ce``, predating #65): the free-electron spin
   degeneracy was counted both in ``(g_n / (2 g_ion))`` and in
   ``alpha^2 omega^2 / (2 E_e)``, making all RR cross sections and
   rate coefficients 2x too small.  Fixed in
   ``rr_cross_section``/``rr_cross_section_rydberg`` and in
   ``milne_rr_cross_section`` — the standard form counts the spin
   once (it cancels the two photon polarisations in detailed
   balance; Gu 2003, ApJ 589, 1085).

After both fixes the energy-normalized calculation agrees with the
exact hydrogenic benchmark to 5-13 % over T = 100 eV - 10 keV and
reproduces the physical decreasing temperature trend; the residual at
100 eV is dominated by relativistic corrections to the near-threshold
K-shell bound-free cross section at Z = 26, which the nonrelativistic
benchmark does not include.
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
    atom = Atom(Z=26, config="1s1", engine="bspline", corrections="none")
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
    #   [1.02e-11, 4.46e-12, 3.06e-12, 1.13e-12, 6.71e-13] cm^3/s.
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
