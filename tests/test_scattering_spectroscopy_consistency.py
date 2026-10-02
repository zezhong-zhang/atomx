"""Scattering <-> spectroscopy consistency gates (FAC method, 2026-10-02).

The anomalous factors in ``atomx.scattering.anomalous`` are derived
from the same dipole photoionization matrix elements as the
bound-free spectroscopy, so their internal consistency is testable
against exact sum rules:

* TRK / dispersion sum rule: (2/pi) \\int f2(omega)/omega domega
  recovers the active electron count N (H: exactly 1, all oscillator
  strength is continuum above the single 1s electron).
* Optical-theorem spot check: H 1s f2 at the K edge sits near the
  literature peak (~1.2 at threshold).
* Kramers-Kronig zero-frequency limit: f1(0) = -N.

These gates are the scattering-side mirror of the PI/RR spectroscopy
gates; a failure means the scattering factors and the spectroscopy
have drifted apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.scattering.anomalous import (  # noqa: E402
    atomic_sigma_pi,
    f1_kramers_kronig,
    f2_from_sigma,
    trk_scattering_sum,
)


@pytest.fixture(scope="module")
def hydrogen_result():
    # engine="bspline" mirrors the FAC PI/RR benchmark setup; the
    # SCFResult shape is consumed via orbitals_by_subshell()
    atom = Atom(Z=1, corrections="none", engine="bspline")
    atom.solve()
    return atom._result


@pytest.fixture(scope="module")
def h_f2_table(hydrogen_result):
    """f2(omega) on a wide log grid above the 1s threshold (0.5 Ha)."""
    omega = np.geomspace(0.55, 400.0, 160)
    sigma = atomic_sigma_pi(hydrogen_result, omega)
    f2 = f2_from_sigma(omega, sigma)
    assert np.any(f2 > 0), "no continuum absorption above threshold"
    return omega, f2


def test_hydrogen_f2_edge_peak(h_f2_table):
    """f2 just above the K edge is near the literature peak (~1.2)."""
    omega, f2 = h_f2_table
    assert 0.6 <= f2.max() <= 2.0, (
        f"H 1s f2 peak {f2.max():.3f} outside [0.6, 2.0] "
        "(literature edge peak ~1.2)"
    )


# Exact H 1s discrete-series oscillator strengths (closed form);
# the spectroscopy side of the sum rule.
def _f_discrete_1s(n_max=200):
    import math
    total = 0.0
    for n in range(2, n_max + 1):
        lg = (math.log(256.0) + 5 * math.log(n)
              + (2 * n - 4) * math.log(n - 1)
              - math.log(3.0) - (2 * n + 4) * math.log(n + 1))
        total += math.exp(lg)
    return total


F_DISCRETE_1S = _f_discrete_1s()          # = 0.5650 (0.4162 from 2p)
F_CONTINUUM_1S = 1.0 - F_DISCRETE_1S      # = 0.4350
# the exact-sigma integral over the GRID window [0.55, 400] Ha
# (threshold slice [0.5, 0.55] excluded): 0.4350 - 0.066 = 0.369
F_CONTINUUM_WINDOW = 0.369


def test_hydrogen_trk_scattering_plus_spectroscopy(h_f2_table):
    """Continuum scattering integral + discrete f = 1 (TRK)."""
    omega, f2 = h_f2_table
    n_eff = trk_scattering_sum(omega, f2)
    assert 0.33 <= n_eff <= 0.41, (
        f"continuum TRK share = {n_eff:.4f} vs the exact-window "
        f"{F_CONTINUUM_WINDOW} (full continuum 0.4350; grid starts "
        "0.05 Ha above threshold)"
    )
    combined = n_eff + F_DISCRETE_1S
    assert 0.88 <= combined <= 0.99, (
        f"scattering + discrete spectroscopy = {combined:.4f} vs the "
        f"window target {F_CONTINUUM_WINDOW + F_DISCRETE_1S:.3f} "
        f"(continuum {n_eff:.4f} + discrete {F_DISCRETE_1S:.4f}; the "
        "missing threshold slice 0.066 completes TRK to 1)"
    )


def test_hydrogen_kramers_kronig_f1_zero_limit(h_f2_table):
    """f1(0) = -(continuum integral); the discrete lines add
    -f_discrete on the spectroscopy side (total -1)."""
    omega, f2 = h_f2_table
    f1_0 = f1_kramers_kronig(omega, f2, np.array([1e-6]))[0]
    assert -0.41 <= f1_0 <= -0.30, (
        f"KK f1(0) = {f1_0:.4f} vs the continuum-window share "
        f"-{F_CONTINUUM_WINDOW} (discrete lines add -0.565)"
    )
