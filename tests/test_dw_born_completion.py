"""Born high-energy completion + Maxwellian rate gates (review C6 / H0).

C6 (critical) — ``solver.py::_geometric_tail`` returned exactly ``0.0`` while
the partial-wave profile was still rising at ``kappa_max``, and
``born.py::born_blend_weights`` propagated that single zero through its
monotone cap, so the Born completion was weighted 0 at *every* energy.  The
shipped DW collision strength then fell with energy: at 50/100/200x threshold
(He-like Ne, ``1s2 -> 1s2p``, L_max=6, kappa_max=10) it was 20x/69x/263x below
the plane-wave Born limit.

H0 (high) — ``solver.py::collision_strength_to_rate`` used a prefactor of
``2*pi*a0^2/sqrt(m_e*kT)`` (dimensions cm^2/eV^(1/2), not cm^3/s) and scaled as
kT^(+1/2) where the Maxwellian average scales as kT^(-1/2).  With Omega = 1,
g_i = 1, kT = 1 eV it returned 9.18e-16 cm^3/s instead of 8.01e-08.

Gates here:

* 50-200x threshold: Omega rises with energy and sits within 30 % of the
  pure plane-wave Born value; the blend weights are not zero;
* a still-rising (unresolved) geometric tail is reported as a finite floor
  with ``converged=False``, never as 0.0, and one zero tail cannot zero the
  weights at lower energies;
* the rate coefficient matches an independent first-principles quadrature of
  ``<sigma v> = sqrt(8/(pi*m_e)) (kT)^(-3/2) Int sigma(E) E exp(-E/kT) dE``
  with ``sigma(E) = pi*hbar^2*Omega/(2*m_e*E*g_i)`` at T = 1, 10, 100, 1000 eV
  to < 1e-6, and the repo's own ``8.629e-6/(g_i*sqrt(T_K))`` convention (see
  ``atomx.plasma.crm``) to the precision of that 4-digit constant.

Note on the review's "exact 1.837e+03": evaluating
``a0^2*sqrt(2*pi/m_e)*(kT)^(-3/2)*Int Omega exp(-E/kT) dE`` with a0 in cm,
m_e in g and kT in erg yields cm*s/g, not cm^3/s — it is the correct rate
divided by ``E_h = 4.3597e-11 erg`` (factor 2.294e10).  The formula is
dimensionally consistent in atomic units, which is the form implemented.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_fac_dw_excitation import SEQUENCES, _threshold_eV  # noqa: E402

from atomx import Atom  # noqa: E402
from atomx.constants import BOLTZMANN_EV  # noqa: E402
from atomx.scattering.distorted_wave import born as born_mod  # noqa: E402
from atomx.scattering.distorted_wave.solver import (  # noqa: E402
    _geometric_tail,
    _geometric_tail_estimate,
    collision_strength_to_rate,
    dw_excitation_cross_section,
)

Z_NE = 10                      # He-like Ne: the repo's own DW benchmark case
HIGH_FACTORS = (50.0, 100.0, 200.0)
L_MAX = 6
KAPPA_MAX = 10
BORN_TOL = 0.30                # accepted |Omega/Omega_Born - 1| at 50-200x

# CODATA 2022, deliberately independent of atomx.constants: this is a check on
# the unit conversion itself, not on the constants table.
_M_E_G = 9.109_383_7139e-28
_E_CHARGE_ERG = 1.602_176_634e-12
_HBAR_ERG_S = 1.054_571_817e-27


def _dw_omega(energies_eV: np.ndarray, threshold_eV: float) -> np.ndarray:
    """Multiplet-summed Omega from the benchmark's own DW entry point."""
    spec = SEQUENCES["he-like"]
    charge = Z_NE - int(spec["charge_offset"])
    ground = Atom(
                  Z=Z_NE,
                  config=str(spec["ground"]),
                  charge=charge,
                  engine="bspline",
                  corrections="none",
             )
    excited = Atom(
                   Z=Z_NE,
                   config=str(spec["excited"]),
                   charge=charge,
                   engine="bspline",
                   corrections="none",
              )
    ground.solve()
    excited.solve()
    _sigma, omega = dw_excitation_cross_section(
        ground, excited, energies_eV,
        L_max=L_MAX, kappa_max=KAPPA_MAX, transition_energy_eV=threshold_eV)
    return np.asarray(omega, dtype=float)


@pytest.fixture(scope="module")
def high_energy_completion() -> dict[str, np.ndarray]:
    """Shipped and pure-Born Omega at 50-200x threshold + blend bookkeeping.

    One DW run with spies on the blend weight and on the geometric-tail
    estimator, plus one run with the blend forced to w = 1 (pure plane-wave
    Born).  Measured on 2026-09-19: 50/100/200x weights 0.743/0.758/0.766,
    Omega 0.1639/0.2178/0.3199 vs Born 0.2124/0.2843/0.4167.
    """
    spec = SEQUENCES["he-like"]
    threshold = _threshold_eV(Z_NE, spec)
    energies = np.asarray([threshold * f for f in HIGH_FACTORS], dtype=float)

    tails: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    tail_estimates: list[tuple[float, bool]] = []
    original_blend = born_mod.born_blend_weights
    original_estimate = _geometric_tail_estimate

    def blend_spy(omega_dw, tail_arr, p_born=0.5):
        w = original_blend(omega_dw, tail_arr, p_born)
        tails.append(np.asarray(tail_arr, dtype=float).copy())
        weights.append(np.asarray(w, dtype=float).copy())
        return w

    def estimate_spy(prof_l, l_last):
        tail, converged = original_estimate(prof_l, l_last)
        tail_estimates.append((float(tail), bool(converged)))
        return tail, converged

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(born_mod, "born_blend_weights", blend_spy)
        mp.setattr("atomx.scattering.distorted_wave.solver."
                   "_geometric_tail_estimate", estimate_spy)
        omega = _dw_omega(energies, threshold)
        shipped_estimates = list(tail_estimates)
        mp.setattr(born_mod, "born_blend_weights",
                   lambda omega_dw, tail_arr, p_born=0.5: np.ones(len(omega_dw)))
        omega_born = _dw_omega(energies, threshold)

    assert len(weights) == 1 and len(tails) == 1, "blend called once per run"
    assert len(shipped_estimates) == len(HIGH_FACTORS)
    return {
        "energies_eV": energies,
        "omega": omega,
        "omega_born": omega_born,
        "weights": weights[0],
        "tails": tails[0],
        "tail_estimates": np.asarray(shipped_estimates, dtype=float),
    }


# ---------------------------------------------------------------------------
# C6: the Born completion engages at high energy
# ---------------------------------------------------------------------------

def test_dw_omega_rises_and_tracks_born_at_high_energy(high_energy_completion):
    """50-200x threshold: Omega rises with energy and is within 30 % of Born.

    Before the fix Omega was 20x/69x/263x below the plane-wave Born limit and
    *fell* with energy (0.01049, 0.00412, 0.00159 at 50/100/200x).
    """
    omega = high_energy_completion["omega"]
    omega_born = high_energy_completion["omega_born"]
    assert np.all(omega_born > 0.0)
    assert np.all(np.diff(omega) > 0.0), (
        f"Omega must rise with energy at 50-200x threshold, got {omega}"
    )
    rel = omega / omega_born - 1.0
    assert np.all(np.abs(rel) <= BORN_TOL), (
        f"Omega/Omega_Born - 1 = {rel} exceeds {BORN_TOL:.0%} "
        f"(Omega={omega}, Born={omega_born})"
    )


def test_born_blend_weights_are_not_zero(high_energy_completion):
    """(b) the blend weights engage, and the top energy is not zeroed."""
    weights = high_energy_completion["weights"]
    assert weights.shape == (len(HIGH_FACTORS),)
    assert float(np.max(weights)) > 0.0, (
        f"Born blend weights are all zero — the completion never engages: "
        f"{weights}"
    )
    assert float(weights[-1]) > 0.0, (
        f"the highest energy has weight 0 (one zero tail propagated down the "
        f"monotone cap): {weights}"
    )
    assert float(np.max(weights)) > 0.5, f"completion barely engages: {weights}"


def test_unresolved_tail_is_flagged_not_zero(high_energy_completion):
    """The still-rising high-E profiles are reported as floors, not as 0."""
    estimates = high_energy_completion["tail_estimates"]
    assert estimates.shape == (len(HIGH_FACTORS), 2)
    assert np.all(estimates[:, 0] > 0.0), (
        f"unresolved tails must be a finite floor, got {estimates[:, 0]}"
    )
    assert not np.any(estimates[:, 1]), (
        "the 50-200x profiles have not decayed at kappa_max, so the estimator "
        f"must report converged=False, got {estimates[:, 1]}"
    )


def test_geometric_tail_estimate_rising_profile_is_a_floor():
    """Unit form: q rising at l_last -> capped floor + converged=False."""
    rising = np.array([0.0, 0.0, 0.5, 0.4, 0.45])
    tail, converged = _geometric_tail_estimate(rising, 4)
    assert converged is False
    assert tail == pytest.approx(0.45 * 0.85 / 0.15)
    # The EII-facing legacy entry point stays at 0.0 for an unresolved
    # profile (its consumer *adds* the tail; changing it is out of scope
    # here), so the flag — not this value — is what the completion reads.
    assert _geometric_tail(rising, 4) == 0.0

    decaying = np.array([0.0, 0.0, 0.5, 0.25, 0.125])
    tail_dec, converged_dec = _geometric_tail_estimate(decaying, 4)
    assert converged_dec is True
    assert tail_dec == pytest.approx(0.125 * 0.5 / 0.5)
    assert _geometric_tail(decaying, 4) == pytest.approx(tail_dec)


def test_blend_weights_single_zero_does_not_propagate():
    """One zero tail must not zero the weight at the lower energies."""
    omega = np.ones(5)
    weights = born_mod.born_blend_weights(
        omega, np.array([0.01, 0.1, 0.5, 0.0, 0.0]))
    assert np.all(weights[:3] > 0.0), f"zero propagated downward: {weights}"
    assert weights[0] == pytest.approx(math.sqrt(0.01))
    assert weights[2] == pytest.approx(math.sqrt(0.5))
    # a point with no tail of its own holds the higher-energy weight ...
    mid = born_mod.born_blend_weights(
        np.ones(3), np.array([0.01, 0.0, 0.5]))
    assert mid[1] == pytest.approx(mid[2])


# ---------------------------------------------------------------------------
# H0: the Maxwellian rate coefficient
# ---------------------------------------------------------------------------

def _closed_form_rate_cm3_s(temperature_eV: float, g_i: float = 1.0,
                            omega: float = 1.0) -> float:
    """Quadrature of the textbook Maxwellian average, cgs-consistent.

    <sigma v> = sqrt(8/(pi m)) (kT)^(-3/2) Int sigma(E) E exp(-E/kT) dE with
    sigma(E) = pi*hbar^2*Omega/(2 m E g_i) -> sigma(E)*E is E-independent, so
    the integrand is analytic and ``quad`` resolves it exactly.
    """
    kT_erg = temperature_eV * _E_CHARGE_ERG
    sigma_times_E = (math.pi * _HBAR_ERG_S ** 2
                     / (2.0 * _M_E_G * g_i) * omega)
    # Finite upper limit (the callers' grid also stops at 60 kT; the neglected
    # tail is e^-60) and epsabs=0: the integral is ~1e-39 in cgs, far below
    # quad's default absolute tolerance, and QUADPACK's infinite-range
    # transform returns 0.0 on this scale.
    integral = quad(lambda energy_erg: sigma_times_E
                    * math.exp(-energy_erg / kT_erg),
                    0.0, 60.0 * kT_erg, epsabs=0.0, epsrel=1e-13,
                    limit=400)[0]
    return math.sqrt(8.0 / (math.pi * _M_E_G)) * kT_erg ** -1.5 * integral


@pytest.mark.parametrize("temperature_eV", [1.0, 10.0, 100.0, 1000.0])
def test_collision_strength_to_rate_matches_closed_form(temperature_eV):
    """Omega = 1: agree with the first-principles quadrature to < 1e-6."""
    energy_eV = np.linspace(0.0, 60.0 * temperature_eV, 200_001)
    omega = np.ones_like(energy_eV)
    rate = collision_strength_to_rate(omega, energy_eV, temperature_eV)
    expected = _closed_form_rate_cm3_s(temperature_eV)
    assert rate == pytest.approx(expected, rel=1e-6)


def test_collision_strength_to_rate_magnitude_and_scaling():
    """The 1 eV value is 8.01e-08 cm^3/s and scales as kT^(-1/2)."""
    rate_1 = collision_strength_to_rate(
        np.ones_like(np.linspace(0.0, 60.0, 200_001)),
        np.linspace(0.0, 60.0, 200_001), 1.0)
    assert 8.00e-8 < rate_1 < 8.02e-8, (
        f"Omega = 1, g_i = 1, kT = 1 eV must give 8.011e-08 cm^3/s, "
        f"got {rate_1:.6e} (the old prefactor gave 9.18e-16)"
    )
    for temperature_eV in (10.0, 100.0):
        energy_eV = np.linspace(0.0, 60.0 * temperature_eV, 200_001)
        rate = collision_strength_to_rate(
            np.ones_like(energy_eV), energy_eV, temperature_eV)
        assert rate == pytest.approx(rate_1 / math.sqrt(temperature_eV),
                                     rel=1e-6)


def test_collision_strength_to_rate_uses_g_i():
    """The initial-state weight enters as 1/g_i (dropped by the old code)."""
    energy_eV = np.linspace(0.0, 600.0, 200_001)
    omega = np.ones_like(energy_eV)
    rate_1 = collision_strength_to_rate(omega, energy_eV, 10.0)
    rate_3 = collision_strength_to_rate(omega, energy_eV, 10.0, g_i=3)
    assert rate_3 == pytest.approx(rate_1 / 3.0, rel=1e-12)


@pytest.mark.parametrize("temperature_eV", [1.0, 10.0, 100.0, 1000.0])
def test_collision_strength_to_rate_matches_crm_convention(temperature_eV):
    """Cross-check against the repo's own 8.629e-6/(g_i sqrt(T_K)) rate.

    ``atomx.plasma.crm`` builds C_exc = 8.6291e-6 * Upsilon / (g_l sqrt(T_K));
    for a constant Omega the effective collision strength Upsilon is that
    constant, so the two paths must agree to the precision of the 4-digit
    constant (measured 3.7e-6).
    """
    energy_eV = np.linspace(0.0, 60.0 * temperature_eV, 200_001)
    rate = collision_strength_to_rate(
        np.ones_like(energy_eV), energy_eV, temperature_eV)
    expected = 8.6291e-6 / math.sqrt(temperature_eV / BOLTZMANN_EV)
    assert rate == pytest.approx(expected, rel=1e-4)


class TestCoulombBetheTopupCoefficient:
    """The Burgess-Sheorey top-up must be the exact Bethe limit.

    ``Omega = (4 g_i f / dE) ln[(k_i + k_f)/(k_i - k_f)]``.  Two defects were
    fixed here (review M16): the ``g_i = 2J_i + 1`` factor was missing, and
    an undocumented ``1 - 1/(L_truncation + 1)`` factor (0.857 at L = 6)
    silently broke the closed-form limit.  The He-like anchor is
    ``1s2 1S0 -> 1s2p 1P1`` with ``f = 0.4162`` and ``dE = 0.375 Ha``
    (Bethe coefficient ``4 g_i f / dE = 8.879`` for ``g_i = 2``).
    """

    def test_matches_the_closed_form_exactly(self):
        from atomx.constants import HARTREE_EV
        from atomx.scattering.distorted_wave.solver import coulomb_bethe_topup

        f, dE_eV = 0.4162, 0.375 * HARTREE_EV
        energies = np.array([dE_eV * 2.0, dE_eV * 10.0, dE_eV * 100.0])
        omega = coulomb_bethe_topup(f, energies, dE_eV, 6, g_i=2.0)

        e_au = energies / HARTREE_EV
        w_au = dE_eV / HARTREE_EV
        ki = np.sqrt(2.0 * e_au)
        kf = np.sqrt(2.0 * (e_au - w_au))
        expected = (4.0 * 2.0 * f / w_au) * np.log((ki + kf) / (ki - kf))
        assert np.allclose(omega, expected, rtol=1e-12)

    def test_linear_in_g_i(self):
        from atomx.constants import HARTREE_EV
        from atomx.scattering.distorted_wave.solver import coulomb_bethe_topup

        f, dE_eV = 0.4162, 0.375 * HARTREE_EV
        energies = np.array([dE_eV * 5.0])
        one = coulomb_bethe_topup(f, energies, dE_eV, 6, g_i=1.0)
        two = coulomb_bethe_topup(f, energies, dE_eV, 6, g_i=2.0)
        assert np.allclose(two, 2.0 * one, rtol=1e-12)

    def test_no_hidden_L_dependence(self):
        """The closed form does not depend on where the sum was truncated."""
        from atomx.constants import HARTREE_EV
        from atomx.scattering.distorted_wave.solver import coulomb_bethe_topup

        f, dE_eV = 0.4162, 0.375 * HARTREE_EV
        energies = np.array([dE_eV * 20.0])
        a = coulomb_bethe_topup(f, energies, dE_eV, 4, g_i=2.0)
        b = coulomb_bethe_topup(f, energies, dE_eV, 40, g_i=2.0)
        assert np.allclose(a, b, rtol=1e-14)
