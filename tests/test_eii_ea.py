"""Excitation-autoionization channel in total EII (review M6, EA half).

He-like Fe: 1s2 + e -> 2s 2p + e (double excitation, 13.5 keV) lies
above the 1s ionization limit (8.8 keV), so it autoionizes and adds to
net ionization.  EA turns on at the excitation threshold and its
magnitude is a DW collision strength times a branching ratio <= 1.
(The 3-open-shell Li-like case below was unblocked by the 2026-07-16
repeated-parentage CI fix.)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402
from atomx.scattering.electron_ionization import (  # noqa: E402
    excitation_autoionization_cross_section,
)


@pytest.fixture(scope="module")
def he_like():
    g = Atom(Z=26, config="1s2", charge=24, engine="bspline", corrections="none")
    e = Atom(Z=26, config="2s1 2p1", charge=24, engine="bspline", corrections="none")
    g.solve()
    e.solve()
    return g, e


def test_ea_threshold_and_shape(he_like):
    g, e = he_like
    th = float((e._result.total_energy - g._result.total_energy)
               * HARTREE_EV)
    # excited config is autoionizing: above the 1s ionization limit
    I_1s = -float(g._result.orbital_energies["1s1/2"]) * HARTREE_EV
    assert th > I_1s
    E = np.array([0.9 * th, 1.5 * th, 3.0 * th])
    sig, per = excitation_autoionization_cross_section(
        g, [e], E, kappa_max=6)
    assert sig[0] == 0.0            # below excitation threshold
    assert sig[1] > 0.0 and sig[2] > 0.0
    assert per[0][1] == sig[1]      # single config


def test_ea_branching_scaling(he_like):
    g, e = he_like
    th = float((e._result.total_energy - g._result.total_energy)
               * HARTREE_EV)
    E = np.array([2.0 * th])
    s1, _ = excitation_autoionization_cross_section(g, [e], E,
                                                    kappa_max=6)
    s2, _ = excitation_autoionization_cross_section(
        g, [e], E, branching=[0.25], kappa_max=6)
    assert s2[0] == pytest.approx(0.25 * s1[0], rel=1e-12)
    with pytest.raises(ValueError, match="branching"):
        excitation_autoionization_cross_section(g, [e], E,
                                                branching=[1.5])


def _saha_capture_coefficient_cm3(temperature_eV):
    """(h^2 / (2 pi m_e k T))^{3/2} in cm^3, from CODATA 2022 by hand.

    This is the detailed-balance factor between the continuum (e^- + ion) and
    a discrete autoionizing state: the dielectronic-capture coefficient is
    ``alpha_cap = P(T) (g_d / 2 g_i) Gamma_auger exp(-E_d/kT)``, i.e. the
    Saha population of the resonance times its autoionization rate.  Written
    out from the constants here so it cannot be common-mode with atomx.
    """
    planck_j_s = 6.626_070_15e-34
    m_e_kg = 9.109_383_7139e-31
    ev_j = 1.602_176_634e-19
    return ((planck_j_s ** 2
             / (2.0 * math.pi * m_e_kg * temperature_eV * ev_j)) ** 1.5
            * 1.0e6)  # m^3 -> cm^3


def test_detailed_balance_capture_flux_is_shared_by_dr_and_re():
    """DR and resonant excitation are two branches of ONE Saha capture flux.

    Detailed balance (Saha) fixes the dielectronic-capture coefficient into a
    resonance d with energy E_d, statistical weight g_d, autoionization width
    Gamma_a and total width Gamma_tot:

        alpha_cap(T) = (h^2/(2 pi m_e kT))^{3/2} (g_d/2g_i)
                       Gamma_a exp(-E_d/kT),

    and every open channel takes its branching fraction of it:
    ``alpha_DR = alpha_cap Gamma_r/Gamma_tot`` and
    ``alpha_RE = alpha_cap Gamma_a^f/Gamma_tot``.

    The limits are exact and are what this test gates:

    * radiative stabilization dominating (Gamma_r >> Gamma_a): alpha_DR ->
      alpha_cap;
    * autoionization into the excited channel dominating (Gamma_a^f >>>
      Gamma_a): alpha_RE -> alpha_cap;
    * both channels weak (Gamma_r, Gamma_a^f << Gamma_a): each rate equals
      ``P(T) (g_d/2g_i) Gamma_x exp(-E_d/kT)`` with its OWN partial width
      Gamma_x — the two channels no longer coincide once they carry different
      widths.

    The old version of this test built a twin state so that ``a_RE == a_DR``
    held to 1e-12 by construction (same kernel, same branch algebra); it is
    replaced by these externally-anchored limits.  Note the documented
    limitation the limits avoid: ``dr_rate_coefficient`` uses
    Gamma_tot = Gamma_a + A_r and ignores a ``partial_auger_to_final`` extra
    channel, while ``resonant_excitation_rate`` adds it, so the two are only
    mutually consistent when the other channel is closed.
    """
    from atomx.scattering.autoionization.auger import AutoionizationState
    from atomx.scattering.dielectronic_recombination.solver import (
        dr_rate_coefficient,
        resonant_excitation_rate,
    )

    T = np.array([100.0, 500.0])
    E_d, gamma_a, g_i = 300.0, 1.0e14, 1.0
    d = AutoionizationState(label="r1", E_resonance_eV=E_d, J=1.5,
                            gamma_auger_per_s=gamma_a, a_radiative_per_s=0.0)
    saha = _saha_capture_coefficient_cm3(T)
    alpha_cap = (saha * (d.g / (2.0 * g_i)) * gamma_a
                 * np.exp(-E_d / T))

    # limit 1: Gamma_r >> Gamma_a -> the radiative branch takes the whole flux
    twin = AutoionizationState(label="r1", E_resonance_eV=E_d, J=1.5,
                               gamma_auger_per_s=gamma_a,
                               a_radiative_per_s=1.0e18)
    a_dr = dr_rate_coefficient([twin], T, initial_state_g=g_i).alpha_cm3_s
    np.testing.assert_allclose(a_dr, alpha_cap, rtol=1e-3)
    # ... and for a moderate radiative width it takes its branching fraction
    twin_mod = AutoionizationState(label="r1", E_resonance_eV=E_d, J=1.5,
                                   gamma_auger_per_s=gamma_a,
                                   a_radiative_per_s=3.0e12)
    a_dr_mod = dr_rate_coefficient([twin_mod], T,
                                   initial_state_g=g_i).alpha_cm3_s
    branch_dr = 3.0e12 / (gamma_a + 3.0e12)
    np.testing.assert_allclose(a_dr_mod, alpha_cap * branch_dr, rtol=1e-6)

    # limit 2: Gamma_a^f >> Gamma_a -> the re-Auger branch takes the whole flux
    a_re = resonant_excitation_rate([d], T, initial_state_g=g_i,
                                    partial_auger_to_final={"r1": 1.0e18})
    np.testing.assert_allclose(a_re, alpha_cap, rtol=1e-3)
    # weaker excited-channel width: its own partial width, same Saha flux
    gamma_f = 1.0e13
    a_re_mod = resonant_excitation_rate([d], T, initial_state_g=g_i,
                                        partial_auger_to_final={"r1": gamma_f})
    np.testing.assert_allclose(
        a_re_mod, alpha_cap * gamma_f / (gamma_a + gamma_f), rtol=1e-6)

    # the two branches are NOT the same number for unequal partial widths
    # (atol=0: np.allclose's 1e-8 floor would swallow these ~1e-13 rates)
    assert not np.allclose(a_dr_mod, a_re_mod, rtol=1e-3, atol=0.0)
    # no partial width -> no resonant excitation
    a0 = resonant_excitation_rate([d], T, initial_state_g=g_i,
                                  partial_auger_to_final={})
    assert np.all(a0 == 0.0)


def test_three_open_shell_solve_physical():
    """Regression: 1s 2s 2p Li-like Fe solved 6.9 keV BELOW its own
    ground state before the repeated-parentage CI fix (identical
    (occupations, J, parity) CSFs got the full configuration-average
    energy off-diagonally -> [[E,E],[E,E]] -> eigenvalues 2E and 0)."""
    from atomx import Atom
    g = Atom(Z=26, config="1s2 2s1", charge=23, engine="bspline", corrections="none")
    e = Atom(
             Z=26,
             config="1s1 2s1 2p1",
             charge=23,
             engine="bspline",
             corrections="none",
        )
    g.solve()
    e.solve()
    dE = (e._result.total_energy - g._result.total_energy) * HARTREE_EV
    # K-shell excitation of Li-like Fe: ~6.6 keV, autoionizing
    assert 6000.0 < dE < 7500.0
    I_2s = -float(g._result.orbital_energies["2s1/2"]) * HARTREE_EV
    assert dE > I_2s
    es = np.asarray(e._result.state_energies)
    # no spurious doubled/zero eigenvalues
    assert float(np.max(es) - np.min(es)) * HARTREE_EV < 200.0
