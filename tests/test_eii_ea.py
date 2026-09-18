"""Excitation-autoionization channel in total EII (review M6, EA half).

He-like Fe: 1s2 + e -> 2s 2p + e (double excitation, 13.5 keV) lies
above the 1s ionization limit (8.8 keV), so it autoionizes and adds to
net ionization.  EA turns on at the excitation threshold and its
magnitude is a DW collision strength times a branching ratio <= 1.
(The 3-open-shell Li-like case below was unblocked by the 2026-07-16
repeated-parentage CI fix.)
"""

from __future__ import annotations

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
    g = Atom(Z=26, config="1s2", charge=24, numerics="bspline", breit=False, qed=False)
    e = Atom(Z=26, config="2s1 2p1", charge=24, numerics="bspline", breit=False, qed=False)
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


def test_resonant_excitation_rate_limits():
    """RE rate: detailed-balance composer over capture x re-Auger."""
    from atomx.scattering.autoionization.auger import AutoionizationState
    from atomx.scattering.dielectronic_recombination.solver import (
        dr_rate_coefficient,
        resonant_excitation_rate,
    )

    d = AutoionizationState(label="r1", E_resonance_eV=300.0, J=1.5,
                            gamma_auger_per_s=1e14,
                            a_radiative_per_s=0.0)
    T = [100.0, 500.0]
    # partial width to the excited channel equal to the radiative width
    # of a DR twin -> alpha_RE == alpha_DR with Gamma_r <-> Gamma_A^f
    twin = AutoionizationState(label="r1", E_resonance_eV=300.0, J=1.5,
                               gamma_auger_per_s=1e14,
                               a_radiative_per_s=3e12)
    a_dr = dr_rate_coefficient([twin], T, initial_state_g=1.0).alpha_cm3_s
    a_re = resonant_excitation_rate(
        [d], T, initial_state_g=1.0,
        partial_auger_to_final={"r1": 3e12})
    assert np.allclose(a_re, a_dr, rtol=1e-12)
    # no partial width -> zero
    a0 = resonant_excitation_rate([d], T, initial_state_g=1.0,
                                  partial_auger_to_final={})
    assert np.all(a0 == 0.0)


def test_three_open_shell_solve_physical():
    """Regression: 1s 2s 2p Li-like Fe solved 6.9 keV BELOW its own
    ground state before the repeated-parentage CI fix (identical
    (occupations, J, parity) CSFs got the full configuration-average
    energy off-diagonally -> [[E,E],[E,E]] -> eigenvalues 2E and 0)."""
    from atomx import Atom
    g = Atom(Z=26, config="1s2 2s1", charge=23, numerics="bspline", breit=False, qed=False)
    e = Atom(Z=26, config="1s1 2s1 2p1", charge=23, numerics="bspline", breit=False, qed=False)
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
