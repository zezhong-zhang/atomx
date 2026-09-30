"""Continuum matrix elements must follow the electron energy on every call."""
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold


def manifold():
    r = np.geomspace(1e-5, 20., 1000)
    waves = {(2, 1): (r*r*np.exp(-2*r), np.zeros_like(r)),
             (3, -1): (r*np.exp(-r), np.zeros_like(r))}
    return ResonanceManifold(
        SimpleNamespace(subshells=[SimpleNamespace(occupation=2)]),
        [([(2, 1)], 1), ([(3, -1)], 2)],
        r_fine=r, dr_fine=np.gradient(r), orbital_map=waves,
        orbital_energies={(2, 1): -10., (3, -1): -2.},
        V_func=lambda radius: -10/radius,
    )


def test_auger_energy_dependence_and_call_order(monkeypatch):
    # A manufactured continuum isolates the linearity of the *real*
    # Coulomb/angular contractions. Doubling its amplitude quadruples Γ.
    # No assumption about a physical continuum's energy scaling is made.
    def continuum(kappa, energy, potential, charge, r, r_max, points):
        return energy*r*r*np.exp(-r/2), np.zeros_like(r)

    monkeypatch.setattr(
        'atomx.scattering.autoionization.mscheme_ci._continuum_on_grid',
        continuum)

    def width(man, energy):
        levels = man.auger_widths(1, energy, Z=10,
            bsp_data={'t': np.array([0., 20.])}, shooting_n_grid=3000)
        assert len(levels) == 1
        return levels[0].gamma_auger_per_s

    man = manifold()
    low = width(man, 100.)
    high = width(man, 200.)
    assert low > 0
    assert high/low == pytest.approx(4., rel=1e-13)
    assert width(man, 100.) == pytest.approx(low, rel=1e-13)
    assert width(manifold(), 200.) == pytest.approx(high, rel=1e-13)
