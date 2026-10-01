"""Matched Fe dr33 frozen-CI calculation: native orbitals, continua and photons.

The independent FAC reference has the same separate hole/parity CI spaces,
point nucleus, infinite mass and corrections disabled. Both codes optimize
local ground potentials independently. Aggregate gates are centered on FAC,
not on historical Atomx biases; individual weak widths remain model-sensitive.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from atomx import Atom, Basis, Sampling
from atomx.constants import HARTREE_EV
from atomx.scattering.autoionization import dr_calculate, dr_rate_coefficient
from atomx.scattering.autoionization.auger import AutoionizationState
from atomx.scattering.autoionization.mscheme_ci import manifold_from_configs

REFERENCE = json.loads((Path(__file__).parent/'data/fac_fe_dr33_matched.json').read_text())
T_MK = np.array([1., 5., 10., 50.])


def state(row):
    return AutoionizationState(
        label=str(row['ilev']), E_resonance_eV=row['energy_eV'],
        J=row['two_J']/2, gamma_auger_per_s=row['auger_per_s'],
        a_radiative_per_s=row['radiative_per_s'])


@pytest.fixture(scope='module')
def dr_atom():
    atom = Atom(26, charge=16, config='1s2 2s2 2p6', engine='shooting',
        exchange='gu', corrections='none', nuclear='point',
        basis=Basis(sampling=Sampling(n_grid=4000)))
    atom.solve(tol=1e-10, max_iter=150)
    assert atom.converged
    return atom


@pytest.fixture(scope='module')
def dr_states(dr_atom):
    atom = dr_atom
    rows = []
    for group, configs in REFERENCE['configs'].items():
        manifold, workspace = manifold_from_configs(atom.result, configs)
        tm = 1
        while manifold.determinants(tm):
            for level in manifold.dr_levels(tm, Z=26, bsp_data=workspace):
                assert level.radiative_channels
                assert all(c.photon_approximation == 'finite_wavelength'
                           for c in level.radiative_channels)
                assert level.a_radiative_per_s == sum(
                    c.A_per_s for c in level.radiative_channels)
                rows.append(dict(group=group, ilev=len(rows), two_J=tm,
                    energy_eV=level.E_res_au*HARTREE_EV,
                    auger_per_s=level.gamma_auger_per_s,
                    radiative_per_s=level.a_radiative_per_s))
            tm += 2
    return rows


@pytest.mark.external_fac
def test_public_driver_preserves_native_model_and_resolved_rates(dr_atom, dr_states, monkeypatch):
    result = dr_atom._result
    original = result._pa.copy()
    def unexpected_solve(*args, **kwargs):
        pytest.fail('DR must reuse the solved orbital model')
    monkeypatch.setattr(dr_atom, 'solve', unexpected_solve)
    temperatures = T_MK*1e6*8.617333262e-5
    public = dr_calculate(dr_atom, '1s2 2s2 2p6 3s1', T_eV=temperatures)
    expected = dr_rate_coefficient([state(r) for r in dr_states], temperatures,
                                   initial_state_g=1.)
    assert dr_atom._result is result
    np.testing.assert_array_equal(result._pa, original)
    assert len(public.resonances) == 237
    assert len({r.label for r in public.resonances}) == 237
    np.testing.assert_allclose(public.alpha_cm3_s, expected.alpha_cm3_s, rtol=2e-12)
    np.testing.assert_allclose(sum(public.per_resonance.values()), public.alpha_cm3_s,
                               rtol=1e-14)
    assert all(r.channels == {'ground': r.gamma_auger_per_s} for r in public.resonances)


def test_nonlocal_shooting_is_not_replaced_by_a_local_model():
    from types import SimpleNamespace
    result = SimpleNamespace(_grid=np.array([0., 1.]), _rv=None, _model='hf')
    with pytest.raises(NotImplementedError, match='nonlocal shooting HF'):
        manifold_from_configs(result, ['2p5 3s2'])


@pytest.mark.external_fac
def test_complete_symmetry_inventory_and_energies(dr_states):
    reference = REFERENCE['resonances']
    assert len(dr_states) == len(reference) == 237
    def inventory(rows):
        return Counter((r['group'], r['two_J']) for r in rows)
    assert inventory(dr_states) == inventory(reference)
    def key(r):
        return (r['group'], r['two_J'], r['energy_eV'])
    for actual, expected in zip(sorted(dr_states, key=key), sorted(reference, key=key), strict=True):
        assert abs(actual['energy_eV']-expected['energy_eV']) < .5


@pytest.mark.external_fac
def test_alpha_dr_vs_matched_fac(dr_states):
    temperatures = T_MK*1e6*8.617333262e-5
    calculated = dr_rate_coefficient([state(r) for r in dr_states], temperatures,
                                     initial_state_g=1.)
    reference = dr_rate_coefficient([state(r) for r in REFERENCE['resonances']],
                                    temperatures, initial_state_g=1.)
    np.testing.assert_allclose(calculated.alpha_cm3_s, reference.alpha_cm3_s, rtol=.02, atol=0)


@pytest.mark.external_fac
@pytest.mark.parametrize('field', ['auger_per_s', 'radiative_per_s'])
def test_total_widths_vs_matched_fac(dr_states, field):
    assert all(np.isfinite(row[field]) and row[field] >= 0 for row in dr_states)
    calculated = sum(row[field] for row in dr_states)
    reference = sum(row[field] for row in REFERENCE['resonances'])
    assert calculated == pytest.approx(reference, rel=.02)
