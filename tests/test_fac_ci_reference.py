"""Independent Ne-like FAC levels/rates with explicit frozen-CI contracts.

The tolerances bound differences between independently optimized local
potentials. They do not certify a common-spinor operator to those tolerances.
"""
import json
from pathlib import Path

import pytest

from atomx import Atom, Basis, Sampling
from atomx.constants import HARTREE_EV
from atomx.properties.transitions.ci_lines import ci_line_strengths
from atomx.properties.transitions.rates import einstein_A_eL
from benchmarks.bench_reference_ci_transitions import compare_levels


REFERENCE = json.loads((Path(__file__).parent/'data/fac_ne_like_ci_reference.json').read_text())


@pytest.mark.external_fac
@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: f"Z{c['z']}")
def test_frozen_ci_complete_levels_and_both_photon_operators(case):
    assert REFERENCE['controls']['fac_configuration_shifts'] is False
    z = case['z']
    atom = Atom(z, charge=z-10, config='1s2 2s2 2p6', engine='shooting', exchange='gu',
                nuclear='point', corrections='none', basis=Basis(sampling=Sampling(n_grid=4000)))
    atom.solve(tol=1e-10, max_iter=150)
    assert atom.converged
    photon_rates = {}
    for name, mode in (('long_wavelength', 1), ('finite_wavelength', 0)):
        refs = case['lines'][name]
        assert len(refs) == 5
        assert all(r['mode'] == mode and r['gauge'] == 'length' for r in refs)
        fac_by_id = {r['upper']: r for r in refs}
        checked_levels = checked_lines = 0
        for config, expected_count in (('2p5 3s1', 4), ('2p5 3d1', 12)):
            ls = ci_line_strengths(atom.result, config, '2p6', photon_approximation=name)
            assert len(ls.levels_i) == expected_count and len(ls.levels_f) == 1
            matches = compare_levels(case['levels'][config], ls.levels_i, ls.levels_f[0].energy_au)
            assert len(matches) == expected_count
            ax_by_id = {r.upper.index: r for r in ls.lines}
            for match in matches:
                assert abs(match['difference_eV']) < 0.75, match
                checked_levels += 1
                if match['two_J'] != 2:
                    continue
                ref, ax = fac_by_id[match['fac_level']], ax_by_id[match['atomx_level']]
                strength = ref['reduced_matrix_element']**2/2
                # Independent printed A also protects the degeneracy/normalization
                # convention. Allow printed rounding and different physical constants.
                rate = einstein_A_eL(ref['DELTA_E_eV']/HARTREE_EV, strength, 1, ref['twoj_upper'])
                assert rate == pytest.approx(ref['A_per_s'], rel=3e-6)
                assert ax.A_per_s/ref['A_per_s'] == pytest.approx(1., abs=.05)
                assert ax.S/strength == pytest.approx(1., abs=.05)
                photon_rates[name, match['fac_level']] = (ax.A_per_s, ref['A_per_s'])
                checked_lines += 1
        assert (checked_levels, checked_lines) == (16, 5)
    for upper in fac_by_id:
        finite_ax, finite_fac = photon_rates['finite_wavelength', upper]
        static_ax, static_fac = photon_rates['long_wavelength', upper]
        assert finite_ax/static_ax == pytest.approx(finite_fac/static_fac, abs=.002)


def test_level_matching_rejects_missing_same_symmetry_state():
    from atomx.properties.transitions.ci_lines import CILevel
    fac = [dict(ILEV=i, TWOJ=2, P=1, ENERGY_eV=e, config='2p5 3s1')
           for i, e in enumerate((10., 12.))]
    incomplete = [CILevel('2p5 3s1', 0, 2, -1, 10./HARTREE_EV, 'test')]
    with pytest.raises(ValueError):
        compare_levels(fac, incomplete, 0.)
