"""Independent FAC CI assembly and radial-operator evidence.

The reference contains FAC matrices, radial integrals and unmodified source
spinors. Common integrals isolate angular assembly; common spinors separately
test the production radial kernels and public CI builder. No fitted shifts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
from atomx.physics.csf.matrix import (
    _group_csf_indices,
    _radial_evaluator,
    _reset_memo,
    build_ci_matrix,
)
from benchmarks.bench_reference_ci_operator import (
    angular_blocks,
    common_grid,
    compare_spectra,
    freeze,
    one_body_constant_shift,
    spectrum,
)

DATA = Path(__file__).parent/'data'
REFERENCE = json.loads((DATA/'fac_ci_operator.json').read_text())
CASES = REFERENCE['cases']


@pytest.fixture(scope='module')
def spinors():
    path = DATA/'fac_ci_spinors.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE['spinors_sha256']
    with np.load(path) as source:
        yield source


def reference_spectrum(case):
    return {(b['two_J'], b['parity']): np.linalg.eigvalsh(b['matrix_Ha']) for b in case['blocks']}


@pytest.mark.external_fac
@pytest.mark.parametrize('case', CASES, ids=lambda c: f"{c['sequence']}-Z{c['z']}")
def test_exact_angular_ci_against_independent_fac(case):
    blocks, keys, _ = angular_blocks(case['configurations'])
    radial = {freeze(row['key']): row['value_Ha'] for row in case['radial']}
    assert set(keys) == radial.keys()
    differences = compare_spectra(spectrum(blocks, radial), reference_spectrum(case))
    counts = {'helium_like': 98, 'carbon_like': 20, 'neon_like': 37}
    assert sum(len(b['difference_Ha']) for b in differences) == counts[case['sequence']]
    # Absolute rounding bound for the largest ~1e4-Ha matrices in this panel.
    for block in differences:
        np.testing.assert_allclose(block['difference_Ha'], 0., atol=2e-10, rtol=0)


@pytest.mark.external_fac
@pytest.mark.parametrize('case', CASES, ids=lambda c: f"{c['sequence']}-Z{c['z']}")
def test_common_spinors_through_public_ci_builder(case, spinors):
    assert REFERENCE['controls']['disable_fac_yk_cache']
    sources = {(o['n'], o['kappa']): spinors[o['array']] for o in case['orbitals']}
    waves, workspace = common_grid(sources, case['z'], 8000, alpha=case['fac_alpha'])
    csfs = enumerate_jj_csfs_many(case['configurations'])
    matrix = build_ci_matrix(csfs, waves, get_shared_orbital_basis(csfs), workspace,
                             case['z'], workspace.n)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-11, rtol=0)
    blocks, keys, _ = angular_blocks(case['configurations'])
    shifts = {key: one_body_constant_shift(waves, workspace, key, 1/case['fac_alpha']) for key in keys}
    from atomx.angular.csf_hamiltonian import exact_block_matrix
    adjusted = {}
    for sym, indices in _group_csf_indices(csfs):
        h = matrix[np.ix_(indices, indices)]
        # The constants differ; the exact fixed-spinor Dirac c-dependence
        # aligns them analytically and leaves Coulomb integrals unchanged.
        correction = exact_block_matrix(blocks[sym], shifts.__getitem__)
        adjusted[sym] = np.linalg.eigvalsh(h+correction)
    for block in compare_spectra(adjusted, reference_spectrum(case)):
        np.testing.assert_allclose(block['difference_Ha'], 0., atol=2e-8, rtol=0)

    _reset_memo()
    evaluate = _radial_evaluator(waves, workspace, case['z'], workspace.n)
    for row in case['radial']:
        key = freeze(row['key'])
        value = evaluate(key)+shifts[key]
        # Check individual integrals as well as eigenvalues; weak off-diagonal
        # errors can be hidden by small configuration mixing coefficients.
        tolerance = 8e-8 if key[0] == 'h' else 8e-11
        assert value == pytest.approx(row['value_Ha'], abs=tolerance, rel=0), key


def test_spectrum_comparison_rejects_missing_levels():
    with pytest.raises(ValueError, match='dimensions'):
        compare_spectra({(0, 1): np.array([-1.])}, {(0, 1): np.array([-1., 0.])})
    with pytest.raises(ValueError, match='symmetries'):
        compare_spectra({(0, 1): np.array([-1.])}, {(2, 1): np.array([-1.])})
