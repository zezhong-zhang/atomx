"""Exact support counting versus all-coordinate enumeration; no physics screening."""

from itertools import product

import numpy as np
import pytest

from atomx.methods.fscc import export_integrals, load_integrals, resource_estimate
from atomx.physics.csf.spinor_integrals import SpinorIntegrals, packed_size


def integrals(occupied, virtual):
    labels = tuple((n, k, m) for n, k in occupied+virtual
                   for m in range(-2*abs(k)+1, 2*abs(k), 2))
    o = sum(2*abs(k) for _, k in occupied)
    n = len(labels)
    return SpinorIntegrals(labels, o, np.zeros((n, n)), np.zeros((n, n)),
        np.zeros(packed_size(n)), 0., 0., {})


def brute(left, right):
    def quantum(row):
        return (sum(t[2] for t in row),
                (-1)**sum(t[1] if t[1] > 0 else -t[1]-1 for t in row))
    return sum(quantum(a) == quantum(b) for a in product(*left) for b in product(*right))


@pytest.mark.parametrize('model', ['CCSD', 'CCSDT'])
@pytest.mark.parametrize('sector', ['0h0p', '0h1p', '1h0p'])
def test_all_coordinate_enumeration_matches_rank_and_every_quartic_partition(model, sector):
    obj = integrals([(1, -1), (2, 1)], [(2, -1), (3, 1)])
    o, v = obj.labels[:4], obj.labels[4:]
    active = () if sector == '0h0p' else ((4, 5) if sector == '0h1p' else (0, 1))
    a = tuple(obj.labels[i] for i in active)
    core = brute([o], [v])+brute([o, o], [v, v])
    valence = 0
    if sector == '0h1p':
        valence = brute([a], [v])+brute([a, o], [v, v])
    elif sector == '1h0p':
        valence = brute([a], [o])+brute([a, v], [o, o])
    if model == 'CCSDT':
        core += brute([o]*3, [v]*3)
        if sector == '0h1p':
            valence += brute([a, o, o], [v]*3)
        elif sector == '1h0p':
            valence += brute([a, v, v], [o]*3)
    quartic = brute([obj.labels]*2, [obj.labels]*2)
    result = resource_estimate(obj, model=model, sector=sector, active_count=len(active),
                               active_indices=active, tensor_symmetry='axial')
    assert result['axial_core_coordinates'] == core
    assert result['axial_valence_coordinates'] == valence
    assert result['axial_all_ordered_quartic_coordinates'] == quartic
    assert result['estimated_working_bytes'] == 512*(core+valence)+32*quartic+obj.dense_eri_bytes
    default = resource_estimate(obj, model=model, sector=sector, active_count=len(active))
    assert default == resource_estimate(obj, model=model, sector=sector,
                                       active_count=len(active), tensor_symmetry='C1')


def test_expanded_range_is_explicit_complete_and_persisted(tmp_path):
    obj = integrals([(1, -1)], [(2, -1), (7, 6), (7, -7)])
    with pytest.raises(ValueError, match='range'):
        export_integrals(obj, tmp_path/'old', tensor_symmetry='axial')
    assert not (tmp_path/'old').exists()
    result = export_integrals(obj, tmp_path/'new', tensor_symmetry='axial', axial_max_omega_x2=126)
    assert result['tensor_symmetry']['max_omega_x2'] == 126
    restored = load_integrals(tmp_path/'new')
    np.testing.assert_array_equal(restored.packed_eri_Ha, obj.packed_eri_Ha)
    assert restored.labels == obj.labels
    with pytest.raises(ValueError, match='indices'):
        resource_estimate(obj, sector='0h1p', active_count=2, tensor_symmetry='axial',
                          axial_max_omega_x2=126)
    with pytest.raises(ValueError, match='multiplets'):
        resource_estimate(obj, sector='0h1p', active_count=1, active_indices=(2,),
                          tensor_symmetry='axial', axial_max_omega_x2=126)
    with pytest.raises(ValueError, match='range'):
        export_integrals(obj, tmp_path/'fake', tensor_symmetry='axial', axial_max_omega_x2=126.)


@pytest.mark.parametrize('maximum', [62, 126])
def test_mixed_parity_raw_heff_numbers_follow_the_full_generated_table(maximum):
    from atomx.methods.fscc import _model_matrix
    # Enumerate the actual upstream table order independently of the parser's
    # numeric formula. Include repeated-radial blocks and both signs/parities.
    table = [(m*sign, p) for p in (1, -1) for m in range(1, maximum, 2) for sign in (1, -1)]
    symmetry = np.array([(3, -1), (-1, 1), (1, -1), (-1, 1)])
    numbers = [table.index(tuple(q)) for q in symmetry]
    blocks = {('0h0p', 1): np.array([[-2.]])}
    expected = np.zeros((4, 4), complex)
    for number in set(numbers):
        indices = [i for i, n in enumerate(numbers) if n == number]
        raw = np.arange(len(indices)**2).reshape(len(indices), len(indices)) + number*1j
        blocks['0h1p', number-min(numbers)+1] = raw
        expected[np.ix_(indices, indices)] = raw
    np.testing.assert_array_equal(_model_matrix(blocks, '0h1p', tuple(range(4)), symmetry,
        axial_max_omega_x2=maximum), expected)
