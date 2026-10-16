"""Published space counts, independent promotion graph, and weighting limits."""
from math import exp
from unittest.mock import patch

import pytest

from atomx.workflows.tin_euv import (
    enumerate_tin_n4_csfs,
    plan_tin_n4_ci,
    summarize_tin_lines,
)


def test_promotion_graph_and_published_space_counts():
    # Independent breadth-first p->d / d->f promotions, not a repeated fold formula.
    frontier, visited = {(6, 2, 0)}, set()
    plan = plan_tin_n4_ci(6)
    for m in range(7):
        rows = [r for r in plan['configurations'] if r['fold'] == m]
        assert {tuple(r['occupations']) for r in rows} == frontier
        visited |= frontier
        next_set = set()
        for p, d, f in frontier:
            if p and d < 10:
                next_set.add((p-1, d+1, f))
            if d and f < 14:
                next_set.add((p, d-1, f+1))
        frontier = next_set-visited
    # Reitsma et al., PRA 114 L020802 (2026), published page 2.
    assert plan['configuration_count'] == 22
    assert plan['levels'] == 115715
    assert plan_tin_n4_ci(3)['levels'] == 5294
    assert plan_tin_n4_ci(1)['levels'] == 139
    assert plan_tin_n4_ci(0)['magnetic_states'] == 45


@pytest.mark.parametrize('fold', [-1, 7, True, 1.5])
def test_invalid_space_rejects(fold):
    with pytest.raises(ValueError):
        plan_tin_n4_ci(fold)


def test_resource_rejection_precedes_csf_expansion():
    with patch('atomx.config.csf.enumerate_jj_csfs', side_effect=AssertionError('must not enumerate')):
        with pytest.raises(ValueError, match='levels'):
            enumerate_tin_n4_csfs(6)
        with pytest.raises(ValueError, match='determinant'):
            enumerate_tin_n4_csfs(1, max_determinants=10)


def test_complete_first_fold_has_all_multiplets_and_electrons():
    csfs, folds, plan = enumerate_tin_n4_csfs(1)
    assert len(csfs) == len(folds) == plan['levels'] == 139
    assert {c.n_electrons for c in csfs} == {38}
    assert sum(c.total_J2+1 for c in csfs) == plan['magnetic_states'] == 905
    assert sum(m == 0 for m in folds) == 9
    assert all(c.parity == (-1)**m for c, m in zip(csfs, folds, strict=True))


def test_gA_energy_centroid_does_not_pick_up_boltzmann_weights():
    levels = [dict(id='l', J2=2, dominant_fold=0, excitation_eV=0.),
              dict(id='u1', J2=0, dominant_fold=1, excitation_eV=1.),
              dict(id='u2', J2=2, dominant_fold=1, excitation_eV=3.)]
    lines = [dict(upper='u1', lower='l', energy_eV=1., A_per_s=1.),
             dict(upper='u2', lower='l', energy_eV=3., A_per_s=2.)]
    cold = summarize_tin_lines(levels, lines, temperature_eV=1.)['1_to_0']
    hot = summarize_tin_lines(levels, lines, temperature_eV=100.)['1_to_0']
    assert cold['gA_sum'] == 7.
    assert cold['gA_centroid_eV'] == hot['gA_centroid_eV'] == pytest.approx(19/7)
    assert cold['Boltzmann_gA_sum'] == pytest.approx(exp(-1)+6*exp(-3))
    assert hot['Boltzmann_gA_sum'] > cold['Boltzmann_gA_sum']


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_invalid_temperature_rejects(temperature):
    with pytest.raises(ValueError):
        summarize_tin_lines([], [], temperature_eV=temperature)
