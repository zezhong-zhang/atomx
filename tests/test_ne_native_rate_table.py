"""Independent physical and graph checks of the regenerated native Ne table."""
import json
import math
from pathlib import Path

import pytest

from atomx.constants import HARTREE_EV
from atomx.xfel import CompiledRateTable
from atomx.xfel.cascade import _table_key

HERE = Path(__file__).resolve().parent / 'data'
BASE = {(1, -1): 2, (2, -1): 2, (2, 1): 2, (2, -2): 4}


@pytest.fixture(scope='module')
def table():
    return json.loads((HERE / 'ne_native_1000eV.json').read_text())


def test_complete_native_graph_and_empty_hole_decay(table):
    graph = CompiledRateTable(table, BASE)
    assert len(table['configs']) == 134
    assert not graph.missing_keys and not table['failed']
    empty_k = {s: q for s, q in BASE.items() if s != (1, -1)}
    row = table['configs'][_table_key(empty_k)]
    assert len(row['auger']) == 6
    assert all(c['hole'] == [1, -1] and c['rate'] > 0 for c in row['auger'])


def test_real_kll_population_and_width(table):
    khole = dict(BASE)
    khole[(1, -1)] = 1
    row = table['configs'][_table_key(khole)]
    assert len(row['auger']) == 6
    # XATOM independent three-family sum: population double counting
    # would exceed this tolerance by factors, not a few percent.
    total = sum(c['rate'] for c in row['auger'])
    assert 0.9 < total / 4.001968962964592e14 < 1.2
    assert all(700 < c['energy_eV'] < 830 for c in row['auger'])


def test_native_energies_close_every_auger_edge(table):
    energies = {key: r['energy_hartree'] for key, r in table['configs'].items()}
    for row in table['configs'].values():
        config = {tuple(map(int, text.split(','))): q for text, q in row['occupations'].items()}
        for ch in row['auger']:
            d = dict(config)
            for sh, delta in ((tuple(ch['hole']), 1), (tuple(ch['s1']), -1), (tuple(ch['s2']), -1)):
                d[sh] = d.get(sh, 0) + delta
            expected = (row['energy_hartree'] - energies[_table_key(d)]) * HARTREE_EV
            assert math.isclose(ch['energy_eV'], expected, rel_tol=1e-12)


def test_edge_closure_is_physical_not_missing_table_row(table):
    state = {(1, -1): 2, (2, -1): 2, (2, 1): 2}
    row = table['configs'][_table_key(state)]
    assert row['thresholds_eV']['1s1/2'] > 1000
    assert row['sigma_cm2']['1,-1'] == 0
    assert row['sigma_cm2']['2,1'] > 0
