"""A reproduced wrong root may need a higher-energy named configuration."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

import scripts.repair_dhf_orbital_branch as branch


@pytest.mark.parametrize('gsz_energy,should_update', [(-100.0, True),
                                                      (-101.0, False)])
def test_repair_requires_reproduced_invalid_source_and_can_raise_energy(
        tmp_path, monkeypatch, gsz_energy, should_update):
    ledger, report = tmp_path / 'jobs.sqlite', tmp_path / 'report.json'
    config = '1s2 2p1 3p1'
    with sqlite3.connect(ledger) as db:
        db.executescript('''CREATE TABLE config_jobs(
            z INTEGER,charge INTEGER,configuration TEXT,status TEXT,
            energy_ha REAL,nuclear_mass_me REAL,starting_potential TEXT,
            nms_ha REAL,elapsed_seconds REAL,converged INTEGER,error TEXT,
            r_max_bohr REAL,grid_hmax_bohr REAL);
            INSERT INTO config_jobs VALUES(23,19,'1s2 2p1 3p1','done',
                -100,92842,'gsz',0,1,1,NULL,50,1);''')
    monkeypatch.setattr(branch, 'basis_for_job',
                        lambda *_: (object(), 50.0, 1.0))
    monkeypatch.setattr(branch, 'update_lines', lambda *_: None)
    monkeypatch.setattr(branch, 'orbital_order_valid', lambda *_: True)
    monkeypatch.setattr(branch, 'orbital_node_valid',
                        lambda result, *_: result.start != 'gsz')

    class FakeAtom:
        def __init__(self, **_kwargs):
            pass

        def solve(self, method, max_iter):
            start = method.starting_potential
            energy = (gsz_energy if start == 'gsz' else
                      -99.9 if method.scf_damping is None else
                      -99.8)
            return SimpleNamespace(start=start, converged=True,
                                   total_energy=energy, residual=1e-9,
                                   energy_components={'nms': .001})

    monkeypatch.setattr(branch, 'Atom', FakeAtom)
    result = branch.repair(ledger, 23, 19, config, report)
    with sqlite3.connect(ledger) as db:
        energy, start = db.execute('''SELECT energy_ha,starting_potential
            FROM config_jobs''').fetchone()
    assert result['source_branch_reproduced'] is should_update
    assert result['ledger_updated'] is should_update
    assert energy == pytest.approx(-99.9 if should_update else -100.0)
    assert start == ('tfda' if should_update else 'gsz')
    assert json.loads(report.read_text())['ledger_updated'] is should_update
