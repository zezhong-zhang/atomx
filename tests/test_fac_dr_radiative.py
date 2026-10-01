"""Independent FAC E1 stabilization, with complete level/channel inventories.

Common spinors and FAC bare h isolate Coulomb CI plus the photon operator.
Small FAC mixing/transition coefficient cuts must be disabled: weak-line
interference is physical even when an individual CI coefficient is tiny.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_reference_dr_radiative import (
    GROUPS,
    MODES,
    calculate,
    common_manifolds_from_data,
    compare,
)

DATA=Path(__file__).parent/'data'
REFERENCE=json.loads((DATA/'fac_dr_radiative_reference.json').read_text())


@pytest.fixture(scope='module')
def sources():
    path=DATA/REFERENCE['spinors_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==REFERENCE['spinors_sha256']
    with np.load(path) as arrays:
        yield arrays


@pytest.mark.external_fac
@pytest.mark.parametrize('case',REFERENCE['cases'],ids=lambda c:f"Z{c['z']}")
def test_complete_common_spinor_radiative_panel(case,sources):
    assert REFERENCE['controls']['common_spinors']
    assert not REFERENCE['controls']['fac_prune_mixtures']
    waves={(o['n'],o['kappa']):sources[o['array']] for o in case['orbitals']}
    bare={tuple(r['key']):r['value_Ha'] for r in case['bare_one_body']}
    manifolds=common_manifolds_from_data(waves,case['z'],8000,case['fac_alpha'],bare)
    calculated=calculate(case['z'],8000,manifolds=manifolds)
    rows=compare(case['reference'],calculated)
    assert len(rows)==237
    assert set(calculated['groups'])==set(GROUPS)
    checked={mode:0 for mode in MODES}
    for row in rows:
        # FAC's printed level energies have eight/nine significant figures.
        assert abs(row['energy_difference_eV'])<1.1e-4
        for mode in MODES:
            width=row['modes'][mode]
            for line in width['lines']:
                if line['fac_A_per_s']:
                    # Direct printed A, independently normalized in FAC.
                    assert line['ratio']==pytest.approx(1.,abs=7e-7), (row,line)
                    checked[mode]+=1
                else:
                    # Forbidden amplitudes may leave floating-point remnants
                    # when a numerical eigensolver mixes nearly pure states.
                    assert line['atomx_A_per_s']<1e-5, (row,line)
            if width['fac_width_per_s']:
                assert width['ratio']==pytest.approx(1.,abs=7e-7)
    assert checked==dict.fromkeys(MODES,393)
