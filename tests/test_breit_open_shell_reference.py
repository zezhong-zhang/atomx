"""Full native matrices pin relative DC/Breit conventions, including interference."""
import hashlib
import json
from pathlib import Path

import numpy as np

from atomx.config.csf import CSF
from atomx.config.subshell import Subshell
from atomx.physics.csf.matrix import GridCIWorkspace, build_ci_matrix


def test_all_open_shell_breit_blocks_against_native_grasp():
    folder = Path(__file__).parent / 'fixtures/grasp_breit_open_shell'
    meta = json.loads((folder / 'metadata.json').read_text())
    assert hashlib.sha256((folder / 'native.npz').read_bytes()).hexdigest() == meta['fixture_sha256']
    with np.load(folder / 'native.npz') as data:
        keys = tuple(tuple(map(int, k)) for k in data['keys'])
        waves = {k: tuple(pq) for k, pq in zip(keys, data['waves'], strict=True)}
        ws = GridCIWorkspace(data['r'], data['jac'], data['zz'])
        csfs = [CSF([Subshell(*s) for s in c['subshells']], c['J2'], c['parity'],
                    coupling=tuple(tuple(x) for x in c['coupling'])) for c in meta['CSFs']]
        dc = build_ci_matrix(csfs, waves, keys, ws, 10, meta['N'])
        dcb = build_ci_matrix(csfs, waves, keys, ws, 10, meta['N'], include_breit=True)
        for tag, block in meta['blocks'].items():
            idx = np.ix_(block['indices'], block['indices'])
            signs = np.array(block['signs'])
            phase = signs[:, None] * signs[None, :]
            # Hartree tolerances fixed before production change: DC <0.0003meV,
            # Breit <0.00001meV; no experimental center or root selection.
            np.testing.assert_allclose(dc[idx] * phase, data[tag + '_dc_native'],
                                       atol=1.1e-8, rtol=0)
            np.testing.assert_allclose((dcb-dc)[idx] * phase, data[tag + '_breit_native'],
                                       atol=3.7e-10, rtol=0)
