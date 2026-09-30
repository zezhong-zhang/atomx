"""Numerical reproducibility and bounds-safety gates.

Two defects motivated these tests, both of which shipped and neither of
which any existing test could see:

* an out-of-bounds read in a numba convolution kernel that produced a
  *nondeterministic* NaN (~30 % of fresh processes) in the README's own
  ``Atom(Z=4, configs=[...])`` example and in ``Atom(Z=10, numerics=
  "bspline")``;
* value drift between fresh interpreter processes, which a digest over
  several subprocesses detects.

``test_panel_is_deterministic_across_processes`` is cheap enough for every
CI run.  ``test_panel_has_no_out_of_bounds_reads`` is marked ``slow``
because numba bounds checking forces a full recompile; CI runs it in a
dedicated job (see ``.github/workflows/test.yml``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PANEL = """
import hashlib
import json
import sys
import numpy as np
from atomx import Atom, Basis, Correlation

cases = [
    dict(Z=6, engine='shooting'),
    dict(Z=10, engine='bspline'),
    dict(Z=4, config=['1s2 2s2', '1s2 2p2'], engine='bspline',
         correlation=Correlation(averaging='ci'), basis=Basis(ns=45)),
]
if sys.argv[1] == 'full':
    cases += [dict(Z=98, engine='shooting')]
records = []
for settings in cases:
    atom = Atom(**settings).solve()
    assert atom.converged
    assert np.isfinite(atom.total_energy)
    assert all(np.isfinite(value) for value in atom.orbital_energies.values())
    digest = hashlib.sha256()
    for label, orbital in sorted(atom.orbitals.items()):
        digest.update(label.encode())
        for array in (orbital.P, orbital.Q):
            if array is not None:
                assert np.all(np.isfinite(array))
                digest.update(np.asarray(array).tobytes())
    records.append(dict(z=atom.Z, result=type(atom.result).__name__,
        energy=atom.total_energy, orbitals=atom.orbital_energies,
        wave_sha256=digest.hexdigest()))
print(json.dumps(records, sort_keys=True))
"""


def _run(panel, cache, *, boundscheck=False):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'src'), NUMBA_CACHE_DIR=str(cache),
               NUMBA_BOUNDSCHECK=str(int(boundscheck)),
               NUMBA_DISABLE_JIT='0', NUMBA_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    process = subprocess.run([sys.executable, '-c', PANEL, panel],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=1800)
    assert process.returncode == 0, (
        f'numerical panel failed ({process.returncode}):\n'
        f'{process.stdout}\n{process.stderr}')
    return json.loads(process.stdout)


def test_panel_is_deterministic_across_processes(tmp_path):
    """Identical input must give bit-identical output in fresh processes."""
    panel = os.environ.get('ATOMX_TEST_DETERMINISM_PANEL', 'quick')
    repeats = int(os.environ.get('ATOMX_TEST_DETERMINISM_REPEATS', '3'))
    assert panel in ('quick', 'full') and repeats >= 2
    results = [_run(panel, tmp_path/'numba') for _ in range(repeats)]
    assert all(result == results[0] for result in results[1:])


@pytest.mark.slow
def test_panel_has_no_out_of_bounds_reads(tmp_path):
    """Run the panel with NUMBA_BOUNDSCHECK=1 — catches OOB kernel reads.

    With bounds checking off such a read returns heap garbage: harmless when
    the value is finite (it is multiplied by a structurally zero rkb entry)
    and a poisoned ``nan`` integral otherwise, which is exactly how the
    nondeterministic-NaN bug stayed invisible.
    """
    assert len(_run('full', tmp_path/'numba-bounds', boundscheck=True)) == 4
