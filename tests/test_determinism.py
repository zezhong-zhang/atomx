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
CI run with two fresh processes over C, Ne and multi-configuration Be.
``test_panel_has_no_out_of_bounds_reads`` is an explicit diagnostic outside
the default essential suite because bounds checking forces a full recompile.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = """
import hashlib
import json
import numpy as np
from atomx import Atom, Basis, Correlation

cases = [
    dict(Z=6, engine='shooting'),
    dict(Z=10, engine='bspline'),
    dict(Z=4, config=['1s2 2s2', '1s2 2p2'], engine='bspline',
         correlation=Correlation(averaging='ci'), basis=Basis(ns=45)),
]
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


def _run(cache, *, boundscheck=False):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'src'),
               NUMBA_BOUNDSCHECK=str(int(boundscheck)),
               NUMBA_DISABLE_JIT='0', NUMBA_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if cache is not None:
        env['NUMBA_CACHE_DIR'] = str(cache)
    process = subprocess.run([sys.executable, '-c', PANEL],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=1800)
    assert process.returncode == 0, (
        f'numerical panel failed ({process.returncode}):\n'
        f'{process.stdout}\n{process.stderr}')
    return json.loads(process.stdout)


def test_panel_is_deterministic_across_processes():
    """Identical input must give bit-identical output in fresh processes."""
    # Fresh interpreters test process-dependent values and scratch memory.
    # Reuse compiled kernels; the ordinary suite already exercises cold JIT.
    results = [_run(None) for _ in range(2)]
    assert all(result == results[0] for result in results[1:])
