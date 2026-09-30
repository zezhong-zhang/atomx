"""Checkpoints retain the native metric and open-shell correction identity."""
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from atomx.config.jj_coupling import generate_jj_average
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
from atomx.results import ShootingResult
from atomx.corrections.stack import CorrectionStack, finalize_with_corrections


@pytest.mark.parametrize('equation', ['dirac', 'schrodinger'])
def test_local_checkpoint_loads_without_importing_methods(tmp_path, equation):
    """The archive reader needs data contracts, not an installed SCF driver."""
    from atomx.config.subshell import Subshell
    from atomx.numerics.shooting.local import ShootingSolver

    state = ShootingSolver(2, [Subshell(1, -1, 2)], n_grid=800,
                           equation=equation).solve()
    assert state.converged
    assert state._jj_configs is None
    path = state.save(tmp_path / 'local')
    restored = ShootingResult.load(path)
    assert restored._jj_configs is None
    assert restored.total_energy == state.total_energy
    assert restored.theory_level == state.theory_level
    np.testing.assert_array_equal(restored._pa, state._pa)
    np.testing.assert_array_equal(restored._qa, state._qa)
    np.testing.assert_array_equal(restored._rv_nuc, state._rv_nuc)
    assert state._rv_nuc is not None
    source = Path(__file__).resolve().parents[1] / 'src'
    code = '''
import importlib.abc
import sys
class RejectMethods(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'atomx.methods' or fullname.startswith('atomx.methods.'):
            raise AssertionError('Archive reader imported solver layer: ' + fullname)
sys.meta_path.insert(0, RejectMethods())
from atomx.results import ShootingResult
state = ShootingResult.load(sys.argv[1])
assert state.converged
assert state.waves.orbitals
assert state.radial_state.metric.weights.size == len(state._grid)
assert state.calculation_spec is not None
assert state.theory_level
'''
    subprocess.run([sys.executable, '-c', code, str(path)], cwd='/tmp',
                   env=dict(os.environ, PYTHONPATH=str(source)), check=True,
                   capture_output=True, text=True)


def test_open_shell_roundtrip_preserves_metric_corrections_and_restart(tmp_path):
    shells, configs = generate_jj_average('1s2 2s2 2p2')
    solver = ShootingHFSolver(6, shells, jj_configs=configs, n_grid=800,
                              nuclear='fermi', aw=12)
    state = solver.solve(max_iter=200, tol=1e-10)
    assert state.converged
    assert [asdict(c) for c in state._jj_configs] == [asdict(c) for c in configs]
    assert state._jj_configs is not configs
    restored = ShootingResult.load(state.save(tmp_path/'bare'))
    assert restored.calculation_spec.to_dict() == state.calculation_spec.to_dict()
    assert [asdict(c) for c in restored._jj_configs] == [asdict(c) for c in configs]
    for name in ('_grid', '_difr', '_pa', '_qa', '_rv_nuc'):
        np.testing.assert_array_equal(getattr(restored,name),getattr(state,name))
    np.testing.assert_array_equal(restored.radial_state.metric.weights,
                                  state.radial_state.metric.weights)
    stack = CorrectionStack(se_backend='mohr', breit_mode='dbsr_hf',
                            nuclear_mass_me=21868., strict=True)
    a,b = stack.apply(state),stack.apply(restored)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].energy_shift == b[name].energy_shift
    warm = solver.solve(seed=restored, max_iter=100, tol=1e-10)
    assert warm.converged
    assert [asdict(c) for c in warm._jj_configs] == [asdict(c) for c in configs]
    assert stack.apply(warm)['breit'].energy_shift != 0.
    assert abs(warm.total_energy-state.total_energy)<1e-8
    finalize_with_corrections(restored, stack)
    corrected = ShootingResult.load(restored.save(tmp_path/'corrected'))
    assert corrected.total_energy == restored.total_energy
    assert corrected.correction_application == restored.correction_application
    assert corrected._correction_results == restored._correction_results
    # Validation failure must not replace a previously valid checkpoint.
    original = (tmp_path/'bare.h5').read_bytes()
    restored._qa = restored._qa[:, :-1]
    with pytest.raises(ValueError, match='spinors'):
        restored.save(tmp_path/'bare')
    assert (tmp_path/'bare.h5').read_bytes() == original
