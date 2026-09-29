"""Public imports and result conversion must not trigger unrelated work."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.methods._result_adapters import to_scf_result
from atomx.state_protocols import grid_data


def test_result_coercion_never_solves_an_atom():
    calls = []
    atom = SimpleNamespace(_result=None, solve=lambda: calls.append('solve'))
    with pytest.raises(ValueError, match='solve.*explicitly'):
        to_scf_result(atom, caller='property')
    assert calls == []


def test_result_coercion_requires_convergence_and_preserves_identity():
    result = SimpleNamespace(converged=False, _bsp_data={})
    with pytest.raises(ValueError, match='converged'):
        to_scf_result(SimpleNamespace(_result=result), caller='property')
    result.converged = True
    assert to_scf_result(SimpleNamespace(_result=result), caller='property') is result
    assert to_scf_result(result, caller='property') is result


def test_legacy_protocol_exports_preserve_identity():
    from atomx import state_protocols
    from atomx.scattering import protocols
    for name in protocols.__all__:
        assert getattr(protocols, name) is getattr(state_protocols, name)


def test_grid_accessor_preserves_native_shooting_quadrature():
    from atomx.numerics.shooting.grid import sgrid
    from atomx.numerics.radial_metric import RadialMetric
    r, dr, _ = sgrid(20., 1e-7, .1, 1000)
    result = SimpleNamespace(_grid=r, _difr=dr)
    workspace = grid_data(result)
    metric = RadialMetric.shooting(r, dr)
    positive = r > 0
    assert workspace['kind'] == 'radial_grid'
    np.testing.assert_array_equal(workspace['gr'], r[positive])
    np.testing.assert_array_equal(workspace['grw'], metric.weights[positive])


def test_grid_accessor_uses_public_radial_state():
    workspace = {'kind': 'test-public-state'}
    class Result:
        @property
        def radial_state(self):
            return SimpleNamespace(correction_workspace=lambda: workspace, origin_offset=0)
    assert grid_data(Result()) is workspace


def test_public_imports_do_not_load_renderers_or_optional_backends():
    source = Path(__file__).resolve().parents[1] / 'src'
    env = dict(os.environ, PYTHONPATH=str(source))
    code = '''
import sys
import atomx
assert 'atomx.atom' not in sys.modules
assert 'atomx.scattering' not in sys.modules
from atomx import Atom, Basis, BreitPauli, HartreeFockSlater, viz, compare
assert callable(compare)
assert Atom and Basis and BreitPauli and HartreeFockSlater
assert 'mpl' in dir(viz)
assert not any(k.split('.')[0] in {'matplotlib', 'plotly', 'manim', 'pyscf'} for k in sys.modules)
assert 'atomx.viz.mpl' not in sys.modules
from atomx.viz import mpl
assert callable(mpl.plot_spectrum)
from atomx import what_changed, theory_ladder
assert callable(atomx.compare) and callable(what_changed) and callable(theory_ladder)
'''
    subprocess.run([sys.executable, '-c', code], cwd='/tmp', env=env, check=True,
                   capture_output=True, text=True)


def test_radial_workspace_compatibility_exports_preserve_identity():
    from atomx.corrections import workspace as legacy
    from atomx.numerics import radial_workspace
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(radial_workspace, name)


def test_shooting_drivers_run_without_method_imports():
    """Numerical drivers and physical operators cannot reach into methods."""
    source = Path(__file__).resolve().parents[1] / 'src'
    code = '''
import importlib.abc
import sys
class RejectMethods(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'atomx.methods' or fullname.startswith('atomx.methods.'):
            raise AssertionError('Numerical engine imported methods: ' + fullname)
sys.meta_path.insert(0, RejectMethods())
from atomx.config.subshell import Subshell
from atomx.numerics.shooting.local import ShootingSolver
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
from atomx.numerics.shooting.electron_space import ShootingSpace
shells = [Subshell(1, -1, 2)]
seed = ShootingSolver(2, shells, n_grid=800).solve()
result = ShootingHFSolver(2, shells, n_grid=800).solve(seed=seed)
assert seed.converged and result.converged
assert result.waves.orbitals
assert ShootingSpace().n_grid > 0
'''
    subprocess.run([sys.executable, '-c', code], cwd='/tmp',
                   env=dict(os.environ, PYTHONPATH=str(source)), check=True,
                   capture_output=True, text=True)


def test_photoionization_public_entry_and_integer_energy_grid(monkeypatch):
    from atomx.scattering.photoionization import direct
    from atomx.scattering.photoionization import photoionization_cross_section
    monkeypatch.setattr(direct, 'solve_continuum',
                        lambda *args, **kwargs: (np.ones(2), np.zeros(2), 1.))
    monkeypatch.setattr(direct, 'dipole_matrix_element', lambda *args: 1.)
    args = (np.ones(2), np.zeros(2), -1, -.5)
    expected = direct.photoionization_cross_section(*args, np.array([1., 2.]), 1, {})
    actual = photoionization_cross_section(*args, [1, 2], 1, {})
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual > 0)
    assert actual.dtype.kind == 'f'
