"""Ownership and dependency boundaries of shared numerical contexts."""
import ast
from importlib.util import resolve_name
from pathlib import Path

import numpy as np
import pytest

from atomx.numerics.shooting.problem import ShootingRadialProblem
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.dirac_response import DiracResponse
from atomx.numerics.shooting.box_continuum import BoxDiracContinuum


def test_shared_problem_cannot_be_invalidated_by_input_mutation():
    r, dr, n = sgrid(30., 1e-7, .1, 1000)
    rv = np.full(n, -2.)
    problem = ShootingRadialProblem(r, dr, rv)
    response = DiracResponse(-2.2, -1, problem, boundary='box')
    continuum = BoxDiracContinuum(problem)
    source = np.array([r*np.exp(-r), .01*r*np.exp(-r)])
    before = response.apply(source)
    fingerprint = problem.fingerprint
    r[:] = -1
    dr[:] = 0
    rv[:] = 100
    np.testing.assert_array_equal(response.apply(source), before)
    assert response.problem is continuum.problem is problem
    assert continuum.potential is problem.spline
    assert response.weights is continuum.weights is problem.weights
    assert problem.fingerprint == fingerprint
    for values in (problem.r, problem.jacobian, problem.rv, problem.weights, *problem.spline):
        with pytest.raises(ValueError, match='read-only'):
            values[0] = 0
    with pytest.raises(ValueError, match='not both'):
        BoxDiracContinuum(problem, dr, rv)


@pytest.mark.parametrize('bad', [np.array(1.), np.zeros(12), np.arange(12)[::-1], np.full(12, np.nan)])
def test_problem_rejects_invalid_geometry(bad):
    with pytest.raises(ValueError, match='invalid shooting'):
        ShootingRadialProblem(bad, np.ones(12), np.zeros(12))


def test_radial_kernels_do_not_depend_on_scf_or_correction_policy():
    root = Path(__file__).resolve().parents[1]/'src/atomx/numerics/shooting'
    for module in ('dirac_green', 'problem', 'dirac_integrator', 'dirac_power_series',
                   'quadrature', 'grid', 'spline', 'dirac_response', 'box_continuum',
                   'dirac_orbital', 'operators', 'coulomb'):
        for node in ast.walk(ast.parse((root/f'{module}.py').read_text())):
            imports = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in imports:
                assert name.split('.')[-1] not in ('hf_shooting', 'solver', 'atom')
                assert not name.startswith(('atomx.methods', 'atomx.corrections'))


def test_whole_numerics_has_no_higher_layer_imports():
    """Resolve relative and local imports; numerics cannot import its callers."""
    root = Path(__file__).resolve().parents[1] / 'src'
    for path in (root / 'atomx/numerics').rglob('*.py'):
        module = '.'.join(path.with_suffix('').relative_to(root).parts)
        package = module.removesuffix('.__init__') if path.name == '__init__.py' else module.rsplit('.', 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                name = (resolve_name('.' * node.level + (node.module or ''), package)
                        if node.level else node.module or '')
                names = [name] + [name + '.' + a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                if any(name == target or name.startswith(target + '.') for target in
                       ('atomx.methods', 'atomx.corrections', 'atomx.properties',
                        'atomx.scattering', 'atomx.atom', 'atomx.workflows', 'atomx.web')):
                    relative = path.relative_to(root / 'atomx/numerics').as_posix()
                    pytest.fail(f'{relative}:{node.lineno} imports higher layer {name}')
