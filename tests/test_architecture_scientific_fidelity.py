"""Scientific request fidelity at public method/property boundaries."""
from types import SimpleNamespace

import pytest

from atomx import Atom, Basis, BreitPauli, HartreeFockSlater, Sampling
from atomx.config.subshell import Subshell
from atomx.properties.transitions.calculator import (
    TransitionCalculator,
    TransitionResult,
)
from atomx.types import NuclearModel


@pytest.mark.parametrize('method', [HartreeFockSlater, BreitPauli])
def test_local_methods_preserve_sampling_nucleus_and_radial_payload(method):
    atom = Atom(2, config='1s2', basis=Basis(sampling=Sampling(n_grid=1200), r_max=50.))
    result = atom.solve(method(nuclear=NuclearModel.POINT))
    spec = result.calculation_spec
    assert spec.physics['nuclear'] == 'point'
    assert spec.numerics['n_grid'] == 1200
    assert spec.numerics['r_max'] == 50.
    assert result.radial_state.metric.weights.size == len(result._grid)
    assert result.subshells and result._pa is not None and result._difr is not None
    if method is BreitPauli:
        assert result.theory_level.name == 'BREIT_PAULI'
        assert 'hydrogenic' in result.method_name



def calculator():
    return TransitionCalculator(SimpleNamespace(converged=True, Z=2,
        _csf_result=None, _result=None, _config_strings=['1s2', '1s1 2p1']))


def test_ci_request_cannot_silently_run_replacement_or_fallback():
    with pytest.raises(NotImplementedError, match='explicit solved'):
        calculator().compute_all(line_resolution='ci')


def test_only_explicit_uta_fallback_records_reason(monkeypatch):
    calc = calculator()
    row = TransitionResult('lower', 'upper', 1., 455., 'E1', 1., 1., 1., 1.)
    monkeypatch.setattr(calc, '_compute_single_config_transitions', lambda *args: [row])
    assert calc.compute_all(line_resolution='ci', fallback='uta') == [row]
    assert row.provenance['requested_resolution'] == 'ci'
    assert row.provenance['actual_resolution'] == 'uta'
    assert 'explicit solved' in row.provenance['fallback_reason']


def test_invalid_calculations_are_not_converted_into_fallback(monkeypatch):
    calc = calculator()
    def invalid(*args):
        raise ValueError('invalid physical input')
    monkeypatch.setattr(calc, '_compute_ci_transitions', invalid)
    with pytest.raises(ValueError, match='invalid physical input'):
        calc.compute_all(line_resolution='ci', fallback='uta')


def test_mcdhf_reuses_its_existing_states(monkeypatch):
    calc = calculator()
    calc.atom._csf_result = object()
    sentinel = object()
    monkeypatch.setattr(calc, '_compute_mcdhf_transitions', lambda *args: sentinel)
    assert calc.compute_all(line_resolution='ci') is sentinel


@pytest.mark.parametrize('equation', ['Dirac', 'relativistic', '', None])
def test_local_solver_rejects_unknown_equation(equation):
    from atomx.numerics.shooting.local import ShootingSolver
    with pytest.raises(ValueError, match='equation must be'):
        ShootingSolver(1, [Subshell(1, -1, 1)], equation=equation)


@pytest.mark.parametrize('controls', [dict(max_iter=0), dict(max_iter=-1),
                                     dict(tol=0.), dict(tol=float('nan'))])
def test_local_solver_rejects_invalid_iteration_controls_before_field(controls):
    from atomx.numerics.shooting.local import ShootingSolver
    def unexpected_field(r):
        pytest.fail('Invalid controls must be rejected before field evaluation')
    solver = ShootingSolver(1, [Subshell(1, -1, 1)],
                            screening_potential=unexpected_field)
    with pytest.raises(ValueError, match='max_iter must be positive'):
        solver.solve(**controls)
