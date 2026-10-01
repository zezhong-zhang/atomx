"""Method dispatch must preserve controls and reject unsupported requests."""
import numpy as np
import pytest

from atomx import Atom, Basis
from atomx.config.subshell import Subshell
from atomx.methods.dirac_fock import DiracFock
from atomx.methods.mcdhf import MCDHF
from atomx.types import NuclearModel


def test_dirac_method_preserves_explicit_basis_and_box():
    atom = Atom(1, config='1s1', basis=Basis(ns=40, r_max=23.))
    result = atom.solve(DiracFock(nuclear=NuclearModel.POINT),
                        max_iter=2)
    assert result._bsp_data['ns'] == 40
    assert result._bsp_data['t'][-1] == pytest.approx(23.)
    assert result.calculation_spec.numerics['r_max'] == pytest.approx(23.)


def test_dirac_method_screening_changes_energy_by_constant_shift():
    method = DiracFock(nuclear=NuclearModel.POINT)
    controls = dict(ns=40, use_physics_grid=False, r_max=23., max_iter=2)
    base = method.solve(1, [Subshell(1, -1, 1)], **controls)
    shifted = method.solve(1, [Subshell(1, -1, 1)],
                           screening_potential=lambda r: np.full_like(r, .01),
                           **controls)
    assert shifted.total_energy - base.total_energy == pytest.approx(.01, abs=1e-9)


def test_dirac_automatic_box_metadata_matches_grid():
    result = DiracFock(nuclear=NuclearModel.POINT).solve(
        1, [Subshell(1, -1, 1)], max_iter=2)
    assert result.calculation_spec.numerics['r_max'] == result._bsp_data['t'][-1]
    assert result.calculation_spec.numerics['kinetic_balance'] is True


@pytest.mark.parametrize('method, configuration', [
    (DiracFock(), [Subshell(1, -1, 1)]), (MCDHF(), ['1s2']),
])
def test_method_rejects_unknown_or_inapplicable_controls(method, configuration):
    with pytest.raises(TypeError, match='unexpected keyword'):
        method.solve(2, configuration, misspelled_tolerance=1e-8)
    with pytest.raises(ValueError, match='grid shape controls'):
        method.solve(2, configuration, use_physics_grid=False, grid_hi=.1)


def test_mcdhf_atom_dispatch_preserves_basis_box_and_screening(monkeypatch):
    import atomx.methods.mcdhf as module
    captured = {}
    sentinel = object()
    class Solver:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def solve(self, **kwargs):
            captured.update(kwargs)
            return sentinel
    monkeypatch.setattr(module, 'MCDHFSolver', Solver)
    atom = Atom(2, config=['1s2', '1s1 2s1'],
                basis=Basis(ns=47, r_max=31.))
    def screening(r):
        return np.zeros_like(r)
    monkeypatch.setattr(atom, '_screening_shift', lambda: screening)
    assert atom.solve(MCDHF(), max_iter=7, tol=2e-7) is sentinel
    assert captured['ns'] == 47
    assert captured['r_max'] == 31.
    assert captured['use_physics_grid'] is False
    assert captured['screening_potential'] is screening
    assert captured['max_iter'] == 7
    assert captured['scf_tol'] == 2e-7
