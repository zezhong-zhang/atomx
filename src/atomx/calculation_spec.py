"""Immutable, engine-independent description of a resolved calculation.

Solvers resolve their defaults before entering this boundary. The record is
attached to the result before corrections and preserved by public dispatch.
It describes model and discretization; iteration convergence is separate.
"""
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import math
import numpy as np


def _freeze(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return MappingProxyType({'shape': tuple(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()})
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if is_dataclass(value):
        return _freeze({f.name: getattr(value, f.name) for f in fields(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if callable(value):
        return MappingProxyType({'callable': f'{getattr(value, "__module__", type(value).__module__)}.{getattr(value, "__qualname__", type(value).__qualname__)}',
            'reproducibility': 'external callable; record sampled potential separately'})
    raise TypeError(f'cannot record calculation option {type(value).__name__}')


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class CalculationSpec:
    Z: float
    method: str
    engine: str
    physics: Mapping
    state: Mapping
    numerics: Mapping
    corrections: Mapping
    schema_version: int = 1

    def __post_init__(self):
        if not math.isfinite(self.Z) or self.Z <= 0:
            raise ValueError('calculation nuclear charge must be positive')
        if self.engine not in ('shooting', 'bspline', 'analytic'):
            raise ValueError('unknown radial engine')
        for name in ('physics', 'state', 'numerics', 'corrections'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def to_dict(self):
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


def resolved_correction_spec(stack):
    if stack is None:
        return {}
    options = {f.name: getattr(stack, f.name) for f in fields(stack)
               if not f.name.startswith('_')}
    options['custom'] = tuple(dict(name=c.name, options=dict(vars(c))) for c in stack._custom)
    return options


def resolved_solver_spec(solver, *, method, engine, controls):
    """Normalize native solver inputs at the single result-recording boundary."""
    aw = float(getattr(solver, 'aw', 0.) or 0.)
    nuclear = 'fermi' if aw > .5 else getattr(solver, 'nuclear', None) or 'point'
    profile = getattr(solver, 'nuclear_profile', None)
    if profile is None and nuclear == 'point':
        profile = dict(model='point', rms_radius_fm=0.)
    elif profile is None and method in ('DHF', 'MCDHF'):
        from atomx.potentials.nuclear import fermi_params
        a, c = fermi_params(solver.Z, A=aw if aw > .5 else None)
        profile = dict(model='fermi', diffuseness_bohr=float(a),
                       half_density_radius_bohr=float(c))
    physics = dict(equation=getattr(solver, 'equation', 'dirac'),
        nuclear=nuclear, nuclear_profile=profile,
        exchange=getattr(solver, 'exchange', 'hf'),
        nuclear_atomic_weight=getattr(solver, 'aw', None),
        screening_potential=getattr(solver, 'screening_potential', None),
        electron_projector=getattr(solver, 'variational_projector', 'self_consistent'),
        rest_energy='subtracted', units='atomic')
    shells = getattr(solver, 'subshells', ())
    state = dict(subshells=tuple((s.n, s.kappa, float(s.occupation)) for s in shells))
    configs = getattr(solver, 'jj_configs', None)
    if configs is not None:
        state['jj_configurations'] = tuple(dict(weight=float(c.weight),
            subshells=tuple((s.n, s.kappa, float(s.occupation)) for s in c.subshells))
            for c in configs)
    csfs = getattr(solver, 'csfs', None)
    if csfs is not None:
        state['csfs'] = tuple(dict(subshells=tuple((s.n, s.kappa, float(s.occupation))
             for s in c.subshells), J2=c.total_J2, parity=c.parity) for c in csfs)
        state['objective'] = getattr(solver, 'radial_optimization', 'mcdhf')
    numerics = {name: getattr(solver, name) for name in (
        'ns', 'ks', 'r_min', 'r_max', 'r_inner', 'n_grid', 'grid_hi', 'grid_he', 'grid_hmax',
        'use_physics_grid', 'kinetic_balance', 'constraint_mode', 'orbital_preconditioner')
        if hasattr(solver, name)}
    if engine == 'shooting' and hasattr(solver, 'shooting_space'):
        numerics['electron_space'] = solver.shooting_space
        for unused in ('ns', 'ks', 'r_min', 'use_physics_grid', 'grid_hi', 'grid_he', 'grid_hmax'):
            numerics.pop(unused, None)
    if engine == 'bspline' and getattr(solver, 'bsp_data', None):
        numerics['basis_knots'] = solver.bsp_data['t']
    numerics.update(controls)
    corrections = resolved_correction_spec(getattr(solver, 'correction_stack', None))
    if hasattr(solver, 'breit_mode'):
        corrections['breit_stage'] = 'CI' if solver._breit_in_ci else 'post_scf'
    return CalculationSpec(float(solver.Z), method, engine, physics, state, numerics, corrections)
