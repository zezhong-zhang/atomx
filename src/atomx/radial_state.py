"""Native radial data boundary for results and correction consumers.

The legacy result layouts are interpreted only here. Correlated CI densities
remain on their result: effective occupations must never replace coherences.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from atomx.numerics.radial_metric import RadialMetric


def _owned(value):
    if isinstance(value, np.ndarray):
        out = value.copy()
        out.setflags(write=False)
        return out
    if isinstance(value, dict):
        return MappingProxyType({k: _owned(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_owned(v) for v in value)
    return value


@dataclass(frozen=True)
class RadialState:
    representation: str
    equation: str
    metric: RadialMetric
    workspace: object
    orbitals: object
    occupied: tuple
    origin_offset: int

    def correction_workspace(self):
        # Backends historically require a dict; arrays remain read-only.
        return dict(self.workspace)


def radial_state_of(result):
    """Build an owned snapshot from any native atomic result layout."""
    bsp = getattr(result, '_bsp_data', None) or getattr(result, 'bsp_data', None)
    equation = getattr(result, '_equation', 'dirac')
    raw = getattr(result, 'orbitals_dict', None)
    occupied = []
    if bsp:
        metric = RadialMetric.bspline(bsp)
        workspace, offset, representation = bsp, 0, 'bspline'
        if raw is None:
            if callable(getattr(result, 'orbitals_by_subshell', None)):
                raw = result.orbitals_by_subshell()
            else:
                raw, counts = {}, {}
                for sh in result.subshells:
                    index = counts.get(sh.kappa, 0)
                    raw[(sh.n, sh.kappa)] = result.orbitals_raw[sh.kappa][index]
                    counts[sh.kappa] = index+1
            for sh in result.subshells:
                p, q, occ = raw[(sh.n, sh.kappa)]
                occupied.append((p, q, occ, sh.kappa, sh.label))
            raw = {k: v[:2] for k, v in raw.items()}
    else:
        r, jacobian = getattr(result, '_grid', None), getattr(result, '_difr', None)
        if r is None or jacobian is None:
            raise ValueError('result has no native radial quadrature')
        metric = RadialMetric.shooting(r, jacobian)
        offset = int(np.searchsorted(r, 0., side='right'))
        workspace = {'kind': 'radial_grid', 'gr': metric.radius[offset:],
                     'grw': metric.weights[offset:], 'integration_rule': 'slag6',
                     'jacobian': jacobian, 'origin_offset': offset}
        representation = 'grid'
        if raw is None:
            pa, qa = getattr(result, '_pa', None), getattr(result, '_qa', None)
            if pa is None or qa is None:
                raise ValueError('result has no native Dirac spinors')
            raw = {}
            for i, sh in enumerate(result.subshells):
                raw[(sh.n, sh.kappa)] = (pa[i], qa[i])
                occupied.append((pa[i][offset:], qa[i][offset:],
                                 float(sh.occupation), sh.kappa, sh.label))
    return RadialState(representation, equation, metric, _owned(workspace),
                       _owned(raw), _owned(occupied), offset)


class RadialStateMixin:
    """Shared ``radial_state`` property for native result classes.

    Single implementation of the owned, lazily cached radial snapshot;
    previously mirrored verbatim in ``results.py``,
    ``methods/dirac_fock.py`` and ``methods/csf/solver.py``.
    """

    @property
    def radial_state(self):
        """Owned native radial snapshot for correction and analysis consumers."""
        if '_radial_state' not in self.__dict__:
            self.__dict__['_radial_state'] = radial_state_of(self)
        return self.__dict__['_radial_state']


def native_radial_state(result):
    """Use the public snapshot, with a boundary adapter for legacy results."""
    if hasattr(type(result), 'radial_state'):
        return result.radial_state
    return radial_state_of(result)


def radial_workspace_of(result):
    """Workspace and origin offset; also accepts legacy metric-only results."""
    if hasattr(type(result), 'radial_state'):
        state = result.radial_state
        return state.correction_workspace(), state.origin_offset
    bsp = getattr(result, '_bsp_data', None) or getattr(result, 'bsp_data', None)
    if bsp:
        return bsp, 0
    metric = RadialMetric.shooting(result._grid, result._difr)
    offset = int(np.searchsorted(metric.radius, 0., side='right'))
    return {'kind': 'radial_grid', 'gr': metric.radius[offset:],
            'grw': metric.weights[offset:]}, offset
