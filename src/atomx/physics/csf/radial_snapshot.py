"""Exact-CI Coulomb integral reuse within one immutable B-spline snapshot."""

import copy
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def _contents(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, Mapping):
        return tuple((key, _contents(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_contents(item) for item in value)
    if isinstance(value, (str, int, float, bool, np.number, type(None))):
        return value
    raise TypeError(f"unsupported CI snapshot workspace entry {type(value).__name__}")


def _freeze(value):
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _potential_contents(potential, workspace):
    if potential is None:
        return None
    return np.asarray(
        [potential(r) for r in workspace["gr"].ravel()], dtype=float
    ).tobytes()


class CIRadialSnapshot:
    """Reuse existing signed radial evaluator across charge/CSF spaces.

    Copies orbitals and B-spline geometry into read-only arrays and freezes
    nuclear values at the integration nodes. ``solve`` consumes those owned
    inputs directly. External matrix builds check exact caller contents, Z,
    ns, nuclear callable identity and its values at the quadrature nodes;
    a mismatch raises. The sampled nuclear operator is not an off-grid
    continuum potential. The existing evaluator's bounded caches and
    arithmetic are unchanged; no pair interchange or truncation is added.

    This caches only Dirac-Coulomb radial values. Per-build one-body shifts
    and Breit operators remain separate. Grid/SLAG6 snapshots are unsupported.
    The object is local to a serial snapshot, not a process-wide cache.
    """

    def __init__(self, orbitals, workspace, Z, ns, *, nuclear_potential=None,
                 pair_potential_cache_size=None):
        from atomx.numerics.bspline.integrals import SampledNuclearPotential
        from atomx.physics.csf.matrix import GridCIWorkspace, _radial_evaluator

        if isinstance(workspace, GridCIWorkspace):
            raise ValueError("CI radial snapshots require a B-spline workspace")
        self._Z, self._ns = Z, ns
        self._workspace_contents = _contents(workspace)
        self._orbital_contents = _contents(orbitals)
        self._nuclear_potential = nuclear_potential
        self._potential_contents = _potential_contents(nuclear_potential, workspace)
        self._frozen_potential = (
            None
            if nuclear_potential is None
            else SampledNuclearPotential(
                workspace["gr"],
                np.frombuffer(self._potential_contents, dtype=float).reshape(
                    workspace["gr"].shape
                ),
            )
        )
        self._workspace = _freeze(workspace)
        self._orbitals = _freeze(orbitals)
        self._evaluate = _radial_evaluator(
            self._orbitals,
            self._workspace,
            Z,
            ns,
            nuclear_potential=self._frozen_potential,
            pair_potential_cache_size=pair_potential_cache_size,
        )
        self._builds = 0

    def evaluator_for(self, orbitals, workspace, Z, ns, *, nuclear_potential=None):
        if (
            orbitals is self._orbitals
            and workspace is self._workspace
            and nuclear_potential is self._frozen_potential
            and Z == self._Z
            and ns == self._ns
        ):
            # Only immutable owned data can use this path. Mutable caller
            # arrays and nuclear closures retain the full content checks.
            self._builds += 1
            return self._evaluate
        if (
            Z != self._Z
            or ns != self._ns
            or nuclear_potential is not self._nuclear_potential
            or _contents(orbitals) != self._orbital_contents
            or _contents(workspace) != self._workspace_contents
            or _potential_contents(nuclear_potential, workspace)
            != self._potential_contents
        ):
            raise ValueError(
                "CI radial snapshot does not match orbitals/workspace/nuclear Hamiltonian"
            )
        self._builds += 1
        return self._evaluate

    def solve(self, csfs, orbital_basis, **options):
        """Solve from immutable owned orbitals/geometry/nuclear quadrature.

        The sampled operator is exact on this B-spline integration grid.
        A changed caller model needs a new snapshot. No mutable replacement
        for the owned Hamiltonian inputs can be passed through this method.
        """
        from atomx.methods.ci import ci_solve

        return ci_solve(
            csfs,
            self._orbitals,
            orbital_basis,
            self._workspace,
            self._Z,
            self._ns,
            nuclear_potential=self._frozen_potential,
            radial_snapshot=self,
            **options,
        )

    def transverse_evaluator(self, orbital_energies, *, cache_size=128):
        """Real transverse exchange from this snapshot's owned frozen inputs.

        The caller owns the returned bounded evaluator and its energy map;
        changing orbital frequencies creates a new evaluator. This does not
        insert transverse terms into the snapshot's Coulomb cache.
        """
        from atomx.physics.csf.transverse import radial_evaluator

        return radial_evaluator(self._orbitals, self._workspace, orbital_energies,
                                cache_size=cache_size)

    def diagnostics(self):
        info = self._evaluate.cache_info()
        pairs = self._evaluate.pair_cache_info()
        return dict(
            matrix_builds=self._builds,
            radial_cache_hits=info.hits,
            radial_cache_misses=info.misses,
            radial_cache_entries=info.currsize,
            radial_cache_capacity=info.maxsize,
            pair_potential_cache_hits=pairs.hits,
            pair_potential_cache_misses=pairs.misses,
            pair_potential_cache_entries=pairs.currsize,
            pair_potential_cache_capacity=pairs.maxsize,
            model="immutable common B-spline Dirac-Coulomb orbital snapshot",
        )
