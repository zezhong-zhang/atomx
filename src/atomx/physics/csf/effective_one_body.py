"""Bound real scalar effective one-body operators for native exact CI.

The producer owns the physical approximation and its P/Q partition. This
module binds signed matrix entries to spinors and a frozen-core ledger; it
does not generate MBPT diagrams or solve energy-dependent eigenproblems.
"""
import hashlib
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np

from atomx.physics.csf.radial_snapshot import _contents, _potential_contents


def _digest(value):
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(repr(_contents(value)).encode()).hexdigest()


def spinor_identity(P, Q, kappa):
    """Include radial coefficients and their phase, not merely orbital names."""
    return _digest((P, Q, int(kappa)))


@dataclass(frozen=True)
class CIReferenceBinding:
    """Immutable producer identity for B-spline core and external spinors."""
    external: tuple
    core: tuple
    geometry: str
    Z: int
    ns: int
    nuclear: str

    @classmethod
    def capture(cls, targets, core_orbitals, workspace, Z, ns, nuclear_potential=None):
        from atomx.physics.csf.matrix import GridCIWorkspace
        if isinstance(workspace, GridCIWorkspace):
            raise ValueError("effective CI operators require a B-spline workspace")
        return cls(tuple(spinor_identity(P, Q, k) for P, Q, k, *_ in targets),
                   tuple((key, spinor_identity(*value, key[1]))
                         for key, value in sorted(core_orbitals.items())),
                   _digest(workspace), Z, ns,
                   _digest(_potential_contents(nuclear_potential, workspace)))

    def validate(self, keys, orbitals, workspace, Z, ns, nuclear_potential):
        if any(key not in orbitals for key in keys) or any(key not in orbitals for key, _ in self.core):
            raise ValueError("effective operator producer orbitals missing")
        actual = type(self).capture(
            [(*orbitals[key], key[1]) for key in keys],
            {key: orbitals[key] for key, _ in self.core},
            workspace, Z, ns, nuclear_potential)
        if actual != self:
            raise ValueError("effective operator producer basis/core/nuclear reference mismatch")


@dataclass(frozen=True, init=False, eq=False)
class CIEffectiveOneBody:
    """Σ₁(E*) with exact signed off-diagonals in one declared frozen reference.

    ``correlated_core`` lists the closed orbitals integrated out by the
    producer. Every CI configuration must retain these filled; opening a
    hole requires rebuilding the complementary Q space and its operators.
    The energy argument is frozen. CI diagonalisation alone is not Dyson
    self-consistency, all-order SD, full CI+MBPT or a dressed observable.
    """

    def __init__(self, orbital_keys, matrix_Ha, orbitals, workspace, Z, ns, *,
                 correlated_core, reference_identity, theory, energy_Ha,
                 nuclear_potential=None, producer_binding=None):
        keys = tuple(tuple(k) for k in orbital_keys)
        matrix = np.asarray(matrix_Ha)
        if (not keys or len(set(keys)) != len(keys)
                or any(len(k) != 2 or any(not isinstance(v, (int, np.integer)) for v in k)
                       or k[0] < 1 or k[1] == 0 for k in keys)
                or matrix.shape != (len(keys), len(keys))
                or not np.isrealobj(matrix) or not np.all(np.isfinite(matrix))):
            raise ValueError("require unique orbital keys and a finite real square matrix")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("effective one-body matrix must be exactly symmetric")
        if any(matrix[i, j] != 0 for i, a in enumerate(keys)
               for j, b in enumerate(keys) if a[1] != b[1]):
            raise ValueError("scalar effective one-body operator cannot mix kappa")
        core = dict(correlated_core)
        if any(key not in orbitals or q != 2*abs(key[1]) for key, q in core.items()):
            raise ValueError("correlated core ledger must contain filled present orbitals")
        if set(keys) & set(core):
            raise ValueError("valence self-energy cannot also act on the correlated core")
        if (not isinstance(reference_identity, str) or not reference_identity.strip()
                or not isinstance(theory, str) or not theory.strip()
                or np.ndim(energy_Ha) != 0 or not np.isrealobj(energy_Ha)
                or not np.isfinite(energy_Ha)):
            raise ValueError("require theory, reference identity and finite real energy argument")
        binding = CIReferenceBinding.capture(
            [(*orbitals[key], key[1]) for key in keys],
            {key: orbitals[key] for key in core}, workspace, Z, ns, nuclear_potential)
        if producer_binding is not None and binding != producer_binding:
            raise ValueError("effective operator producer basis/core/nuclear reference mismatch")
        owned = np.frombuffer(np.asarray(matrix, dtype=float).tobytes(), dtype=float).reshape(matrix.shape)
        values = dict(orbital_keys=keys, matrix_Ha=owned, correlated_core=MappingProxyType(core),
                      reference_identity=reference_identity, theory=theory, energy_Ha=float(energy_Ha),
                      binding=binding, _orbital_context=_digest(orbitals))
        values['_lookup'] = MappingProxyType({("h", a[0], a[1], b[0]): float(owned[i, j])
                                        for i, a in enumerate(keys) for j, b in enumerate(keys)
                                        if a[1] == b[1]})
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def evaluator_for(self, csfs, orbitals, workspace, Z, ns, *, nuclear_potential=None):
        self.binding.validate(self.orbital_keys, orbitals, workspace, Z, ns, nuclear_potential)
        if _digest(orbitals) != self._orbital_context:
            raise ValueError("effective CI operator orbital context mismatch")
        for csf in csfs:
            occupations = {(sh.n, sh.kappa): sh.occupation for sh in csf.subshells}
            if any(occupations.get(key, 0) != q for key, q in self.correlated_core.items()):
                raise ValueError("CI opens a correlated-core hole: rebuild complementary P/Q operators to avoid double counting")
        return lambda key: self._lookup.get(key, 0.)

    def provenance(self):
        return dict(theory=self.theory, reference_identity=self.reference_identity,
                    energy_Ha=self.energy_Ha, units="Hartree", frozen_energy_argument=True,
                    orbital_keys=self.orbital_keys,
                    correlated_core=[dict(orbital_key=key, occupation=q)
                                     for key, q in sorted(self.correlated_core.items())],
                    producer_binding=asdict(self.binding), matrix_sha256=_digest(self.matrix_Ha))
