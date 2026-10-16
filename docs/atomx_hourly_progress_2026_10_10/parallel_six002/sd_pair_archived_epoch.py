"""Complete scalar core doubles source and PP/HH SD ladder subgraph.

Generic signed J-pair operators replace Coulomb-specific radial/angular
selection in these diagrams. This is an actual partial reduced SD consumer;
it does not supply the remaining Fock/singles/ring, valence or Sigma2 maps.
"""

from dataclasses import dataclass
from math import sqrt
from types import MappingProxyType

import numpy as np

from atomx.methods.sd_transverse import JCoupledSDInteraction


def _owned(a):
    a = np.asarray(a, dtype=np.complex128)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _sectors(interaction):
    if not isinstance(interaction, JCoupledSDInteraction) or not interaction.gates:
        raise TypeError("certified signed J-pair owner required")
    occupied = {x[:2] for x in interaction.labels[: interaction.noccupied]}
    virtual = {x[:2] for x in interaction.labels[interaction.noccupied :]}
    for block, rows in interaction.channels.items():
        o = tuple(i for i, pair in enumerate(rows) if set(pair) <= occupied)
        v = tuple(i for i, pair in enumerate(rows) if set(pair) <= virtual)
        if o and v:  # Only exactly unreachable scalar coordinates are absent.
            yield block, o, v


def plan_core_pair_ladders(interaction, *, batch=1):
    """Complete diagram payload/work before integral-block copies.

    Already-owned J integrals/Fock/h and Python/interpreter are excluded.
    This matrix representation reuses each J block across every occupied
    pair; it forms no ordered magnetic quartic amplitude tensor.
    """
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("positive complete batch required")
    coordinates = pp = hh = work = 0
    for _, occupied, virtual in _sectors(interaction):
        no, nv = len(occupied), len(virtual)
        coordinates += no * nv
        pp += nv * nv
        hh += no * no
        work += no * nv * (no + nv)
    if not coordinates:
        raise ValueError("nonempty complete scalar doubles space required")
    return dict(
        scalar_doubles=coordinates,
        pp_entries=pp,
        hh_entries=hh,
        batch=batch,
        working_numeric_payload_bound_bytes=16 * (2 * (pp + hh) + (4 * batch + 2) * coordinates),
        multiply_add_bound=batch * work,
        scope="complete source and PP/HH diagrams only; excludes owned J integral payload and interpreter RSS",
    )


@dataclass(frozen=True)
class CorePairLadderResult:
    source_Ha: np.ndarray
    particle_particle_Ha: np.ndarray
    hole_hole_Ha: np.ndarray
    doubles_linked_energy_Ha: complex
    provenance: object

    def __post_init__(self):
        for name in ("source_Ha", "particle_particle_Ha", "hole_hole_Ha"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class CorePairLadders:
    """Immutable input owner with complete J-sector matrix contractions.

    Coordinates are orthonormal scalar T2 states
    sum_M A†(virtual;JM) A(occupied;JM)|core>/sqrt(2J+1).
    Every retained J/parity channel remains. Inputs carry the complete
    noncanonical Fock/h/frozen/reference ledger unchanged; this subgraph
    does not pretend to evaluate the still-unimplemented mixed-rank terms.
    """

    def __init__(self, interaction, *, memory_bytes=128 * 1024**2):
        if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes < 1:
            raise ValueError("positive memory cap required")
        plan = plan_core_pair_ladders(interaction)
        if plan["working_numeric_payload_bound_bytes"] > memory_bytes:
            raise MemoryError("complete core ladder payload exceeds cap")
        self.interaction, self.cap = interaction, memory_bytes
        self.plan = MappingProxyType(plan)
        pp, hh, source, layout, coordinates, cursor = {}, {}, [], [], [], 0
        for block, occupied, virtual in _sectors(interaction):
            matrix = interaction.blocks_Ha[block]
            pp[block] = _owned(matrix[np.ix_(virtual, virtual)])
            hh[block] = _owned(matrix[np.ix_(occupied, occupied)])
            rows = interaction.channels[block]
            no, nv = len(occupied), len(virtual)
            source.extend((sqrt(block[0] + 1) * matrix[np.ix_(virtual, occupied)]).ravel())
            layout.append((block, occupied, virtual, cursor, cursor + no * nv))
            coordinates.extend((block, rows[v], rows[o]) for v in virtual for o in occupied)
            cursor += no * nv
        self.pp_Ha, self.hh_Ha = MappingProxyType(pp), MappingProxyType(hh)
        self.source_Ha, self.layout, self.coordinates = _owned(source), tuple(layout), tuple(coordinates)
        self.provenance = MappingProxyType(dict(
            interaction_identity=interaction.identity,
            normal_ordering="all-occupied HH uses core normal-ordered V; all-virtual PP has no occupied contraction",
            coordinate_metric="orthonormal scalar paired T2 excitation states",
            implemented_diagrams=("complete scalar doubles source", "complete particle-particle ladder", "complete hole-hole ladder", "linked doubles energy contraction"),
            remaining="Fock/singles/mixed-rank/ring and remaining core terms; valence and Sigma2; no complete SD solve",
            highest_level_reached=False,
        ))

    def evaluate(self, amplitudes):
        """Return complete signed separated diagram arrays, no solving/fitting."""
        t = np.asarray(amplitudes)
        if t.shape != self.source_Ha.shape or not np.all(np.isfinite(t)):
            raise ValueError("complete finite scalar doubles vector required")
        pp, hh = np.zeros_like(self.source_Ha), np.zeros_like(self.source_Ha)
        for block, occupied, virtual, first, last in self.layout:
            a = t[first:last].reshape(len(virtual), len(occupied))
            pp[first:last] = (self.pp_Ha[block] @ a).ravel()
            hh[first:last] = (a @ self.hh_Ha[block]).ravel()
        return CorePairLadderResult(
            self.source_Ha, pp, hh, complex(np.vdot(self.source_Ha, t)), self.provenance
        )
