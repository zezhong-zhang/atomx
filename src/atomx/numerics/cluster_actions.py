"""Complete signed finite cluster actions without dense generator matrices.

This dictionary engine builds pure occupied-to-virtual excitation strings and
applies exact nilpotent exponentials. It does not enumerate determinant masks,
derive Wick diagrams, select physical integrals, solve CC equations, or supply
valence-universal FSCC. Reachable states can still fill the complete sector;
the caller-bound full-sector census, not observed sparsity, admits each action.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import comb, isfinite
from numbers import Complex, Integral

from atomx.numerics.fermion import apply_terms

__all__ = ["ClusterAction"]


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"integer {name} >= {minimum} required")
    return int(value)


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise ValueError(f"finite scalar {name} required")
    try:
        finite = isfinite(value.real) and isfinite(value.imag)
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"finite scalar {name} required")
    return value


@dataclass(frozen=True)
class ClusterAction:
    """Reference-normalized pure cluster strings in one complete finite sector.

    ``rank_limit`` selects complete singles/doubles/triples coordinates, not
    exponential powers. The core reference may have a different particle number
    from ``sector_electrons`` (for example N+1 electron attachment). Methods
    accept mask-to-coefficient mappings and return owned mappings with only
    exact zeros removed. Complex signed amplitudes/states are supported.

    ``term_application_cap`` is a per-call cap on the conservative full-sector
    count of operator-string/state pairs, including zero amplitudes. Every
    string has at most 2*spinor_count elementary operations. Python dictionary
    storage is O(D), not a certified RSS bound; worst-case D=comb(M,N) remains.
    No generator bank or complete determinant mask list is allocated here.
    """

    reference_mask: int
    spinor_count: int
    rank_limit: int
    sector_electrons: int
    determinant_cap: int = 256
    amplitude_cap: int = 128
    term_application_cap: int = 2_000_000
    excitations: tuple = field(init=False)
    amplitudes_by_rank: tuple = field(init=False)
    sector_dimension: int = field(init=False)
    max_order: int = field(init=False)
    _unit_terms: tuple = field(init=False, repr=False)

    def __post_init__(self):
        for name, minimum in (("spinor_count", 1), ("rank_limit", 1),
                              ("sector_electrons", 0), ("reference_mask", 0),
                              ("determinant_cap", 1), ("amplitude_cap", 1),
                              ("term_application_cap", 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.spinor_count > 64 or self.reference_mask >= 1 << self.spinor_count:
            raise ValueError("valid reference mask within at most 64 spinors required")
        if self.rank_limit not in (1, 2, 3):
            raise ValueError("connected rank limit must be 1, 2 or 3")
        if self.sector_electrons > self.spinor_count:
            raise ValueError("particle sector exceeds spinor count")
        occupied_count = self.reference_mask.bit_count()
        virtual_count = self.spinor_count-occupied_count
        ranks = tuple((r, comb(occupied_count, r)*comb(virtual_count, r))
                      for r in range(1, min(self.rank_limit, occupied_count, virtual_count)+1))
        count = sum(a for _, a in ranks)
        dimension = comb(self.spinor_count, self.sector_electrons)
        if dimension > self.determinant_cap or count > self.amplitude_cap:
            raise ValueError(f"complete sector/ranks need {dimension} states/{count} amplitudes; cap exceeded")
        object.__setattr__(self, "amplitudes_by_rank", ranks)
        object.__setattr__(self, "sector_dimension", dimension)
        # The same complete core bound works in every declared particle sector.
        # It is not the connected-rank cutoff or an observed-state truncation.
        object.__setattr__(self, "max_order", min(occupied_count, virtual_count))
        self._admit(count*dimension)
        occupied = tuple(i for i in range(self.spinor_count) if self.reference_mask & (1 << i))
        virtual = tuple(i for i in range(self.spinor_count) if not self.reference_mask & (1 << i))
        excitations, terms = [], []
        for rank, _ in ranks:
            for holes in itertools.combinations(occupied, rank):
                for particles in itertools.combinations(virtual, rank):
                    operations = (tuple((a, True) for a in particles)
                                  +tuple((i, False) for i in reversed(holes)))
                    acted = apply_terms(((1.0, operations),), {self.reference_mask: 1.0})
                    if len(acted) != 1:
                        raise RuntimeError("pure excitation reference phase census failed")
                    terms.append((1.0/next(iter(acted.values())), operations))
                    excitations.append((holes, particles))
        object.__setattr__(self, "excitations", tuple(excitations))
        object.__setattr__(self, "_unit_terms", tuple(terms))

    def _admit(self, upper):
        if upper > self.term_application_cap:
            raise ValueError(f"complete finite action needs <= {upper} term/state pairs; cap exceeded")

    def _state(self, state):
        if not isinstance(state, Mapping) or len(state) > self.sector_dimension:
            raise ValueError("finite particle-sector state mapping required")
        owned = {}
        for mask, value in state.items():
            mask = _integer(mask, "state mask", 0)
            if mask >= 1 << self.spinor_count or mask.bit_count() != self.sector_electrons:
                raise ValueError("state mask outside the complete declared particle sector")
            value = _scalar(value, "state coefficient")
            if value != 0:
                owned[mask] = value
        return owned

    def _terms(self, amplitudes):
        try:
            if len(amplitudes) != len(self.excitations):
                raise ValueError("complete ordered amplitude vector required")
        except TypeError as exc:
            raise ValueError("complete ordered amplitude vector required") from exc
        return tuple((_scalar(a, "cluster amplitude")*phase, operations)
                     for a, (phase, operations) in zip(amplitudes, self._unit_terms, strict=True))

    def _operators(self, terms):
        try:
            count = len(terms)
        except TypeError as exc:
            raise ValueError("bounded operator-term sequence required") from exc
        upper = self.sector_dimension*(2*self.max_order*len(self.excitations)+count)
        self._admit(upper)
        owned = []
        for value, operations in terms:
            value = _scalar(value, "operator coefficient")
            try:
                count_ops = len(operations)
            except TypeError as exc:
                raise ValueError("bounded operator-operation sequence required") from exc
            if count_ops > 2*self.spinor_count:
                raise ValueError("operator string exceeds declared elementary-operation bound")
            ops = []
            for bit, create in operations:
                bit = _integer(bit, "operator index", 0)
                if bit >= self.spinor_count or not isinstance(create, bool):
                    raise ValueError("in-range index and Boolean creation flag required")
                ops.append((bit, create))
            if sum(1 if create else -1 for _, create in ops) != 0:
                raise ValueError("similarity operator must conserve particle number")
            owned.append((value, tuple(ops)))
        return tuple(owned)

    def _apply(self, terms, state):
        # Verify every produced coefficient/particle sector; no epsilon filter.
        return self._state(apply_terms(terms, state))

    def _exponential(self, terms, state, sign):
        out = state.copy()
        term = state.copy()
        for order in range(1, self.max_order+1):
            term = {mask: (sign/order)*value
                    for mask, value in self._apply(terms, term).items()}
            for mask, value in term.items():
                out[mask] = out.get(mask, 0.0)+value
        return self._state(out)

    def apply(self, amplitudes, state):
        """Apply complete T1/T2[/T3] once, with a full-sector work preflight."""
        self._admit(self.sector_dimension*len(self.excitations))
        return self._apply(self._terms(amplitudes), self._state(state))

    def unit_action(self, index, state):
        """Apply one reference-normalized coordinate for exact CC derivatives."""
        index = _integer(index, "excitation index", 0)
        if index >= len(self.excitations):
            raise ValueError("excitation index outside complete ordered coordinates")
        self._admit(self.sector_dimension)
        return self._apply((self._unit_terms[index],), self._state(state))

    def exponential(self, amplitudes, state, sign):
        """Apply exact exp(sign*T); only sign +1/-1 is accepted."""
        sign = _scalar(sign, "exponential sign")
        if sign not in (-1, 1):
            raise ValueError("exponential sign must be +1 or -1")
        self._admit(self.max_order*self.sector_dimension*len(self.excitations))
        return self._exponential(self._terms(amplitudes), self._state(state), sign)

    def similarity(self, amplitudes, hamiltonian_terms, state):
        """Apply exp(-T) H exp(T), retaining all signed intermediate states.

        H is an explicit number-conserving operator sequence, including any
        constant term. No BCH depth, operator rank or intermediate excitation
        truncation is substituted for the exact finite exponentials.
        """
        operators = self._operators(hamiltonian_terms)
        terms, state = self._terms(amplitudes), self._state(state)
        right = self._exponential(terms, state, 1)
        acted = self._apply(operators, right)
        return self._exponential(terms, acted, -1)
