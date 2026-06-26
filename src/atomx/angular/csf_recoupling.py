"""Multi-subshell CSF recoupling helpers.

This module bridges the single-subshell RCFP primitives in
``atomx.angular.rcfp`` to full configuration-state-function coefficients.
It starts with scalar same-subshell operators, where the recoupling is exact
and no inter-shell Wigner phase is introduced.  Higher-rank and multi-subshell
operator products will extend the same data model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from atomx.angular.rcfp import (
    ReducedSubshellState,
    reduced_annihilation_coeff,
    reduced_creation_coeff,
    reduced_pair_annihilation_coeff,
    reduced_pair_creation_coeff,
    reduced_w_coeff,
    reduced_wxw0_coeff,
)
from atomx.angular.mscheme import subshell_j_counts_mscheme
from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.wigner import wigner_3j, wigner_6j


@dataclass(frozen=True)
class RecoupledCSFState:
    """Reduced product-state label for a multi-subshell CSF.

    ``subshells`` are ordered.  The ordering is part of the fermionic phase
    convention and must match the ordering used by the configuration/CSF
    generator.
    """

    subshells: tuple[ReducedSubshellState, ...]
    total_J2: int
    parity: int = 1
    label: str = ""


@dataclass(frozen=True)
class TwoSubshellCoupledState:
    """Reduced state for two coupled subshell terms.

    This is the first explicit intermediate-coupling scaffold for full CSF
    recoupling.  It carries the two subshell reduced states and their coupled
    total angular momentum.  Later matrix-element routines can use this type
    to apply Wigner-6j recoupling without guessing state ordering.
    """

    left: ReducedSubshellState
    right: ReducedSubshellState
    total_J2: int
    parity: int = 1
    label: str = ""


@dataclass(frozen=True)
class ChainCoupledCSFState:
    """Left-associated multi-subshell coupled CSF state.

    ``cumulative_J2[i]`` is the angular momentum after coupling subshells
    ``0..i``.  The convention is therefore
    ``(((s0 s1)J01 s2)J012 ... sn)J``.  This compact tree is enough to express
    FAC-style ordered recoupling coefficients while keeping the current API
    independent of the eventual CSF generator internals.
    """

    subshells: tuple[ReducedSubshellState, ...]
    cumulative_J2: tuple[int, ...]
    parity: int = 1
    label: str = ""

    @property
    def total_J2(self) -> int:
        return self.cumulative_J2[-1]


def angular_momenta_couple(j_left2: int, j_right2: int, total_J2: int) -> bool:
    """Triangle/parity test in twice-angular-momentum notation."""
    return (
        abs(j_left2 - j_right2) <= total_J2 <= j_left2 + j_right2
        and (j_left2 + j_right2 + total_J2) % 2 == 0
    )


@lru_cache(maxsize=None)
def allowed_coupled_J2(left_J2: int, right_J2: int) -> tuple[int, ...]:
    """Allowed total ``2J`` values for coupling two angular momenta."""
    return tuple(range(abs(left_J2 - right_J2), left_J2 + right_J2 + 1, 2))


def make_two_subshell_coupled_state(left: ReducedSubshellState,
                                    right: ReducedSubshellState,
                                    total_J2: int,
                                    parity: int = 1,
                                    label: str = "") -> TwoSubshellCoupledState:
    """Construct a validated two-subshell coupled state."""
    if not angular_momenta_couple(left.J2, right.J2, total_J2):
        raise ValueError(
            f"Cannot couple 2J={left.J2} and 2J={right.J2} to 2J={total_J2}"
        )
    return TwoSubshellCoupledState(
        left=left,
        right=right,
        total_J2=total_J2,
        parity=parity,
        label=label,
    )


def make_chain_coupled_state(
    subshells: tuple[ReducedSubshellState, ...],
    cumulative_J2: tuple[int, ...],
    parity: int = 1,
    label: str = "",
) -> ChainCoupledCSFState:
    """Construct a validated left-associated coupled CSF chain."""
    if not subshells:
        raise ValueError("A coupled CSF chain requires at least one subshell")
    if len(cumulative_J2) != len(subshells):
        raise ValueError("cumulative_J2 must contain one entry per subshell")
    if cumulative_J2[0] != subshells[0].J2:
        raise ValueError(
            f"First cumulative 2J must equal first subshell 2J={subshells[0].J2}"
        )
    for i in range(1, len(subshells)):
        if not angular_momenta_couple(cumulative_J2[i - 1], subshells[i].J2, cumulative_J2[i]):
            raise ValueError(
                "Invalid cumulative coupling at subshell "
                f"{i}: 2J={cumulative_J2[i - 1]} with 2j={subshells[i].J2} "
                f"cannot give 2J={cumulative_J2[i]}"
            )
    return ChainCoupledCSFState(
        subshells=tuple(subshells),
        cumulative_J2=tuple(cumulative_J2),
        parity=parity,
        label=label,
    )


def chain_state_from_recoupled_csf(
    state: RecoupledCSFState,
    cumulative_J2: tuple[int, ...],
) -> ChainCoupledCSFState:
    """Attach an explicit left-associated coupling tree to a recoupled CSF."""
    chain = make_chain_coupled_state(
        state.subshells,
        cumulative_J2,
        parity=state.parity,
        label=state.label,
    )
    if chain.total_J2 != state.total_J2:
        raise ValueError(
            f"Chain total 2J={chain.total_J2} does not match CSF 2J={state.total_J2}"
        )
    return chain


def _chain_prefix(state: ChainCoupledCSFState) -> ChainCoupledCSFState:
    return ChainCoupledCSFState(
        subshells=state.subshells[:-1],
        cumulative_J2=state.cumulative_J2[:-1],
        parity=state.parity,
        label=state.label,
    )


def _chain_spectators_match(
    bra: ChainCoupledCSFState,
    ket: ChainCoupledCSFState,
    active: int,
) -> bool:
    if len(bra.subshells) != len(ket.subshells):
        return False
    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        if i != active and b != k:
            return False
    return True


def chain_w_reduced(
    bra: ChainCoupledCSFState,
    ket: ChainCoupledCSFState,
    active: int,
    K2: int,
) -> float:
    """Reduced ``W^K`` matrix element on a left-associated CSF chain.

    This recursively applies the same Wigner-6j spectator recoupling used by
    the two-subshell helpers, so it works for any active subshell position in
    a validated ordered CSF chain.
    """
    n = len(bra.subshells)
    if n != len(ket.subshells) or n == 0:
        return 0.0
    if not (0 <= active < n) or bra.parity != ket.parity:
        return 0.0
    if not _chain_spectators_match(bra, ket, active):
        return 0.0
    if n == 1:
        return reduced_w_coeff(bra.subshells[0], ket.subshells[0], K2)

    bra_prefix = _chain_prefix(bra)
    ket_prefix = _chain_prefix(ket)
    bra_right = bra.subshells[-1]
    ket_right = ket.subshells[-1]

    if active == n - 1:
        if bra_prefix != ket_prefix:
            return 0.0
        red = reduced_w_coeff(bra_right, ket_right, K2)
        if abs(red) < 1e-14:
            return 0.0
        six = wigner_6j(
            bra_right.J2, bra.total_J2, bra_prefix.total_J2,
            ket.total_J2, ket_right.J2, K2,
        )
        if abs(six) < 1e-14:
            return 0.0
        phase = _phase_from_twice(bra_right.J2 + bra_prefix.total_J2 + ket.total_J2 + K2)
        return float(phase * ((bra.total_J2 + 1) * (ket.total_J2 + 1)) ** 0.5 * six * red)

    if bra_right != ket_right:
        return 0.0
    red = chain_w_reduced(bra_prefix, ket_prefix, active, K2)
    if abs(red) < 1e-14:
        return 0.0
    six = wigner_6j(
        bra_prefix.total_J2, bra.total_J2, bra_right.J2,
        ket.total_J2, ket_prefix.total_J2, K2,
    )
    if abs(six) < 1e-14:
        return 0.0
    phase = _phase_from_twice(bra_prefix.total_J2 + bra_right.J2 + ket.total_J2 + K2)
    return float(phase * ((bra.total_J2 + 1) * (ket.total_J2 + 1)) ** 0.5 * six * red)


@lru_cache(maxsize=None)
def chain_projection_amplitudes(
    state: ChainCoupledCSFState,
    M2: int | None = None,
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Expand a coupled CSF chain into uncoupled subshell ``M`` projections."""
    if M2 is None:
        M2 = state.total_J2
    if abs(M2) > state.total_J2 or (state.total_J2 - M2) % 2:
        return tuple()
    if len(state.subshells) == 1:
        if abs(M2) > state.subshells[0].J2 or (state.subshells[0].J2 - M2) % 2:
            return tuple()
        return (((M2,), 1.0),)

    prefix = _chain_prefix(state)
    right_J2 = state.subshells[-1].J2
    rows: list[tuple[tuple[int, ...], float]] = []
    for M_prefix2 in range(-prefix.total_J2, prefix.total_J2 + 1, 2):
        M_right2 = M2 - M_prefix2
        if abs(M_right2) > right_J2 or (right_J2 - M_right2) % 2:
            continue
        cg = clebsch_gordan(
            prefix.total_J2, M_prefix2,
            right_J2, M_right2,
            state.total_J2, M2,
        )
        if abs(cg) < 1e-14:
            continue
        for prefix_ms, prefix_coeff in chain_projection_amplitudes(prefix, M_prefix2):
            rows.append((prefix_ms + (M_right2,), float(prefix_coeff * cg)))
    return tuple(rows)


def chain_wxw_scalar(
    bra: ChainCoupledCSFState,
    ket: ChainCoupledCSFState,
    active_left: int,
    active_right: int,
    K2: int,
) -> float:
    """Scalar ``[W_left^K x W_right^K]^0`` on a coupled CSF chain.

    The implementation expands the ordered chain into uncoupled projection
    amplitudes and applies Wigner-Eckart component matrix elements.  It is a
    validation backend for general multi-subshell recoupling; optimized 9j
    reductions can replace it later behind the same API.
    """
    n = len(bra.subshells)
    if n != len(ket.subshells) or n == 0:
        return 0.0
    if active_left == active_right:
        return 0.0
    if not (0 <= active_left < n and 0 <= active_right < n):
        return 0.0
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return 0.0
    active = {active_left, active_right}
    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        if i not in active and b != k:
            return 0.0

    left_red = reduced_w_coeff(
        bra.subshells[active_left],
        ket.subshells[active_left],
        K2,
    )
    right_red = reduced_w_coeff(
        bra.subshells[active_right],
        ket.subshells[active_right],
        K2,
    )
    if abs(left_red) < 1e-14 or abs(right_red) < 1e-14:
        return 0.0

    total = 0.0
    M2 = bra.total_J2
    bra_terms = chain_projection_amplitudes(bra, M2)
    ket_terms = chain_projection_amplitudes(ket, M2)
    for bra_ms, bra_coeff in bra_terms:
        for ket_ms, ket_coeff in ket_terms:
            for i in range(n):
                if i not in active and bra_ms[i] != ket_ms[i]:
                    break
            else:
                for q2 in range(-K2, K2 + 1, 2):
                    if bra_ms[active_left] != ket_ms[active_left] + q2:
                        continue
                    if bra_ms[active_right] != ket_ms[active_right] - q2:
                        continue
                    scalar_cg = clebsch_gordan(K2, q2, K2, -q2, 0, 0)
                    if abs(scalar_cg) < 1e-14:
                        continue
                    wl = wigner_3j(
                        bra.subshells[active_left].J2,
                        K2,
                        ket.subshells[active_left].J2,
                        -bra_ms[active_left],
                        q2,
                        ket_ms[active_left],
                    )
                    wr = wigner_3j(
                        bra.subshells[active_right].J2,
                        K2,
                        ket.subshells[active_right].J2,
                        -bra_ms[active_right],
                        -q2,
                        ket_ms[active_right],
                    )
                    if abs(wl) < 1e-14 or abs(wr) < 1e-14:
                        continue
                    phase_l = _phase_from_twice(
                        bra.subshells[active_left].J2 - bra_ms[active_left]
                    )
                    phase_r = _phase_from_twice(
                        bra.subshells[active_right].J2 - bra_ms[active_right]
                    )
                    total += (
                        bra_coeff * ket_coeff * scalar_cg
                        * phase_l * wl * left_red
                        * phase_r * wr * right_red
                    )
    return float(total)


def _tensor_component_from_reduced(
    bra_J2: int,
    bra_M2: int,
    op_K2: int,
    op_Q2: int,
    ket_J2: int,
    ket_M2: int,
    reduced: float,
) -> float:
    """Component matrix element from the module's Wigner-Eckart convention."""
    if abs(reduced) < 1e-14:
        return 0.0
    if bra_M2 != ket_M2 + op_Q2:
        return 0.0
    w3j = wigner_3j(bra_J2, op_K2, ket_J2, -bra_M2, op_Q2, ket_M2)
    if abs(w3j) < 1e-14:
        return 0.0
    return float(_phase_from_twice(bra_J2 - bra_M2) * w3j * reduced)


def _creation_component(
    bra: ReducedSubshellState,
    bra_M2: int,
    ket: ReducedSubshellState,
    ket_M2: int,
    m2: int,
) -> float:
    """Component ``<bra|a†_m|ket>`` for one subshell."""
    return _tensor_component_from_reduced(
        bra.J2,
        bra_M2,
        bra.j2,
        m2,
        ket.J2,
        ket_M2,
        reduced_creation_coeff(bra, ket),
    )


def _tilde_annihilation_component(
    bra: ReducedSubshellState,
    bra_M2: int,
    ket: ReducedSubshellState,
    ket_M2: int,
    mu2: int,
) -> float:
    """Component ``<bra|ã_mu|ket>`` for the tensor annihilation operator."""
    return _tensor_component_from_reduced(
        bra.J2,
        bra_M2,
        ket.j2,
        mu2,
        ket.J2,
        ket_M2,
        reduced_annihilation_coeff(bra, ket),
    )


def chain_one_body_excitation_reduced(
    bra: ChainCoupledCSFState,
    ket: ChainCoupledCSFState,
    source: int,
    destination: int,
    K2: int,
) -> float:
    """Reduced CSF coefficient for ``[a†_destination x ã_source]^K``.

    This exact validation backend expands both coupled CSF chains into
    uncoupled subshell projections.  It is intended for simple inferred chains
    and benchmark checks; optimized 6j/9j recoupling can replace the expansion
    later without changing the public API.
    """
    n = len(bra.subshells)
    if n != len(ket.subshells) or n == 0:
        return 0.0
    if source == destination:
        return 0.0
    if not (0 <= source < n and 0 <= destination < n):
        return 0.0

    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        if i == destination:
            if b.j2 != k.j2 or b.q != k.q + 1:
                return 0.0
        elif i == source:
            if b.j2 != k.j2 or b.q + 1 != k.q:
                return 0.0
        elif b != k:
            return 0.0

    Q2 = bra.total_J2 - ket.total_J2
    if abs(Q2) > K2:
        return 0.0
    actual = 0.0
    bra_terms = chain_projection_amplitudes(bra, bra.total_J2)
    ket_terms = chain_projection_amplitudes(ket, ket.total_J2)
    ket_q = [int(s.q) for s in ket.subshells]
    prefix_source = sum(ket_q[:source])
    after_annihilation = list(ket_q)
    after_annihilation[source] -= 1
    prefix_destination = sum(after_annihilation[:destination])
    fermion_phase = -1.0 if (prefix_source + prefix_destination) % 2 else 1.0

    for bra_ms, bra_coeff in bra_terms:
        for ket_ms, ket_coeff in ket_terms:
            for i in range(n):
                if i not in (source, destination) and bra_ms[i] != ket_ms[i]:
                    break
            else:
                for m_create2 in range(
                    -bra.subshells[destination].j2,
                    bra.subshells[destination].j2 + 1,
                    2,
                ):
                    mu2 = Q2 - m_create2
                    if (
                        abs(mu2) > ket.subshells[source].j2
                        or (ket.subshells[source].j2 - mu2) % 2
                    ):
                        continue
                    cg = clebsch_gordan(
                        bra.subshells[destination].j2,
                        m_create2,
                        ket.subshells[source].j2,
                        mu2,
                        K2,
                        Q2,
                    )
                    if abs(cg) < 1e-14:
                        continue
                    cd = _creation_component(
                        bra.subshells[destination],
                        bra_ms[destination],
                        ket.subshells[destination],
                        ket_ms[destination],
                        m_create2,
                    )
                    if abs(cd) < 1e-14:
                        continue
                    ca = _tilde_annihilation_component(
                        bra.subshells[source],
                        bra_ms[source],
                        ket.subshells[source],
                        ket_ms[source],
                        mu2,
                    )
                    if abs(ca) < 1e-14:
                        continue
                    actual += (
                        fermion_phase * bra_coeff * ket_coeff * cg * cd * ca
                    )

    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(
        bra.total_J2,
        K2,
        ket.total_J2,
        -bra.total_J2,
        Q2,
        ket.total_J2,
    )
    if abs(w3j) < 1e-14:
        return 0.0
    return float(actual / w3j)


def _phase_from_twice(exponent2: int) -> float:
    exponent = exponent2 // 2
    return -1.0 if exponent % 2 else 1.0


def two_subshell_left_w_reduced(bra: TwoSubshellCoupledState,
                                ket: TwoSubshellCoupledState,
                                K2: int) -> float:
    """Reduced matrix element for ``W^K`` acting on the left subshell.

    Formula:
    ``< (j1 j2)J || T^K(1) || (j1' j2)J' > =
       (-1)^(j1+j2+J'+K) sqrt((2J+1)(2J'+1))
       { j1  J  j2 ; J' j1' K } <j1||T^K||j1'>``.
    """
    if bra.right != ket.right or bra.parity != ket.parity:
        return 0.0
    red = reduced_w_coeff(bra.left, ket.left, K2)
    if abs(red) < 1e-14:
        return 0.0
    six = wigner_6j(
        bra.left.J2, bra.total_J2, bra.right.J2,
        ket.total_J2, ket.left.J2, K2,
    )
    if abs(six) < 1e-14:
        return 0.0
    phase = _phase_from_twice(bra.left.J2 + bra.right.J2 + ket.total_J2 + K2)
    return float(phase * ((bra.total_J2 + 1) * (ket.total_J2 + 1)) ** 0.5 * six * red)


def two_subshell_right_w_reduced(bra: TwoSubshellCoupledState,
                                 ket: TwoSubshellCoupledState,
                                 K2: int) -> float:
    """Reduced matrix element for ``W^K`` acting on the right subshell."""
    if bra.left != ket.left or bra.parity != ket.parity:
        return 0.0
    red = reduced_w_coeff(bra.right, ket.right, K2)
    if abs(red) < 1e-14:
        return 0.0
    six = wigner_6j(
        bra.right.J2, bra.total_J2, bra.left.J2,
        ket.total_J2, ket.right.J2, K2,
    )
    if abs(six) < 1e-14:
        return 0.0
    phase = _phase_from_twice(bra.right.J2 + bra.left.J2 + ket.total_J2 + K2)
    return float(phase * ((bra.total_J2 + 1) * (ket.total_J2 + 1)) ** 0.5 * six * red)


def two_subshell_wxw_scalar(bra: TwoSubshellCoupledState,
                            ket: TwoSubshellCoupledState,
                            K2: int) -> float:
    """Scalar ``[W_left^K x W_right^K]^0`` matrix element.

    This evaluates the coupled scalar product by expanding over the coupled
    two-subshell product basis.  It is intentionally explicit and compact for
    validation; optimized 9j/6j formulae can replace the expansion later behind
    the same API.
    """
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return 0.0
    left_red = reduced_w_coeff(bra.left, ket.left, K2)
    right_red = reduced_w_coeff(bra.right, ket.right, K2)
    if abs(left_red) < 1e-14 or abs(right_red) < 1e-14:
        return 0.0

    total = 0.0
    J2 = bra.total_J2
    M2 = J2
    for Mb_l2 in range(-bra.left.J2, bra.left.J2 + 1, 2):
        Mb_r2 = M2 - Mb_l2
        cb = clebsch_gordan(bra.left.J2, Mb_l2, bra.right.J2, Mb_r2, J2, M2)
        if abs(cb) < 1e-14:
            continue
        for Mk_l2 in range(-ket.left.J2, ket.left.J2 + 1, 2):
            Mk_r2 = M2 - Mk_l2
            ck = clebsch_gordan(ket.left.J2, Mk_l2, ket.right.J2, Mk_r2, J2, M2)
            if abs(ck) < 1e-14:
                continue
            for q2 in range(-K2, K2 + 1, 2):
                if Mb_l2 != Mk_l2 + q2 or Mb_r2 != Mk_r2 - q2:
                    continue
                scalar_cg = clebsch_gordan(K2, q2, K2, -q2, 0, 0)
                if abs(scalar_cg) < 1e-14:
                    continue
                wl = wigner_3j(bra.left.J2, K2, ket.left.J2,
                               -Mb_l2, q2, Mk_l2)
                wr = wigner_3j(bra.right.J2, K2, ket.right.J2,
                               -Mb_r2, -q2, Mk_r2)
                if abs(wl) < 1e-14 or abs(wr) < 1e-14:
                    continue
                phase_l = _phase_from_twice(bra.left.J2 - Mb_l2)
                phase_r = _phase_from_twice(bra.right.J2 - Mb_r2)
                total += (
                    cb * ck * scalar_cg
                    * phase_l * wl * left_red
                    * phase_r * wr * right_red
                )
    return float(total)


@dataclass(frozen=True)
class TwoBodyCSFCoefficient:
    """A conservative CSF-level two-body angular coefficient."""

    active_left: int
    active_right: int
    K2: int
    coefficient: float
    kind: str = "WxW0"


@dataclass(frozen=True)
class CSFOccupationChange:
    """Occupation-shape change between two recoupled CSFs."""

    sources: tuple[int, ...]
    destinations: tuple[int, ...]

    @property
    def order(self) -> int:
        return max(len(self.sources), len(self.destinations))

    @property
    def kind(self) -> str:
        if not self.sources and not self.destinations:
            return "diagonal"
        if len(self.sources) == 1 and len(self.destinations) == 1:
            return "single"
        if len(self.sources) == 2 and len(self.destinations) == 2:
            return "double"
        return "unsupported"


def classify_recoupled_occupation_change(bra: RecoupledCSFState,
                                         ket: RecoupledCSFState) -> CSFOccupationChange:
    """Classify subshell occupation changes from ket to bra.

    The returned indices refer to positions in ``subshells``.  ``sources`` are
    depleted in the bra relative to ket; ``destinations`` are populated in the
    bra relative to ket.  Parentage/J changes without occupation changes are
    treated as diagonal shape changes and left to the coefficient routines.
    """
    if len(bra.subshells) != len(ket.subshells):
        return CSFOccupationChange(tuple(range(len(ket.subshells))), tuple(range(len(bra.subshells))))
    sources = []
    destinations = []
    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        dq = b.q - k.q
        if dq < 0:
            sources.extend([i] * abs(dq))
        elif dq > 0:
            destinations.extend([i] * dq)
    return CSFOccupationChange(tuple(sources), tuple(destinations))


def two_body_scalar_coefficients_simple(bra: RecoupledCSFState,
                                        ket: RecoupledCSFState,
                                        active_left: int,
                                        active_right: int,
                                        K2: int) -> tuple[TwoBodyCSFCoefficient, ...]:
    """Return scalar two-body coefficients for a simple two-active-subshell CSF.

    This is the first safe CSF-level bridge to the recoupling layer.  It
    handles unchanged occupations/parentage on all spectator subshells and
    evaluates a scalar product of W tensors on two active subshells.  It does
    not yet handle general parentage, occupation-changing operators, or
    multi-step recoupling trees.
    """
    if active_left == active_right:
        return tuple()
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return tuple()
    if len(bra.subshells) != len(ket.subshells):
        return tuple()
    n = len(bra.subshells)
    if not (0 <= active_left < n and 0 <= active_right < n):
        return tuple()
    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        if i in (active_left, active_right):
            continue
        if b != k:
            return tuple()

    left_bra = bra.subshells[active_left]
    right_bra = bra.subshells[active_right]
    left_ket = ket.subshells[active_left]
    right_ket = ket.subshells[active_right]
    if not angular_momenta_couple(left_bra.J2, right_bra.J2, bra.total_J2):
        return tuple()
    if not angular_momenta_couple(left_ket.J2, right_ket.J2, ket.total_J2):
        return tuple()

    coupled_bra = TwoSubshellCoupledState(
        left=left_bra,
        right=right_bra,
        total_J2=bra.total_J2,
        parity=bra.parity,
    )
    coupled_ket = TwoSubshellCoupledState(
        left=left_ket,
        right=right_ket,
        total_J2=ket.total_J2,
        parity=ket.parity,
    )
    coeff = two_subshell_wxw_scalar(coupled_bra, coupled_ket, K2)
    if abs(coeff) < 1e-14:
        return tuple()
    return (TwoBodyCSFCoefficient(
        active_left=active_left,
        active_right=active_right,
        K2=K2,
        coefficient=coeff,
    ),)


def chain_two_body_scalar_coefficients(
    bra: ChainCoupledCSFState,
    ket: ChainCoupledCSFState,
    active_left: int,
    active_right: int,
    K2: int,
) -> tuple[TwoBodyCSFCoefficient, ...]:
    """Return scalar two-body coefficients for explicit CSF coupling chains."""
    coeff = chain_wxw_scalar(bra, ket, active_left, active_right, K2)
    if abs(coeff) < 1e-14:
        return tuple()
    return (TwoBodyCSFCoefficient(
        active_left=active_left,
        active_right=active_right,
        K2=K2,
        coefficient=coeff,
        kind="chain-WxW0",
    ),)


def discover_chain_two_body_scalar_coefficients(
    bra: ChainCoupledCSFState,
    ket: ChainCoupledCSFState,
    K2_values: tuple[int, ...] = (0,),
) -> tuple[TwoBodyCSFCoefficient, ...]:
    """Discover scalar ``WxW0`` coefficients across all chain active pairs."""
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return tuple()
    if len(bra.subshells) != len(ket.subshells):
        return tuple()
    out: list[TwoBodyCSFCoefficient] = []
    n = len(bra.subshells)
    for i in range(n):
        for j in range(i + 1, n):
            for K2 in K2_values:
                out.extend(chain_two_body_scalar_coefficients(bra, ket, i, j, K2))
    return tuple(out)


def discover_two_body_scalar_coefficients_simple(
    bra: RecoupledCSFState,
    ket: RecoupledCSFState,
    K2_values: tuple[int, ...] = (0,),
) -> tuple[TwoBodyCSFCoefficient, ...]:
    """Discover simple scalar two-body coefficients for all active pairs.

    This scans every pair of subshell positions and returns non-zero
    conservative scalar `WxW0` coefficients.  It is suitable for diagonal or
    same-parentage simple CSFs and intentionally returns no coefficients when
    the state shape is outside that scope.
    """
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return tuple()
    if len(bra.subshells) != len(ket.subshells):
        return tuple()
    out = []
    n = len(bra.subshells)
    for i in range(n):
        for j in range(i + 1, n):
            for K2 in K2_values:
                out.extend(two_body_scalar_coefficients_simple(bra, ket, i, j, K2))
    return tuple(out)


def discover_two_body_coefficients_conservative(
    bra: RecoupledCSFState,
    ket: RecoupledCSFState,
    K2_values: tuple[int, ...] = (0,),
) -> tuple[TwoBodyCSFCoefficient, ...]:
    """Conservative CSF two-body coefficient dispatcher.

    Currently supports diagonal/simple scalar `WxW0` coefficients.  Single and
    double occupation-changing cases are classified but return no coefficients
    until their AxW/WxA/pair-operator recoupling is implemented.
    """
    change = classify_recoupled_occupation_change(bra, ket)
    if change.kind == "diagonal":
        return discover_two_body_scalar_coefficients_simple(bra, ket, K2_values)
    if change.kind == "single":
        return discover_single_excitation_coefficients_simple(bra, ket, change)
    if change.kind == "double":
        return discover_double_excitation_coefficients_simple(bra, ket, change)
    return tuple()


def discover_single_excitation_coefficients_simple(
    bra: RecoupledCSFState,
    ket: RecoupledCSFState,
    change: CSFOccupationChange | None = None,
) -> tuple[TwoBodyCSFCoefficient, ...]:
    """Conservative single-excitation CSF coefficients.

    Handles one source and one destination subshell with all other subshells as
    matching spectators.  The coefficient is the product of a reduced
    annihilation coefficient on the source, a reduced creation coefficient on
    the destination, and scalar spectator ``W`` factors for each unchanged
    occupied subshell.  This is a deliberately limited bridge for testing the
    occupation-changing API before full recoupling phases are introduced.
    """
    if change is None:
        change = classify_recoupled_occupation_change(bra, ket)
    if change.kind != "single":
        return tuple()
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return tuple()
    source = change.sources[0]
    dest = change.destinations[0]
    if len(bra.subshells) != len(ket.subshells):
        return tuple()
    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        if i in (source, dest):
            continue
        if b != k:
            return tuple()
    remove_coeff = reduced_annihilation_coeff(bra.subshells[source], ket.subshells[source])
    add_coeff = reduced_creation_coeff(bra.subshells[dest], ket.subshells[dest])
    if abs(remove_coeff) < 1e-14 or abs(add_coeff) < 1e-14:
        return tuple()
    coeff = remove_coeff * add_coeff
    for i, state in enumerate(bra.subshells):
        if i in (source, dest):
            continue
        if state.q <= 0:
            continue
        coeff *= reduced_w_coeff(state, state, K2=0)
    if abs(coeff) < 1e-14:
        return tuple()
    return (TwoBodyCSFCoefficient(
        active_left=source,
        active_right=dest,
        K2=0,
        coefficient=coeff,
        kind="single",
    ),)


def discover_double_excitation_coefficients_simple(
    bra: RecoupledCSFState,
    ket: RecoupledCSFState,
    change: CSFOccupationChange | None = None,
    K2: int = 0,
) -> tuple[TwoBodyCSFCoefficient, ...]:
    """Conservative paired double-excitation coefficient.

    Supports the paired case where two electrons are removed from one source
    subshell and added to one destination subshell.  Mixed two-source/two-dest
    recoupling is left for the general two-body layer.
    """
    if change is None:
        change = classify_recoupled_occupation_change(bra, ket)
    if change.kind != "double":
        return tuple()
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return tuple()
    if len(bra.subshells) != len(ket.subshells):
        return tuple()
    paired = len(set(change.sources)) == 1 and len(set(change.destinations)) == 1
    mixed = len(set(change.sources)) == 2 and len(set(change.destinations)) == 2
    if not (paired or mixed):
        return tuple()
    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        if i in set(change.sources) | set(change.destinations):
            continue
        if b != k:
            return tuple()
    if mixed:
        return _discover_mixed_double_excitation_coefficients_simple(bra, ket, change)

    source = change.sources[0]
    dest = change.destinations[0]
    if source == dest:
        return tuple()
    remove_pair = reduced_pair_annihilation_coeff(bra.subshells[source], ket.subshells[source], K2)
    add_pair = reduced_pair_creation_coeff(bra.subshells[dest], ket.subshells[dest], K2)
    if abs(remove_pair) < 1e-14 or abs(add_pair) < 1e-14:
        return tuple()
    coeff = remove_pair * add_pair
    for i, state in enumerate(bra.subshells):
        if i in (source, dest):
            continue
        if state.q <= 0:
            continue
        coeff *= reduced_w_coeff(state, state, K2=0)
    if abs(coeff) < 1e-14:
        return tuple()
    return (TwoBodyCSFCoefficient(
        active_left=source,
        active_right=dest,
        K2=K2,
        coefficient=coeff,
        kind="double",
    ),)


def _discover_mixed_double_excitation_coefficients_simple(
    bra: RecoupledCSFState,
    ket: RecoupledCSFState,
    change: CSFOccupationChange,
) -> tuple[TwoBodyCSFCoefficient, ...]:
    sources = tuple(dict.fromkeys(change.sources))
    destinations = tuple(dict.fromkeys(change.destinations))
    coeff = 1.0
    for source in sources:
        c = reduced_annihilation_coeff(bra.subshells[source], ket.subshells[source])
        if abs(c) < 1e-14:
            return tuple()
        coeff *= c
    for dest in destinations:
        c = reduced_creation_coeff(bra.subshells[dest], ket.subshells[dest])
        if abs(c) < 1e-14:
            return tuple()
        coeff *= c
    for i, state in enumerate(bra.subshells):
        if i in set(sources) | set(destinations):
            continue
        if state.q <= 0:
            continue
        coeff *= reduced_w_coeff(state, state, K2=0)
    if abs(coeff) < 1e-14:
        return tuple()
    return (TwoBodyCSFCoefficient(
        active_left=sources[0],
        active_right=destinations[0],
        K2=0,
        coefficient=coeff,
        kind="double-mixed",
    ),)


def spectator_subshells_match(bra: RecoupledCSFState,
                              ket: RecoupledCSFState,
                              active: int) -> bool:
    """Whether all non-active subshell reduced states are identical."""
    if len(bra.subshells) != len(ket.subshells):
        return False
    for i, (b, k) in enumerate(zip(bra.subshells, ket.subshells)):
        if i == active:
            continue
        if b != k:
            return False
    return True


def scalar_w_subshell_coeff(bra: RecoupledCSFState,
                            ket: RecoupledCSFState,
                            active: int) -> float:
    """Scalar same-subshell ``W`` coefficient for a multi-subshell CSF.

    This evaluates ``[a† x atilde]^0`` on one active subshell and requires all
    spectator subshell states to match.  The result is the exact scalar
    product-state coefficient for the current reduced-state convention.
    """
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return 0.0
    if active < 0 or active >= len(bra.subshells):
        return 0.0
    if not spectator_subshells_match(bra, ket, active):
        return 0.0
    return reduced_w_coeff(bra.subshells[active], ket.subshells[active], K2=0)


def scalar_wxw0_subshell_coeff(bra: RecoupledCSFState,
                               ket: RecoupledCSFState,
                               active: int) -> float:
    """Scalar same-subshell ``[W^0 x W^0]^0`` coefficient."""
    if bra.total_J2 != ket.total_J2 or bra.parity != ket.parity:
        return 0.0
    if active < 0 or active >= len(bra.subshells):
        return 0.0
    if not spectator_subshells_match(bra, ket, active):
        return 0.0
    return reduced_wxw0_coeff(bra.subshells[active], ket.subshells[active], K2=0)


def single_state_for_subshell(j2: int, occupation: int) -> ReducedSubshellState | None:
    """Return the unique reduced state for a closed/simple subshell if present.

    This helper is intentionally conservative.  It is exact for empty, one-
    electron, one-hole, and closed subshells where the reduced state is unique.
    Ambiguous open shells return ``None`` until parentage labels are threaded
    through the CSF generator.
    """
    counts = subshell_j_counts_mscheme(j2, occupation)
    if sum(counts.values()) != 1:
        return None
    J2 = next(iter(counts))
    return ReducedSubshellState(j2=j2, q=occupation, J2=J2, index=0)


def recoupled_state_from_simple_csf(csf) -> RecoupledCSFState | None:
    """Build a recoupled state for CSFs with unique subshell parentage."""
    subshell_states = []
    for sh in getattr(csf, "subshells", []):
        state = single_state_for_subshell(sh.j2, int(round(sh.occupation)))
        if state is None:
            return None
        subshell_states.append(state)
    return RecoupledCSFState(
        subshells=tuple(subshell_states),
        total_J2=int(getattr(csf, "total_J2", 0)),
        parity=int(getattr(csf, "parity", 1)),
        label=str(getattr(csf, "label", "")),
    )


def chain_state_from_simple_csf_auto(csf) -> ChainCoupledCSFState | None:
    """Infer a left-associated coupling chain for simple CSFs when unique.

    The current CSF generator stores total ``J`` but not intermediate coupling
    labels.  This helper fills that gap only for deterministic cases: closed
    or zero-``J`` spectators, a single open subshell, or two nonzero open
    subshells whose final coupling is fixed by the CSF total ``J``.  More
    complicated open-shell chains return ``None`` until explicit parentage and
    coupling-tree labels are available.
    """
    state = recoupled_state_from_simple_csf(csf)
    if state is None or not state.subshells:
        return None

    total_J2 = state.total_J2
    cumulative: list[int] = [state.subshells[0].J2]
    current_J2 = state.subshells[0].J2
    nonzero_open_seen = 1 if current_J2 != 0 else 0

    for i, subshell in enumerate(state.subshells[1:], start=1):
        if subshell.J2 == 0:
            next_J2 = current_J2
        elif current_J2 == 0:
            next_J2 = subshell.J2
            nonzero_open_seen += 1
        else:
            nonzero_open_seen += 1
            if i != len(state.subshells) - 1 or nonzero_open_seen > 2:
                return None
            if not angular_momenta_couple(current_J2, subshell.J2, total_J2):
                return None
            next_J2 = total_J2
        cumulative.append(next_J2)
        current_J2 = next_J2

    if current_J2 != total_J2:
        return None
    try:
        return make_chain_coupled_state(
            state.subshells,
            tuple(cumulative),
            parity=state.parity,
            label=state.label,
        )
    except ValueError:
        return None


def aligned_subshell_keys_from_simple_csfs(csf_a, csf_b) -> tuple[tuple[int, int], ...] | None:
    """Return a deterministic merged subshell order for simple CSFs.

    The FAC recoupling inventory needs a stable shell chain even when one CSF
    has zero occupation in an active subshell.  We only expose this for CSFs
    whose occupied subshells have unique reduced parentage; ambiguous open
    shells still return ``None``.
    """
    for csf in (csf_a, csf_b):
        for sh in getattr(csf, "subshells", []):
            if single_state_for_subshell(sh.j2, int(round(sh.occupation))) is None:
                return None

    keys = {
        (int(sh.n), int(sh.kappa))
        for csf in (csf_a, csf_b)
        for sh in getattr(csf, "subshells", [])
    }
    return tuple(sorted(keys, key=lambda key: (key[0], abs(key[1]), key[1])))


def fac_angular_zxz0_branch_case(indices: tuple[int, int, int, int]):
    """Mirror FAC ``AngularZxZ0`` shell sorting and branch selection."""
    order = [0, 1, 2, 3]
    phase = 0
    for j in range(3):
        for i in range(3, j, -1):
            if indices[order[i]] < indices[order[i - 1]]:
                order[i], order[i - 1] = order[i - 1], order[i]
                phase += 1

    counts = []
    for pos in reversed(order):
        if not counts or indices[pos] != counts[-1][0]:
            counts.append([indices[pos], 1])
        else:
            counts[-1][1] += 1
    nops = [count for _idx, count in counts]

    if nops == [4]:
        branch = 0
    elif nops == [3, 1]:
        branch = 1
    elif nops == [1, 3]:
        branch = 3
    elif nops == [1, 2, 1]:
        branch = 5
    elif nops == [2, 2]:
        branch = 6
    elif nops == [2, 1, 1]:
        branch = 7
    elif nops == [1, 1, 2]:
        branch = 8
    else:
        branch = 9
    return tuple(order), phase, branch


def fac_simple_axw_wxa_zxz0_postprocess(*_args, **_kwargs):
    """Placeholder for FAC branch-7/8 composite-seed diagnostics."""
    return None
