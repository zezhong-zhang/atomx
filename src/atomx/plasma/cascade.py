"""X-ray / decay cascade calculator (radiative + non-radiative).

After an inner-shell ionisation or excitation event the resulting
core-hole state typically decays through a *cascade* of transitions:
the hole migrates outward via radiative + Auger decays until the ion
reaches a stable charge state.  This module solves the rate-equation
system for the population fractions of every level in the cascade
chain — equivalent to FAC's ``SetCascade`` knob in the CRM module.

Approach
--------
Given a list of levels ``i = 0, 1, …, N−1`` with total decay rates
Γ_i and per-channel branching ratios

    p_i→j = A_i→j / Γ_i

the cascade probability that an ion initialised in level i_0 ends up
in level j_final is

    P(i_0 → j_final) = Σ_paths Π_steps p_i→j

We compute it by direct sparse-vector propagation: starting with a
unit population in i_0, we iteratively apply the transition matrix
T_{ij} = p_i→j / (1 + δ_{ij}) until the population concentrates in
fully-stable terminal levels.  Convergence is exponential because
the chain is acyclic when feeding only "down-going" transitions.

Public API
----------
:func:`cascade_yields(levels, transitions, initial_idx)` —
    Returns ``{terminal_idx: population_fraction}``.

:func:`cascade_x_ray_spectrum(levels, transitions, initial_idx)` —
    Returns the integrated photon yield ``{(i, j): yield}`` over the
    full cascade.  Useful for predicting fluorescence / Auger yields
    after an inner-shell ionisation event.

References
----------
* B. Crasemann, *Atomic Inner-Shell Processes* (Academic Press, 1975)
  — cascade theory.
* M. F. Gu, *Astrophys. J.* **657**, 1172 (2007) — FAC ``SetCascade``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class CascadeLevel:
    """Single energy level in the cascade chain."""
    label: str
    energy_eV: float = 0.0


@dataclass
class CascadeTransition:
    """Decay channel  i → j  with rate (radiative or Auger)."""
    upper: int
    lower: int
    rate_per_s: float
    photon_eV: float = 0.0      # 0 for non-radiative (Auger / Coster-Kronig)


@dataclass
class CascadeResult:
    terminal_yields: Dict[int, float]
    photon_yields: Dict[Tuple[int, int], float] = field(default_factory=dict)
    auger_yields: Dict[Tuple[int, int], float] = field(default_factory=dict)


def _compute_branching(
    n_levels: int,
    transitions: Sequence[CascadeTransition],
) -> Tuple[List[float], List[List[Tuple[int, float, float]]]]:
    """For each upper level i return (Γ_i, [(j, branching_ratio, photon_eV)])."""
    Gamma = [0.0] * n_levels
    by_upper: List[List[Tuple[int, float, float]]] = [[] for _ in range(n_levels)]
    for tr in transitions:
        if tr.rate_per_s <= 0:
            continue
        if tr.upper == tr.lower:
            raise ValueError(
                f"CascadeTransition: upper == lower ({tr.upper})"
            )
        Gamma[tr.upper] += tr.rate_per_s
        by_upper[tr.upper].append((tr.lower, tr.rate_per_s, tr.photon_eV))
    # Convert per-channel rates to branching ratios
    out: List[List[Tuple[int, float, float]]] = []
    for i in range(n_levels):
        rows = by_upper[i]
        if Gamma[i] <= 0:
            out.append([])
        else:
            out.append([(j, rate / Gamma[i], hv) for (j, rate, hv) in rows])
    return Gamma, out


def cascade_yields(
    levels: Sequence[CascadeLevel],
    transitions: Sequence[CascadeTransition],
    initial_idx: int,
    *,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> CascadeResult:
    """Solve the cascade rate equations and return decay yields.

    Iterates the population vector P_i with

        P^{n+1}_j = Σ_i P^{n}_i · b_{i→j}

    until populations only persist in terminal (no-decay) levels.
    Also accumulates photon and Auger yields per (i, j) channel.

    Parameters
    ----------
    levels : sequence of CascadeLevel
    transitions : sequence of CascadeTransition
    initial_idx : int
        Index of the level where the cascade starts (e.g. the
        inner-shell-ionised core-hole state).
    max_iter : int
        Safety cap; the cascade should converge in O(N_levels) iters
        for an acyclic graph.
    tol : float
        Convergence threshold on the maximum population in any
        non-terminal level.

    Returns
    -------
    CascadeResult
    """
    n = len(levels)
    if not (0 <= initial_idx < n):
        raise ValueError(
            f"initial_idx {initial_idx} out of range [0, {n})"
        )
    Gamma, branches = _compute_branching(n, transitions)

    pop = [0.0] * n
    pop[initial_idx] = 1.0
    photon_yields: Dict[Tuple[int, int], float] = {}
    auger_yields: Dict[Tuple[int, int], float] = {}

    for _ in range(max_iter):
        new_pop = [0.0] * n
        moved = False
        for i in range(n):
            if pop[i] == 0.0 or Gamma[i] <= 0:
                # Terminal: keep population in place
                new_pop[i] += pop[i]
                continue
            moved = True
            for (j, b, hv) in branches[i]:
                contrib = pop[i] * b
                new_pop[j] += contrib
                if hv > 0:
                    photon_yields[(i, j)] = photon_yields.get((i, j), 0.0) + contrib
                else:
                    auger_yields[(i, j)] = auger_yields.get((i, j), 0.0) + contrib
        pop = new_pop
        if not moved:
            break
        # Convergence: check no significant population in non-terminal levels.
        residual = sum(p for i, p in enumerate(pop) if Gamma[i] > 0)
        if residual < tol:
            break

    terminal = {i: p for i, p in enumerate(pop) if p > tol}
    return CascadeResult(
        terminal_yields=terminal,
        photon_yields=photon_yields,
        auger_yields=auger_yields,
    )


def fluorescence_yield(
    cascade: CascadeResult,
) -> float:
    """Total photon emission probability per initialised core hole.

    ω_x  =  Σ_{(i,j)} photon_yields(i, j)

    For a K-shell hole this approaches the famous K-fluorescence
    yield ω_K ≈ Z⁴ / (Z⁴ + a) with a depending on the element.
    """
    return float(sum(cascade.photon_yields.values()))


def auger_yield(
    cascade: CascadeResult,
) -> float:
    """Total Auger / Coster-Kronig probability per initialised hole.

    For mass conservation this should satisfy
    ω_x + ω_A = 1 − Σ trapping-into-cascade-loop terms.
    """
    return float(sum(cascade.auger_yields.values()))


__all__ = [
    "CascadeLevel",
    "CascadeTransition",
    "CascadeResult",
    "cascade_yields",
    "fluorescence_yield",
    "auger_yield",
]
