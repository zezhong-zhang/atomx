"""LS-coupling (Russell-Saunders) angular momentum algebra.

Provides term symbols, Slater-Condon parameters, and LS-coupling
coefficients for non-relativistic atomic calculations.

In LS-coupling, the total orbital angular momentum L = sum(l_i) and
total spin S = sum(s_i) are separately good quantum numbers, and the
total J = L + S is formed by their vector coupling.  This is valid
when spin-orbit coupling is small compared to Coulomb repulsion
(typically Z < 30).
"""

from __future__ import annotations

from typing import List, Tuple


def term_symbol(L: int, S2: int, J2: int) -> str:
    """Format a spectroscopic term symbol ^{2S+1}L_J.

    Parameters
    ----------
    L : int
        Total orbital angular momentum quantum number.
    S2 : int
        Twice the total spin (2S).
    J2 : int
        Twice the total angular momentum (2J).

    Returns
    -------
    str
        Term symbol like "^3P_2" or "^1S_0".
    """
    L_symbols = "SPDFGHIKLMNO"
    L_char = L_symbols[L] if len(L_symbols) > L else f"[{L}]"
    multiplicity = S2 + 1
    J_str = f"{J2}/2" if J2 % 2 != 0 else str(J2 // 2)
    return f"^{multiplicity}{L_char}_{J_str}"


def ls_expectation(l: int, j2: int) -> float:
    """Compute <L·S> for a single electron with quantum numbers l, j.

    <L·S> = (1/2)[j(j+1) - l(l+1) - s(s+1)]  where s = 1/2.

    Parameters
    ----------
    l : int
        Orbital angular momentum.
    j2 : int
        Twice the total angular momentum (2j).

    Returns
    -------
    float
        The expectation value <L·S>.
    """
    j = j2 / 2.0
    s = 0.5
    return 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))


def allowed_terms(l: int, n_electrons: int) -> List[Tuple[int, int]]:
    """Find allowed LS terms for n equivalent electrons in shell l.

    Returns list of (L, 2S) pairs for the allowed terms of the
    configuration l^n (e.g., p^2 → ^3P, ^1D, ^1S).

    Parameters
    ----------
    l : int
        Orbital angular momentum of the shell.
    n_electrons : int
        Number of electrons in the shell.

    Returns
    -------
    list of (L, 2S)
        Allowed LS terms, sorted by Hund's rules (highest S first,
        then highest L).
    """
    max_occ = 2 * (2 * l + 1)
    if n_electrons > max_occ:
        raise ValueError(f"Cannot place {n_electrons} electrons in l={l} shell (max {max_occ})")
    if n_electrons == 0 or n_electrons == max_occ:
        return [(0, 0)]  # ^1S only

    # Use particle-hole symmetry
    if n_electrons > max_occ // 2:
        n_electrons = max_occ - n_electrons

    # Generate all Slater determinants and extract unique (L, S) terms
    # For small shells, enumerate ml, ms assignments
    ml_values = list(range(-l, l + 1))
    ms_values = [+1, -1]  # 2*ms = +1 or -1

    # Single-particle states: (ml, 2ms)
    states = [(ml, ms) for ml in ml_values for ms in ms_values]

    from itertools import combinations

    ML_MS_counts = {}

    for combo in combinations(range(len(states)), n_electrons):
        ML = sum(states[i][0] for i in combo)
        MS = sum(states[i][1] for i in combo)
        key = (ML, MS)
        ML_MS_counts[key] = ML_MS_counts.get(key, 0) + 1

    # Extract terms by peeling off from highest (ML, MS)
    terms = []
    counts = dict(ML_MS_counts)

    while any(v > 0 for v in counts.values()):
        # Find the maximum MS with positive count
        max_MS = max(ms for (ml, ms), c in counts.items() if c > 0)
        # Among those, find max ML
        max_ML = max(ml for (ml, ms), c in counts.items() if c > 0 and ms == max_MS)

        L = max_ML
        S2 = max_MS  # This is 2S

        terms.append((L, S2))

        # Subtract this term's contribution
        for ML in range(-L, L + 1):
            for MS in range(-S2, S2 + 1, 2):
                key = (ML, MS)
                if key in counts:
                    counts[key] -= 1

    # Sort by Hund's rules: highest S first, then highest L
    terms.sort(key=lambda x: (-x[1], -x[0]))
    return terms


def slater_condon_f(k: int, l: int) -> float:
    """Compute the angular factor c^k(l, l) for Slater-Condon F^k integrals.

    c^k(l,l) = <l 0 k 0 | l 0>^2 (Clebsch-Gordan coefficient squared).

    Parameters
    ----------
    k : int
        Multipole rank (must be even, 0 <= k <= 2l).
    l : int
        Orbital angular momentum.

    Returns
    -------
    float
        The angular factor.
    """
    if k % 2 != 0 or k > 2 * l:
        return 0.0

    from atomx.angular.wigner import wigner_3j
    # c^k(l,l) = (-1)^l * sqrt(2l+1) * (l k l; 0 0 0) * sqrt(2l+1)
    # = (2l+1) * (l k l; 0 0 0)^2
    w3j = wigner_3j(2 * l, 2 * k, 2 * l, 0, 0, 0)
    return (2 * l + 1) ** 2 * w3j ** 2
