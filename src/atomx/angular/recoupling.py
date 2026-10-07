"""LS-jj recoupling transformations.

Provides transformation coefficients between LS-coupling and jj-coupling
representations.  These are essential for:

1. Converting non-relativistic HF results (LS-coupled) to compare with
   relativistic Dirac-Fock results (jj-coupled)
2. Intermediate coupling calculations where spin-orbit mixes LS terms
3. Understanding the transition from light atoms (LS-coupled) to
   heavy atoms (jj-coupled)

The transformation uses Wigner 9j symbols:

    <(l1 s1)j1, (l2 s2)j2; J | (l1 l2)L, (s1 s2)S; J>
    = sqrt((2j1+1)(2j2+1)(2L+1)(2S+1)) * { l1 s1 j1 }
                                            { l2 s2 j2 }
                                            { L  S  J  }
"""

from __future__ import annotations

import numpy as np

from atomx.angular.wigner import wigner_9j


def ls_jj_transformation(l1: int, l2: int, J2: int) -> np.ndarray:
    """Compute LS→jj transformation matrix for two electrons.

    For two electrons with orbital angular momenta l1, l2 (each with s=1/2),
    computes the unitary matrix that transforms from LS-coupled basis
    |(l1 l2)L, (s1 s2)S; J> to jj-coupled basis |(l1 s1)j1, (l2 s2)j2; J>.

    Parameters
    ----------
    l1, l2 : int
        Orbital angular momenta of the two electrons.
    J2 : int
        Twice the total angular momentum (2J).

    Returns
    -------
    np.ndarray
        Transformation matrix T[jj_idx, ls_idx].
        Rows: jj-coupled states, columns: LS-coupled states.

    Notes
    -----
    ``J2`` is **twice** J.  Passing J instead of 2J (e.g. 1 for a J = 1
    block) silently yields rectangular matrices, because the LS
    enumeration (integer 2L, 2S) then no longer matches the jj one —
    that is how the 2026-09-19 review's "22 of 45 non-square blocks"
    figure arises (finding M11).

    No parity guard is missing, and neither of the two guards the review
    suggested would change anything: with every argument doubled, both
    ``(2j1 + 2j2 + J2) % 2`` and ``(2L + 2S + J2) % 2`` are identically
    even (every 2j is odd, every 2L/2S/2J is even), i.e. vacuous.  Both
    bases enumerate the same J-subspace of (l1 ⊗ l2) ⊗ (s1 ⊗ s2), so
    with the documented 2J convention ``n_jj == n_ls`` and
    ``T @ T.T == I`` for every (l1, l2, J) — measured |dev| ≤ 4e-15 for
    all 33 non-empty blocks with l1, l2 ≤ 2, J ≤ 4 (12 blocks are empty
    because J < |l1 − l2|); gated in
    ``tests/test_review_m9_m13_gates.py``.

    For two *equivalent* electrons the Pauli-allowed subspace is the
    L + S even (equivalently, for j1 = j2, ``(-1)^(2j-J) = -1``)
    sub-block; restricting *both* bases to it keeps T unitary.
    """
    s2 = 1  # 2*s = 1 for each electron

    # Enumerate jj-coupled basis: (j1, j2) pairs
    jj_basis = []
    for j1_2 in [2 * l1 - 1, 2 * l1 + 1]:
        if j1_2 < 1:
            continue
        for j2_2 in [2 * l2 - 1, 2 * l2 + 1]:
            if j2_2 < 1:
                continue
            # Triangle rule: |j1 - j2| <= J <= j1 + j2
            if abs(j1_2 - j2_2) <= J2 <= j1_2 + j2_2:
                jj_basis.append((j1_2, j2_2))

    # Enumerate LS-coupled basis: (L, S) pairs
    ls_basis = []
    for L in range(abs(l1 - l2), l1 + l2 + 1):
        for S2 in [0, 2]:  # 2S = 0 (singlet) or 2 (triplet)
            # Triangle: |L - S| <= J <= L + S
            L2 = 2 * L
            if abs(L2 - S2) <= J2 <= L2 + S2:
                ls_basis.append((L, S2))

    n_jj = len(jj_basis)
    n_ls = len(ls_basis)
    T = np.zeros((n_jj, n_ls))

    for i, (j1_2, j2_2) in enumerate(jj_basis):
        for j, (L, S2) in enumerate(ls_basis):
            # 9j symbol with all arguments as twice the values
            L2 = 2 * L
            l1_2 = 2 * l1
            l2_2 = 2 * l2
            w9j = wigner_9j(l1_2, s2, j1_2,
                            l2_2, s2, j2_2,
                            L2, S2, J2)
            T[i, j] = (np.sqrt((j1_2 + 1) * (j2_2 + 1)
                                * (L2 + 1) * (S2 + 1))
                        * w9j)

    return T


def l_of_kappa(kappa: int) -> int:
    """Orbital angular momentum ``l`` of the large component for ``kappa``.

    ``kappa < 0`` gives ``l = -kappa - 1`` (``j = l + 1/2``);
    ``kappa > 0`` gives ``l = kappa`` (``j = l - 1/2``).

    This is the canonical κ→l mapping for the whole package; modules that
    previously carried private copies (``_l_of``, ``_l_from_kappa``,
    ``_l_of_kappa``, ...) import it from here.
    """
    return kappa if kappa > 0 else -kappa - 1


def twoj_of_kappa(kappa: int) -> int:
    """Twice the total angular momentum, ``2j = 2|kappa| - 1``.

    Canonical κ→2j mapping; the former per-module copies
    (``_tj_of``, ``_twoj_from_kappa``, ``twoj``, ...) delegate here.
    """
    return 2 * abs(kappa) - 1


def kappa_from_lj(l: int, j2: int) -> int:
    """Convert (l, 2j) to relativistic quantum number kappa.

    kappa = -(l+1) for j = l + 1/2  (j > l)
    kappa = +l     for j = l - 1/2  (j < l)
    """
    if j2 == 2 * l + 1:
        return -(l + 1)
    elif j2 == 2 * l - 1:
        return l
    else:
        raise ValueError(f"Invalid (l={l}, 2j={j2}) combination")


def lj_from_kappa(kappa: int) -> tuple:
    """Convert relativistic kappa to (l, 2j).

    Returns
    -------
    (l, j2) : tuple of int
    """
    return l_of_kappa(kappa), twoj_of_kappa(kappa)
