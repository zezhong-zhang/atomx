"""Signed pCI particle-hole rho_k convention in shared magnetic algebra.

The coefficient is defined without dividing electric reduced matrix
elements: rho channels with vanishing C^k can carry exchange contributions.
Each radial key retains its complete magnetic multiplet. Antisymmetrization
and fermion excitation signs are applied by the consuming method.
"""

import hashlib
from pathlib import Path

from atomx.angular.recoupling import twoj_of_kappa
from atomx.angular.wigner import wigner_3j

_import_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def partial_index(kappa):
    """One-based original pCI partial-wave index for nonzero kappa."""
    if not isinstance(kappa, int) or isinstance(kappa, bool) or kappa == 0:
        raise ValueError("integer nonzero kappa required")
    return -2 * kappa - 1 if kappa < 0 else 2 * kappa


def kappa_from_partial(index):
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise ValueError("positive integer partial-wave index required")
    return -(index + 1) // 2 if index % 2 else index // 2


def rho_coefficient(rank, a, b, i, j):
    """Angular coefficient of direct rho_k(abij), labels (n,kappa,2m).

    Matches X_k=(-1)^k S_pCI(k,a,i) S_pCI(k,b,j) R^k(ai,bj).
    No conjugation, orbital phase change, angular cutoff or normalization
    fit is performed. The core SD consumer currently requires real orbitals.
    """
    ja, jb, ji, jj = (twoj_of_kappa(x[1]) for x in (a, b, i, j))
    ma, mb, mi, mj = (x[2] for x in (a, b, i, j))
    if ma + mb != mi + mj:
        return 0.0
    q = (ma - mi) // 2
    return (
        (-1)
        ** ((ja - ma + jb - mb) // 2 + q + rank + sum(abs(x[1]) for x in (a, b, i, j)))
        * wigner_3j(ja, 2 * rank, ji, -ma, 2 * q, mi)
        * wigner_3j(jb, 2 * rank, jj, -mb, -2 * q, mj)
    )
