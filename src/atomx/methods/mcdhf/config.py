"""Configuration State Functions (CSFs) in jj-coupling.

A CSF specifies an electron configuration: which subshells (n, kappa) are
occupied and by how many electrons. In jj-coupling, each subshell is
characterized by (n, kappa), and the occupation q <= 2j+1.

For MCDHF, we need multiple CSFs that can mix via CI.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from atomx.angular.csf_recoupling import allowed_coupled_J2
from atomx.angular.mscheme import subshell_j_counts_mscheme
from atomx.methods.dirac_fock import Subshell


@dataclass
class CSF:
    """A Configuration State Function in jj-coupling.

    Attributes
    ----------
    subshells : list of Subshell
        Occupied subshells with occupations.
    total_J2 : int
        Twice the total angular momentum J (for state labeling).
    parity : int
        +1 (even) or -1 (odd).
    label : str
        Human-readable label (e.g., "1s2 2s2 2p6").
    """
    subshells: List[Subshell]
    total_J2: int = 0
    parity: int = 1
    label: str = ""

    @property
    def n_electrons(self):
        return sum(sh.occupation for sh in self.subshells)

    def occupation_vector(self, orbital_basis):
        """Return occupation array aligned with a shared orbital basis.

        Parameters
        ----------
        orbital_basis : list of (n, kappa) tuples
            The shared orbital set.

        Returns
        -------
        q : ndarray
            Occupation per orbital in the basis.
        """
        q = np.zeros(len(orbital_basis))
        for sh in self.subshells:
            key = (sh.n, sh.kappa)
            for i, (n, k) in enumerate(orbital_basis):
                if (n, k) == key:
                    q[i] = sh.occupation
                    break
        return q

    def differs_from(self, other):
        """Find subshells that differ between self and other.

        Returns
        -------
        diffs : list of ((n, kappa), q_self, q_other) tuples
            Subshells where occupations differ.
        """
        # Build occupation dicts
        occ_self = {(sh.n, sh.kappa): sh.occupation for sh in self.subshells}
        occ_other = {(sh.n, sh.kappa): sh.occupation for sh in other.subshells}

        all_keys = set(occ_self.keys()) | set(occ_other.keys())
        diffs = []
        for key in sorted(all_keys):
            qs = occ_self.get(key, 0)
            qo = occ_other.get(key, 0)
            if qs != qo:
                diffs.append((key, qs, qo))
        return diffs

    def n_electron_diff(self, other):
        """Number of electrons that must be moved to go from self to other."""
        diffs = self.differs_from(other)
        moved = sum(abs(qs - qo) for _, qs, qo in diffs)
        return moved // 2  # Each moved electron changes 2 subshells


def make_csf_list(Z, config_strings):
    """Create CSF list from configuration strings.

    Parameters
    ----------
    Z : int
        Nuclear charge (for validation).
    config_strings : list of str
        e.g. ["1s2 2s2 2p6", "1s2 2s1 2p6 3s1"]

    Returns
    -------
    csfs : list of CSF
    """
    csfs = []
    for s in config_strings:
        subshells = _parse_config_string(s)
        csf = CSF(subshells=subshells, label=s)
        csfs.append(csf)
    return csfs


def enumerate_jj_csfs(config_str: str) -> List[CSF]:
    """Enumerate all jj-coupled CSFs of a single NR configuration string.

    For each open NR shell ``nl^q`` with 0 < q < 4l+2, lists every integer
    split ``(q-, q+)`` with ``q- + q+ == q``, ``0 <= q- <= 2l``,
    ``0 <= q+ <= 2l+2``. Closed shells and s-shells contribute exactly one
    CSF. The resulting set is the basis on which the CI Hamiltonian of the
    "average-of-jj-configurations" is defined: diagonalising it and using
    the ground-state amplitudes cⱼ as weights in

        ϖᵢ = Σⱼ cⱼ² nᵢʲ                               (Guerra Eq 10)

    realises the correct CI-weighted averaged monoconfiguration — as
    opposed to the binomial statistical or equal-weight surrogates.

    Parameters
    ----------
    config_str : str
        Non-relativistic configuration, e.g. ``"1s2 2s2 2p6 6p2"``.

    Returns
    -------
    csfs : list of :class:`CSF`
        One CSF per jj-split combination. For fully closed configurations
        the list has length 1.
    """
    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    # Parse into (n, l, q) shells
    nr_shells = []
    for token in config_str.strip().split():
        n = int(token[0])
        l = l_map[token[1]]
        q = int(token[2:])
        nr_shells.append((n, l, q))

    # Per-shell list of (n, kappa, occ) tuples for each valid jj-split
    per_shell_options: List[List[List[Tuple[int, int, int]]]] = []
    for n, l, q in nr_shells:
        if l == 0:
            # s-shell: single kappa = -1
            per_shell_options.append([[(n, -1, q)]])
            continue
        kappa_minus = l                 # j = l - 1/2
        kappa_plus = -(l + 1)           # j = l + 1/2
        max_minus = 2 * l               # 2j- + 1
        max_plus = 2 * (l + 1)          # 2j+ + 1
        options = []
        for q_minus in range(max(0, q - max_plus), min(q, max_minus) + 1):
            q_plus = q - q_minus
            subs = []
            if q_minus > 0:
                subs.append((n, kappa_minus, q_minus))
            if q_plus > 0:
                subs.append((n, kappa_plus, q_plus))
            # Even if both are zero (shouldn't happen for q>0) still add for completeness
            if not subs:
                subs.append((n, kappa_plus, 0))
            options.append(subs)
        per_shell_options.append(options)

    # Cartesian product across shells
    def product(lists):
        if not lists:
            return [[]]
        acc = [[]]
        for choices in lists:
            acc = [prev + choice for prev in acc for choice in choices]
        return acc

    combos = product(per_shell_options)

    csfs: List[CSF] = []
    for combo in combos:
        subs = []
        labels = []
        for n, kappa, occ in combo:
            subs.append(Subshell(n=n, kappa=kappa, occupation=occ))
            labels.append(f"{subs[-1].label}^{occ}")
        csfs.append(CSF(subshells=subs, label=" ".join(labels)))
    return csfs


def _couple_j_count_maps(
    left: dict[int, int],
    right: dict[int, int],
) -> dict[int, int]:
    out: dict[int, int] = {}
    for Jl2, nl in left.items():
        for Jr2, nr in right.items():
            for J2 in allowed_coupled_J2(Jl2, Jr2):
                out[J2] = out.get(J2, 0) + nl * nr
    return dict(sorted(out.items()))


def _nr_shell_j_resolved_counts(n: int, l: int, q: int) -> dict[int, int]:
    if l == 0:
        return subshell_j_counts_mscheme(1, q)

    jminus2 = 2 * l - 1
    jplus2 = 2 * l + 1
    cap_minus = jminus2 + 1
    cap_plus = jplus2 + 1
    out: dict[int, int] = {}
    for q_minus in range(max(0, q - cap_plus), min(q, cap_minus) + 1):
        q_plus = q - q_minus
        minus_counts = subshell_j_counts_mscheme(jminus2, q_minus)
        plus_counts = subshell_j_counts_mscheme(jplus2, q_plus)
        shell_counts = _couple_j_count_maps(minus_counts, plus_counts)
        for J2, count in shell_counts.items():
            out[J2] = out.get(J2, 0) + count
    return dict(sorted(out.items()))


def count_j_resolved_csfs(config_str: str) -> dict[int, int]:
    """Count antisymmetric jj-coupled CSFs by total ``2J``.

    The input accepts FAC-style non-relativistic shell tokens such as
    ``"2p2"`` or ``"1s2 2s2 2p6"``.  Each ``nl^q`` shell is split into its
    relativistic ``j=l±1/2`` subshells, exact m-scheme multiplicities are used
    inside each subshell, and shell angular momenta are convolved across the
    configuration.
    """
    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    total_counts: dict[int, int] = {0: 1}
    for token in config_str.strip().split():
        if len(token) < 3:
            raise ValueError(f"Invalid configuration token {token!r}")
        n = int(token[0])
        l_char = token[1]
        if l_char not in l_map:
            raise ValueError(f"Unsupported orbital letter {l_char!r}")
        q = int(token[2:])
        shell_counts = _nr_shell_j_resolved_counts(n, l_map[l_char], q)
        total_counts = _couple_j_count_maps(total_counts, shell_counts)
    return dict(sorted(total_counts.items()))


def _csf_parity(subshells: List[Subshell]) -> int:
    exponent = 0
    for sh in subshells:
        exponent += int(sh.l * round(sh.occupation))
    return -1 if exponent % 2 else 1


def enumerate_j_resolved_csfs(config_str: str) -> List[CSF]:
    """Enumerate J-resolved CSF placeholders for a configuration.

    This expands each jj occupation split into the exact antisymmetric
    subshell ``J`` multiplicities from the m-scheme backend, then couples
    subshell angular momenta to total ``J``.  Multiple parentage states with
    the same occupation and total ``J`` are represented by distinct CSF rows
    with stable labels.
    """
    out: List[CSF] = []
    for jj_csf in enumerate_jj_csfs(config_str):
        total_counts: dict[int, int] = {0: 1}
        for sh in jj_csf.subshells:
            q = int(round(sh.occupation))
            total_counts = _couple_j_count_maps(
                total_counts,
                subshell_j_counts_mscheme(sh.j2, q),
            )
        parity = _csf_parity(jj_csf.subshells)
        for J2, count in sorted(total_counts.items()):
            for idx in range(count):
                suffix = f" J={J2}/2"
                if count > 1:
                    suffix += f" #{idx + 1}"
                out.append(CSF(
                    subshells=list(jj_csf.subshells),
                    total_J2=J2,
                    parity=parity,
                    label=f"{jj_csf.label}{suffix}",
                ))
    return out


def _parse_config_string(config_str):
    """Parse "1s2 2s2 2p6" into list of Subshell objects.

    Notation: nl^q where n=principal, l=s/p/d/f, q=occupation.
    For relativistic: each nl expands to nl- (j=l-1/2) and nl+ (j=l+1/2).
    """
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
    subshells = []

    for token in config_str.strip().split():
        n = int(token[0])
        l_char = token[1]
        l = l_map[l_char]
        q = int(token[2:])

        if l == 0:
            # s orbital: only j = l+1/2 = 1/2, kappa = -(l+1) = -1
            subshells.append(Subshell(n=n, kappa=-(l + 1), occupation=q))
        else:
            # Split occupation between j=l-1/2 (kappa=+l) and j=l+1/2 (kappa=-(l+1))
            max_minus = 2 * l  # 2*(l-1/2)+1 = 2l
            max_plus = 2 * (l + 1)  # 2*(l+1/2)+1 = 2l+2

            if q <= max_minus:
                subshells.append(Subshell(n=n, kappa=l, occupation=q))
            else:
                subshells.append(Subshell(n=n, kappa=l, occupation=max_minus))
                subshells.append(Subshell(n=n, kappa=-(l + 1),
                                          occupation=q - max_minus))

    return subshells


def get_shared_orbital_basis(csfs):
    """Extract the union of all orbitals across CSFs.

    Returns
    -------
    basis : list of (n, kappa) tuples, sorted by (n, |kappa|)
    """
    all_orbs = set()
    for csf in csfs:
        for sh in csf.subshells:
            all_orbs.add((sh.n, sh.kappa))
    return sorted(all_orbs, key=lambda nk: (nk[0], abs(nk[1]), -nk[1]))
