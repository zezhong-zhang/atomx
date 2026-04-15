"""Configuration State Functions (CSFs) in jj-coupling.

A CSF specifies an electron configuration: which subshells (n, kappa) are
occupied and by how many electrons. In jj-coupling, each subshell is
characterized by (n, kappa), and the occupation q <= 2j+1.

For MCDHF, we need multiple CSFs that can mix via CI.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

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
