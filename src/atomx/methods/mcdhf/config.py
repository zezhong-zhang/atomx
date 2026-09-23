"""Configuration State Functions (CSFs) in jj-coupling.

A CSF specifies an electron configuration: which subshells (n, kappa) are
occupied and by how many electrons. In jj-coupling, each subshell is
characterized by (n, kappa), and the occupation q <= 2j+1.

For MCDHF, we need multiple CSFs that can mix via CI.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from atomx.angular.csf_recoupling import allowed_coupled_J2
from atomx.angular.mscheme import subshell_j_counts_mscheme
from atomx.config.parsing import parse_nr_shells
from atomx.methods.dirac_fock import Subshell
from atomx.methods.mcdhf.exact_ci import coupling_paths


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
    coupling : tuple of (int, int, int), optional
        The sequential coupling tree: one ``(cumulative_total_2J,
        this_subshell_2J, degeneracy_index)`` entry per subshell, in
        ``subshells`` order, with the last cumulative value equal to
        ``total_J2``.  ``None`` (the default, and what hand-built CSFs
        get) means "derive the canonical tree": the first path of
        :func:`atomx.methods.mcdhf.exact_ci.coupling_paths` that reaches
        ``total_J2``.  When several paths reach the same total J they span
        the same block and the canonical choice is just as valid a basis
        vector as any other; only the *number* of paths matters for the
        dimension.  The tree is what makes the exact jj matrix element
        ⟨CSF_r|Ĥ|CSF_s⟩ well defined — ``total_J2`` alone does not
        determine it.
    """
    subshells: List[Subshell]
    total_J2: int = 0
    parity: int = 1
    label: str = ""
    coupling: Optional[Tuple[Tuple[int, int, int], ...]] = None

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


def subshell_allowed_J2(twoj: int, q: int) -> List[int]:
    """Allowed values of ``2J`` for ``q`` electrons in a single (j) subshell.

    Implements the Pauli-allowed J multiplet of ``j^q`` for the cases that
    appear in MCDHF benchmarks of light/medium ions: q ∈ {0, 1, 2j, 2j+1}
    trivially, plus the standard tables for ``(j)^2`` (j = 1/2, 3/2, 5/2,
    7/2, ...) where Pauli antisymmetry restricts J to even values
    J = 0, 2, 4, ..., 2j-1.  Particle-hole symmetry handles the
    ``2j+1 - q`` cases by reflection.

    For higher seniority cases (e.g. (j=5/2)^3 with J ∈ {3/2, 5/2, 9/2})
    the Grant/Cowan tabulated terms are used.  Anything outside the
    supported envelope raises ``NotImplementedError`` — these
    configurations don't currently arise in the benchmark suite.

    Parameters
    ----------
    twoj : int
        Twice the subshell angular momentum j (e.g. 1 for s/p1/2, 3 for
        p3/2 / d3/2, 5 for d5/2).
    q : int
        Number of electrons in the subshell, 0 ≤ q ≤ 2j+1.

    Returns
    -------
    list of int
        Distinct ``2J`` values realised by ``j^q``.  Repeats (multiple
        terms with the same J — possible for j ≥ 9/2) are NOT generated;
        downstream consumers treat each (kappa^q, 2J) tuple as a single
        CSF.  For the Fe XVII / Li-like benchmark grid this is exact.
    """
    if q < 0 or q > twoj + 1:
        raise ValueError(f"q={q} out of range for 2j={twoj}")

    # Closed or empty subshell: J = 0
    if q == 0 or q == twoj + 1:
        return [0]

    # Single electron or single hole: J = j (electron-hole symmetry)
    if q == 1 or q == twoj:
        return [twoj]

    # Two electrons (or two holes by particle-hole symmetry):
    # Pauli-allowed J = 0, 2, 4, ..., 2j-1 (i.e. even 2J from 0 to 2j-2)
    if q == 2 or q == twoj - 1:
        return list(range(0, twoj, 2))

    # (j = 5/2)^3 — Grant Appendix A: J = 3/2, 5/2, 9/2
    if twoj == 5 and q == 3:
        return [3, 5, 9]

    # (j = 7/2)^3 — Grant Appendix A: J = 3/2, 5/2, 7/2, 9/2, 11/2, 15/2
    if twoj == 7 and q == 3:
        return [3, 5, 7, 9, 11, 15]

    # (j = 7/2)^4 — Grant Appendix A: J = 0, 2, 2, 4, 4, 5, 6, 8
    # We collapse repeated J values (seniority distinguishes them but the
    # current CI matrix doesn't carry seniority labels yet).
    if twoj == 7 and q == 4:
        return [0, 2, 4, 5, 6, 8]

    raise NotImplementedError(
        f"Seniority table missing for (2j={twoj})^{q}; extend "
        f"subshell_allowed_J2 with the Grant Appendix A entry."
    )


def _triangle_J2(a: int, b: int) -> List[int]:
    """Return list of 2J values satisfying |a-b| ≤ 2J ≤ a+b in steps of 2."""
    return list(range(abs(a - b), a + b + 1, 2))


def couple_J2_list(twoJ_per_shell: List[int]) -> List[int]:
    """Repeated CG triangle coupling: return list of allowed total 2J.

    Couples a list of subshell ``2J`` values sequentially: the result is
    the multiset of all 2J that can be reached by combining the inputs
    in order.  This is the relativistic analogue of ``j1 ⊗ j2 ⊗ j3 ⊗ …``
    where each ⊗ enumerates all triangle-rule outputs.

    Identical 2J values arising from different recoupling paths are kept
    as separate entries — they index distinct CSFs in the angular basis.
    """
    if not twoJ_per_shell:
        return [0]
    accum: List[int] = [twoJ_per_shell[0]]
    for nxt in twoJ_per_shell[1:]:
        new: List[int] = []
        for cur in accum:
            new.extend(_triangle_J2(cur, nxt))
        accum = new
    return accum


def enumerate_jj_csfs(config_str: str) -> List[CSF]:
    """Enumerate J-resolved jj-coupled CSFs of a single NR configuration.

    For each open NR shell ``nl^q`` with 0 < q < 4l+2, lists every integer
    split ``(q-, q+)`` with ``q- + q+ == q``, ``0 <= q- <= 2l``,
    ``0 <= q+ <= 2l+2``.  For each (split) the Pauli-allowed J of each
    subshell is enumerated (see :func:`subshell_allowed_J2`), and the
    overall total ``2J`` is produced by repeated CG triangle coupling
    across all subshells (see :func:`couple_J2_list`).

    The resulting set is the J-resolved basis on which the CI
    Hamiltonian of the average-of-jj-configurations is defined.
    Diagonalising it block-by-(J, parity) and using the ground-state
    amplitudes cⱼ as weights in

        ϖᵢ = Σⱼ cⱼ² nᵢʲ                               (Guerra Eq 10)

    realises the correct CI-weighted averaged monoconfiguration with
    proper J-multiplet structure.

    Parameters
    ----------
    config_str : str
        Non-relativistic configuration, e.g. ``"1s2 2s2 2p6 6p2"``.

    Returns
    -------
    csfs : list of :class:`CSF`
        One CSF per *(jj-split, sequential coupling path)*.  ``total_J2``
        is ``2J``, ``parity`` is ``(-1)^(Σ l_i q_i)`` and ``coupling``
        records the tree; CSFs sharing a total ``2J`` but differing in an
        intermediate coupling are distinct, orthogonal basis vectors.
        Closed-shell configurations produce exactly one CSF (J=0,
        parity +1).
    """
    # Parse into (n, l, q) shells (shared parser: handles two-digit n and
    # the implicit unit occupation).
    nr_shells = parse_nr_shells(config_str)

    # Parity = (-1)^Σ l_i q_i (intrinsic parity from orbital ℓ × q)
    parity = 1
    for _, l, q in nr_shells:
        if (l * q) % 2:
            parity = -parity

    # Per-shell list of jj-split options. Each option is a list of
    # (n, kappa, occ, twoj) tuples for that NR shell.
    per_shell_options: List[List[List[Tuple[int, int, int, int]]]] = []
    for n, l, q in nr_shells:
        if l == 0:
            # s-shell: single kappa = -1, twoj = 1
            per_shell_options.append([[(n, -1, q, 1)]])
            continue
        kappa_minus = l                 # j = l - 1/2  → 2j = 2l - 1
        kappa_plus = -(l + 1)           # j = l + 1/2  → 2j = 2l + 1
        twoj_minus = 2 * l - 1
        twoj_plus = 2 * l + 1
        max_minus = 2 * l               # 2j- + 1
        options = []
        for q_minus in range(max(0, q - (twoj_plus + 1)), min(q, max_minus) + 1):
            q_plus = q - q_minus
            subs: List[Tuple[int, int, int, int]] = []
            if q_minus > 0:
                subs.append((n, kappa_minus, q_minus, twoj_minus))
            if q_plus > 0:
                subs.append((n, kappa_plus, q_plus, twoj_plus))
            if not subs:
                subs.append((n, kappa_plus, 0, twoj_plus))
            options.append(subs)
        per_shell_options.append(options)

    # Cartesian product across shells: one flat list of subshells per
    # jj-split combination.
    def product(lists):
        if not lists:
            return [[]]
        acc = [[]]
        for choices in lists:
            acc = [prev + choice for prev in acc for choice in choices]
        return acc

    csfs: List[CSF] = []
    for combo in product(per_shell_options):
        subs = []
        labels = []
        for n, kappa, occ, _twoj in combo:
            subs.append(Subshell(n=n, kappa=kappa, occupation=occ))
            labels.append(f"{subs[-1].label}^{occ}")

        # Couple exact antisymmetric subshell-J multiplicities.  This keeps
        # repeated parentage rows with the same total J as distinct CSFs.
        total_counts: dict[int, int] = {0: 1}
        for _, _kappa, occ, twoj in combo:
            total_counts = _couple_j_count_maps(
                total_counts,
                subshell_j_counts_mscheme(twoj, occ),
            )

        # One CSF per sequential coupling path, each carrying its tree.
        # ``coupling_paths`` enumerates the same paths ``total_counts``
        # counts, so the CSF count and the multiset of total 2J values are
        # unchanged; the tree is what makes the exact matrix element
        # <CSF_r|H|CSF_s> well defined.  The cross-check below is the
        # invariant, not a formality: if the two ever disagree the CSF list
        # is no longer a complete basis of the intended space.
        order = tuple((int(sh.n), int(sh.kappa), int(sh.occupation))
                      for sh in subs)
        paths = coupling_paths(order)
        path_counts: dict[int, int] = {}
        for chain in paths:
            path_counts[chain[-1][0]] = path_counts.get(chain[-1][0], 0) + 1
        if path_counts != total_counts:
            raise RuntimeError(
                f"internal inconsistency for configuration "
                f"{' '.join(labels)}: coupling paths give {path_counts} but "
                f"the subshell-2J convolution gives {total_counts}")

        for total_J2, count in sorted(total_counts.items()):
            idx = 0
            for chain in paths:
                if chain[-1][0] != total_J2:
                    continue
                idx += 1
                suffix = f" [J={total_J2}/2]"
                if count > 1:
                    suffix += f" #{idx}"
                suffix += " tree=" + "-".join(
                    f"{cum}:{j2}:{deg}" for cum, j2, deg in chain)
                csfs.append(CSF(
                    subshells=list(subs),
                    total_J2=total_J2,
                    parity=parity,
                    label=" ".join(labels) + suffix,
                    coupling=chain,
                ))
    return csfs


def enumerate_jj_csfs_many(config_strings: List[str]) -> List[CSF]:
    """J-resolved enumeration over a list of NR configuration strings.

    Equivalent to concatenating :func:`enumerate_jj_csfs` over each
    config string in order.  Used by the public ``Atom(..., config=[...])``
    multi-configuration path so that each NR config contributes its full
    J-multiplet to the CI basis.
    """
    csfs: List[CSF] = []
    for s in config_strings:
        csfs.extend(enumerate_jj_csfs(s))
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
    total_counts: dict[int, int] = {0: 1}
    for n, l, q in parse_nr_shells(config_str):
        shell_counts = _nr_shell_j_resolved_counts(n, l, q)
        total_counts = _couple_j_count_maps(total_counts, shell_counts)
    return dict(sorted(total_counts.items()))


def _csf_parity(subshells: List[Subshell]) -> int:
    exponent = 0
    for sh in subshells:
        exponent += int(sh.l * round(sh.occupation))
    return -1 if exponent % 2 else 1


def _parse_config_string(config_str):
    """Parse "1s2 2s2 2p6" into list of Subshell objects.

    Notation: nl^q where n=principal, l=s/p/d/f, q=occupation.
    For relativistic: each nl expands to nl- (j=l-1/2) and nl+ (j=l+1/2).
    """
    l_map = {ch: i for i, ch in enumerate("spdfghiklmno")}
    subshells = []

    for token in config_str.strip().split():
        # Parse nl^q with multi-digit n (e.g. "10d3") and high-l letters.
        m = re.fullmatch(r"(\d+)([spdfghiklmno])(\d+)", token)
        if m is None:
            raise ValueError(f"cannot parse configuration token {token!r}")
        n = int(m.group(1))
        l_char = m.group(2)
        l = l_map[l_char]
        q = int(m.group(3))

        if l == 0:
            # s orbital: only j = l+1/2 = 1/2, kappa = -(l+1) = -1
            subshells.append(Subshell(n=n, kappa=-(l + 1), occupation=q))
        else:
            # Split occupation between j=l-1/2 (kappa=+l) and j=l+1/2 (kappa=-(l+1))
            max_minus = 2 * l  # 2*(l-1/2)+1 = 2l

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
