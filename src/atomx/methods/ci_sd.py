"""Complete declared NR singles/doubles spaces for finite native CI.

This selects a configuration space, not coupled-cluster amplitude equations.
Every allowed relativistic occupation and jj coupling tree of each retained
NR configuration is kept. Frozen shells, target orbitals, reference union,
N/J/parity and resource caps are explicit. No energy-driven selection occurs.
"""
from dataclasses import dataclass

import numpy as np

from atomx.angular.csf_hamiltonian import csf_signature
from atomx.config.csf import count_j_resolved_csfs, enumerate_jj_csfs_many
from atomx.config.parsing import parse_nr_shells
from atomx.config.subshell import L_LETTERS
from atomx.methods.mcdhf.excitations import generate_sd_excitations

__all__ = ["SDCISpace", "plan_sd_ci_space", "build_sd_ci_effective_hamiltonian"]


def _configuration(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty reference configuration required")
    shells = parse_nr_shells(text)
    if len({(n, l) for n, l, _ in shells}) != len(shells):
        raise ValueError("duplicate reference shell")
    return " ".join(f"{n}{L_LETTERS[l]}{q}" for n, l, q in sorted(shells) if q)


def _labels(values):
    if isinstance(values, str):
        raise ValueError("orbital labels require a sequence, not one string")
    labels = tuple(values)
    for label in labels:
        shells = parse_nr_shells(label)
        if len(shells) != 1:
            raise ValueError("one nl orbital label required")
        n, l, q = shells[0]
        if q != 1 or label != f"{n}{L_LETTERS[l]}":
            raise ValueError("orbital labels must omit occupations")
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate orbital label")
    return tuple(sorted(labels, key=lambda s: parse_nr_shells(s)[0][:2]))


@dataclass(frozen=True)
class SDCISpace:
    """Preflight identity with exact J-CSF count and fixed-M size bound.

    Counts use angular multiplicities before CSF vectors or Hamiltonians
    are allocated. ``materialize`` independently checks the complete CSF
    enumeration; callers cannot replace a failed cap with a selected subset.
    """
    reference_configurations: tuple
    configurations: tuple
    target_orbitals: tuple
    frozen_orbitals: tuple
    max_excitations: int
    electrons: int
    J2: int
    parity: int
    n_csfs: int
    determinant_bound: int
    dimension_cap: int
    determinant_cap: int

    def materialize(self):
        sector = (self.J2, self.parity)
        csfs = tuple(c for c in enumerate_jj_csfs_many(self.configurations)
                     if (c.total_J2, c.parity) == sector)
        identities = tuple(csf_signature(c).key for c in csfs)
        if len(csfs) != self.n_csfs or len(set(identities)) != len(identities):
            raise RuntimeError("complete SD CSF enumeration disagrees with angular preflight")
        reference = {csf_signature(c).key for c in enumerate_jj_csfs_many(self.reference_configurations)
                     if (c.total_J2, c.parity) == sector}
        p = tuple(i for i, key in enumerate(identities) if key in reference)
        if not p or len(p) != len(reference):
            raise RuntimeError("SD reference space missing in complete enumeration")
        return csfs, p


def plan_sd_ci_space(reference_configurations, target_orbitals, *, frozen_orbitals=(),
                     J2, parity, max_excitations=2, dimension_cap=256,
                     max_determinants=2000):
    """Plan a complete reference-union NR excitation space at one symmetry.

    Singles/doubles refer to moves between NR shells; each NR occupation
    retains all its relativistic parentage. Thus this may be larger than
    determinant SD from one chosen jj reference. It is not linearized
    atomic all-order SD, CCSD, FS-CCSDT or complete radial convergence.
    """
    if isinstance(reference_configurations, str):
        raise ValueError("reference configurations require a sequence")
    references = tuple(sorted({_configuration(c) for c in reference_configurations}))
    if not references:
        raise ValueError("nonempty reference configurations required")
    if (isinstance(max_excitations, bool) or max_excitations not in (1, 2)
            or not isinstance(max_excitations, (int, np.integer))):
        raise ValueError("only explicit singles or singles/doubles supported")
    for name, value in (("dimension_cap", dimension_cap), ("max_determinants", max_determinants)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"positive integer {name} required")
    if (isinstance(J2, bool) or not isinstance(J2, (int, np.integer)) or J2 < 0
            or isinstance(parity, bool) or not isinstance(parity, (int, np.integer))
            or parity not in (-1, 1)):
        raise ValueError("valid integer J2 and parity required")
    occupations = [{f"{n}{L_LETTERS[l]}": q for n, l, q in parse_nr_shells(c)} for c in references]
    totals = {sum(q.values()) for q in occupations}
    if len(totals) != 1:
        raise ValueError("reference configurations must share electron count")
    electrons = totals.pop()
    if not electrons or (J2-electrons) % 2:
        raise ValueError("J2 must agree with electron-number parity")
    targets = _labels(target_orbitals)
    frozen = _labels(frozen_orbitals)
    if any(label not in occupations[0] or any(q.get(label) != occupations[0][label] for q in occupations)
           for label in frozen):
        raise ValueError("frozen shells require identical occupied references")
    if set(targets) & set(frozen):
        raise ValueError("frozen shells cannot be excitation targets")
    configurations = tuple(sorted({_configuration(c) for ref in references
                                   for c in generate_sd_excitations(ref, targets, frozen, max_excitations)}))
    n_csfs = n_determinants = 0
    for c in configurations:
        if (-1)**sum(l*q for _, l, q in parse_nr_shells(c)) != parity:
            continue
        counts = count_j_resolved_csfs(c)
        n_csfs += counts.get(J2, 0)
        n_determinants += sum(n for j, n in counts.items() if j >= J2)
    if n_csfs > dimension_cap or n_determinants > max_determinants:
        raise ValueError(f"complete SD space needs CSFs={n_csfs}, determinants={n_determinants}; "
                         f"caps={dimension_cap}/{max_determinants}")
    if not n_csfs or not any((-1)**sum(l*q for _, l, q in parse_nr_shells(c)) == parity
                            and count_j_resolved_csfs(c).get(J2, 0) for c in references):
        raise ValueError("reference configurations do not contain requested N/J/parity")
    return SDCISpace(references, configurations, targets, frozen, int(max_excitations),
                     electrons, int(J2), int(parity), n_csfs, n_determinants,
                     int(dimension_cap), int(max_determinants))


def build_sd_ci_effective_hamiltonian(space, orbitals_dict, bsp_data, Z, ns, *,
                                     reference_identity, nuclear_potential=None,
                                     include_breit=False, radial_snapshot=None):
    """Materialize a declared SD plan and use shared finite P/Q elimination."""
    from atomx.methods.ci_effective import build_finite_ci_effective_hamiltonian
    if not isinstance(space, SDCISpace):
        raise TypeError("SDCISpace plan required")
    csfs, p = space.materialize()
    return build_finite_ci_effective_hamiltonian(
        csfs, p, orbitals_dict, bsp_data, Z, ns, reference_identity=reference_identity,
        nuclear_potential=nuclear_potential, dimension_cap=space.dimension_cap,
        max_determinants=space.determinant_cap, include_breit=include_breit,
        radial_snapshot=radial_snapshot)
