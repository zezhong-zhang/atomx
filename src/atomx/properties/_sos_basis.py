"""Shared sum-over-states basis construction for properties."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from atomx.angular.recoupling import l_of_kappa


def build_sos_virtual_basis(
    scf_result,
    *,
    cache,
    kappa_max: int = 4,
    n_virt_max: Optional[int] = None,
    e_virt_max: Optional[float] = None,
    potential_mode: str = "VN-1",
) -> Tuple[
    List[int],
    Dict[int, List[Tuple[np.ndarray, np.ndarray, float, str]]],
    Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]],
]:
    """Build the V^{N-1} sum-over-states virtual orbital basis for all kappa.

    Returns ``(kappa_list, occ_basis, virt_basis)`` where:
    - ``kappa_list`` contains signed kappa values with l <= ``kappa_max``,
      built exactly as the previous inlined loops did.
    - ``occ_basis`` maps kappa to ``(P, Q, eps, label)`` occupied orbitals.
      Labels come from matching ``scf_result.subshells`` by position within
      kappa, with eps taken from ``scf_result.orbital_energies`` when
      available, otherwise the V^{N-1} eigenvalue.
    - ``virt_basis`` maps kappa to ``(P, Q, eps)`` virtual orbitals,
      truncated to the ``n_virt_max`` lowest-energy virtuals when set.
    """
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        kappa_list.append(+k_abs)
    kappa_list = [
        k for k in kappa_list if abs(k) > 0 and l_of_kappa(k) <= kappa_max
    ]

    occ_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float, str]]] = {}
    virt_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {}

    # Deferred import keeps properties -> methods off the module-load path.
    from atomx.methods.mbpt import build_virtual_basis

    for kappa in kappa_list:
        occ, virt = build_virtual_basis(
            scf_result, kappa, cache=cache, e_max=e_virt_max,
            potential_mode=potential_mode
        )
        kappa_subshells = [s for s in scf_result.subshells if s.kappa == kappa]
        occ_with_labels: List[Tuple[np.ndarray, np.ndarray, float, str]] = []
        for i, (P, Q, _eps_v_n1) in enumerate(occ):
            if i < len(kappa_subshells):
                sh = kappa_subshells[i]
                lab = sh.label
                eps_use = scf_result.orbital_energies.get(lab, _eps_v_n1)
            else:
                lab = f"occ-{kappa}-{i}"
                eps_use = _eps_v_n1
            occ_with_labels.append((P, Q, eps_use, lab))
        if n_virt_max is not None:
            virt = virt[:n_virt_max]
        occ_basis[kappa] = occ_with_labels
        virt_basis[kappa] = virt

    return kappa_list, occ_basis, virt_basis
