"""Per-orbital QED energy shifts for scattering energetics.

The scattering/autoionization machinery works from one-body orbital
energies (thresholds, resonance positions via the determinant-CI
Hamiltonian diagonal).  atomx's DR resonance energies are pure Coulomb
by design and sit ~8 eV from FAC-with-QED for Fe-K; this module exposes
the per-electron QED shift of each orbital so callers can fold it into
``orbital_energies`` before building thresholds or a
:class:`~atomx.scattering.autoionization.mscheme_ci.ResonanceManifold`:

    shifts = qed_orbital_shifts(atom._result, Z=atom.Z)
    eps = {lab: e + shifts.get(lab, 0.0)
           for lab, e in atom._result.orbital_energies.items()}

Self-energy comes from the qedmod backend registry (per-orbital values
are occupation-weighted there; divided out here), vacuum polarization
as the per-orbital Uehling expectation value.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def qed_orbital_shifts(result, *, Z=None, se_method: str = "auto",
                       include_vp: bool = True) -> Dict[str, float]:
    """Per-electron QED shift of each bound orbital, Hartree.

    Positive shift = orbital pushed up (binding reduced), the usual sign
    for s-state self-energy.  Keys match ``result.orbital_energies``.
    """
    from atomx.corrections.adapters import _flatten_scf_orbitals
    from atomx.corrections.qed import resolve_se_method, uehling_potential
    from atomx.corrections import qedmod
    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

    bsp = getattr(result, "_bsp_data", None) or getattr(
        result, "bsp_data", None)
    if bsp is None:
        raise ValueError("result carries no B-spline data")
    Z = int(Z if Z is not None else result.Z)
    orbitals = _flatten_scf_orbitals(result)

    shifts: Dict[str, float] = {}

    method = resolve_se_method(se_method, Z)
    backend = qedmod.get_backend(method)
    se = backend.compute(orbitals, bsp, Z)
    for (P, Q, occ, kappa, label) in orbitals:
        if occ > 0 and label in se.per_orbital:
            shifts[label] = se.per_orbital[label] / occ

    if include_vp:
        nv, ns = bsp["nv"], bsp["ns"]
        gr, grw = bsp["gr"], bsp["grw"]
        ksp = bsp.get("ksp", bsp["ks"])
        ksq = bsp.get("ksq", bsp["ks"])
        nsp = bsp.get("nsp", ns)
        nsq = bsp.get("nsq", ns)
        bsp_p = bsp["bsp_p"]
        bsp_q = bsp.get("bsp_q", bsp_p)
        V_vp = uehling_potential(gr, Z)
        for (P, Q, occ, kappa, label) in orbitals:
            fP = eval_orbital_on_grid(P, bsp_p, nv, ksp, nsp)
            fQ = eval_orbital_on_grid(Q, bsp_q, nv, ksq, nsq)
            rho = fP * fP + fQ * fQ
            shifts[label] = (shifts.get(label, 0.0)
                             + float(np.sum(grw * rho * V_vp)))

    return shifts


def shifted_orbital_energies(result, *, Z=None, **kw) -> Dict[str, float]:
    """``result.orbital_energies`` with QED shifts folded in (Hartree)."""
    shifts = qed_orbital_shifts(result, Z=Z, **kw)
    return {lab: float(e) + shifts.get(lab, 0.0)
            for lab, e in result.orbital_energies.items()}


__all__ = ["qed_orbital_shifts", "shifted_orbital_energies"]
