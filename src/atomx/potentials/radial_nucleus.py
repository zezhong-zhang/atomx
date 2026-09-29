"""Nuclear and environment fields on a shooting radial quadrature.

Fermi shape parameters come from the shared nuclear model. Charge is
normalized on the supplied native SLAG6 grid, preserving the shooting
Hamiltonian convention independently of the many-electron method.
"""
import numpy as np
from atomx.numerics.shooting.quadrature import slag6


def nuclear_rv(Z, r, difr, np_, *, nuclear=None, atomic_weight=0.):
    """Return r V_nuc on the native grid, including the origin limit.

    ``atomic_weight`` preserves the historical shooting convention: a
    positive value selects the finite Fermi profile; otherwise ``nuclear``
    selects point or the element's reference Fermi profile.
    """
    aw = atomic_weight
    if nuclear is not None and not (aw and aw > 0.0):
        if nuclear == "point":
            aw = 0.0
        elif nuclear == "fermi":
            from atomx.potentials.nuclear import _get_mass
            aw = float(_get_mass(Z))
        else:
            raise ValueError(
                f"nuclear must be 'point' or 'fermi' "
                f"(got {nuclear!r})")
    z = float(Z)
    if aw <= 0.5:
        return np.full(np_, -z)
    from atomx.potentials.nuclear import fermi_params
    a_nuc, c_nuc = fermi_params(Z, A=aw)
    arg = np.minimum((c_nuc - r) / a_nuc, 500.0)
    xx = np.where(arg < 500.0, np.exp(arg), 1.0e200)
    denn = xx / (1.0 + xx)
    fwd = slag6(1.0, denn * r ** 2 * difr, np_)
    aux = np.zeros(np_)
    idx = np_ - 1 - np.arange(np_ - 1)
    aux[:np_ - 1] = denn[idx] * r[idx] * difr[idx]
    bwd = slag6(1.0, aux, np_)
    fnorm = z / fwd[np_ - 1] if abs(fwd[np_ - 1]) > 1.0e-30 else z
    return -fnorm * (fwd + bwd[np_ - 1 - np.arange(np_)] * r)


def screened_nuclear_rv(Z, r, difr, np_, *, nuclear=None, atomic_weight=0.,
                        screening_potential=None):
    """Return the common nuclear/environment field for SCF and CI."""
    rv = nuclear_rv(Z, r, difr, np_, nuclear=nuclear, atomic_weight=atomic_weight)
    if screening_potential is not None:
        rv = rv + r * np.asarray(screening_potential(r), dtype=float)
    return rv
