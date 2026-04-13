"""Closed-form hydrogenic Dirac energy eigenvalues for validation."""

import numpy as np
from atomx.constants import ALPHA, C_SQ


def hydrogenic_dirac_energy(n: int, kappa: int, Z: int) -> float:
    """Exact Dirac energy eigenvalue for a hydrogenic ion.

    E_{n,kappa} = c^2 * [1 + (alpha*Z)^2 / (n_r + gamma)^2]^{-1/2} - c^2

    where:
        gamma = sqrt(kappa^2 - (alpha*Z)^2)
        n_r = n - |kappa|  (radial quantum number)

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    kappa : int
        Relativistic angular momentum quantum number (kappa != 0).
        kappa = -(l+1) for j = l + 1/2
        kappa = +l     for j = l - 1/2
    Z : int
        Nuclear charge.

    Returns
    -------
    E : float
        Binding energy in Hartree (negative for bound states).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if kappa == 0:
        raise ValueError("kappa cannot be zero")

    aZ = ALPHA * Z
    gamma = np.sqrt(kappa * kappa - aZ * aZ)
    n_r = n - abs(kappa)

    if n_r < 0:
        raise ValueError(f"Invalid quantum numbers: n={n}, kappa={kappa} "
                         f"gives n_r = {n_r} < 0")

    denom = n_r + gamma
    E = C_SQ * (1.0 / np.sqrt(1.0 + (aZ / denom) ** 2) - 1.0)
    return E


def list_hydrogenic_states(n_max: int, Z: int):
    """List all hydrogenic states up to principal quantum number n_max.

    Parameters
    ----------
    n_max : int
        Maximum principal quantum number.
    Z : int
        Nuclear charge.

    Returns
    -------
    states : list of dict
        Each dict has keys: n, l, j, kappa, energy_hartree.
    """
    states = []
    for n in range(1, n_max + 1):
        for l in range(n):
            # j = l + 1/2  → kappa = -(l+1)
            kappa_minus = -(l + 1)
            E = hydrogenic_dirac_energy(n, kappa_minus, Z)
            states.append({
                "n": n, "l": l, "j": l + 0.5,
                "kappa": kappa_minus, "energy_hartree": E,
            })

            # j = l - 1/2  → kappa = +l (only for l >= 1)
            if l >= 1:
                kappa_plus = l
                E = hydrogenic_dirac_energy(n, kappa_plus, Z)
                states.append({
                    "n": n, "l": l, "j": l - 0.5,
                    "kappa": kappa_plus, "energy_hartree": E,
                })
    return states
