"""Einstein A-coefficients, oscillator strengths, and lifetimes.

Formulas follow Grant (2007), "Relativistic Quantum Theory of Atoms
and Molecules", and Johnson (2007), "Atomic Structure and Lifetimes".

Units:
- All inputs in atomic units (Hartree, Bohr)
- A-coefficients returned in s^-1
- Oscillator strengths are dimensionless
- Lifetimes in seconds
"""

import numpy as np
from atomx.constants import ALPHA, AU_RATE_PER_S


def _double_factorial(n):
    """Compute (2n+1)!! = 1*3*5*...*(2n+1)."""
    result = 1
    for i in range(1, n + 1, 2):
        result *= i
    return result


def einstein_A_eL(omega, line_strength, L, tj_i):
    """Einstein A-coefficient for electric 2^L-pole transition.

    A_EL = 2*alpha*omega^(2L+1) * (L+1) / [L * ((2L+1)!!)^2]
           * S_EL / (tj_i + 1)

    where S_EL is the line strength and tj_i = 2*j_i (twice initial j).

    Parameters
    ----------
    omega : float
        Transition energy in Hartree (positive).
    line_strength : float
        Line strength S = |<f||T^L||i>|^2 in atomic units (a_0^{2L}).
    L : int
        Multipole rank (1=E1, 2=E2, ...).
    tj_i : int
        Twice the initial-state total angular momentum.

    Returns
    -------
    float
        A-coefficient in s^-1.
    """
    if omega <= 0 or line_strength <= 0:
        return 0.0

    dbl_fact = _double_factorial(2 * L + 1)
    prefactor = (2.0 * ALPHA * omega ** (2 * L + 1) * (L + 1)
                 / (L * dbl_fact ** 2))
    A_au = prefactor * line_strength / (tj_i + 1)
    return A_au * AU_RATE_PER_S


def einstein_A_mL(omega, line_strength, L, tj_i):
    """Einstein A-coefficient for magnetic 2^L-pole transition.

    A_ML = 2*alpha*omega^(2L+1) * (L+1) / [L * ((2L+1)!!)^2]
           * S_ML / (tj_i + 1)

    Same formula as EL but the line strength contains the 1/c factor
    from the magnetic radial matrix element.

    Parameters
    ----------
    omega, line_strength, L, tj_i : same as einstein_A_eL

    Returns
    -------
    float
        A-coefficient in s^-1.
    """
    # Same functional form; the 1/c is already in the radial ME
    return einstein_A_eL(omega, line_strength, L, tj_i)


def oscillator_strength(omega, line_strength, L, tj_i, tj_f):
    """Absorption oscillator strength f_{i->f} for electric multipole.

    gf = (2*omega) / (2L+1) * S_EL    (weighted oscillator strength)
    f = gf / (tj_i + 1)

    Parameters
    ----------
    omega : float
        Transition energy in Hartree (positive).
    line_strength : float
    L : int
    tj_i, tj_f : int
        Twice the angular momenta.

    Returns
    -------
    f : float
        Absorption oscillator strength (dimensionless).
    gf : float
        Weighted oscillator strength g_i * f.
    """
    if omega <= 0 or line_strength <= 0:
        return 0.0, 0.0

    gf = (2.0 * omega / (2 * L + 1)) * line_strength
    f = gf / (tj_i + 1)
    return f, gf


def lifetime(A_rates):
    """Compute the total radiative lifetime from a list of A-coefficients.

    tau = 1 / sum(A_k)

    Parameters
    ----------
    A_rates : list or array of float
        Individual A-coefficients in s^-1.

    Returns
    -------
    float
        Lifetime in seconds. Returns inf if no decay channels.
    """
    total = sum(A for A in A_rates if A > 0)
    if total <= 0:
        return np.inf
    return 1.0 / total


def branching_ratio(A_channel, A_total):
    """Branching ratio for a specific decay channel.

    Parameters
    ----------
    A_channel : float
        A-coefficient for this channel in s^-1.
    A_total : float
        Total A-coefficient (sum over all channels) in s^-1.

    Returns
    -------
    float
        Branching ratio (0 to 1).
    """
    if A_total <= 0:
        return 0.0
    return A_channel / A_total
