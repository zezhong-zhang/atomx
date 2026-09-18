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
    """Odd double factorial ``n!! = 1*3*5*...*n`` (n odd).

    NOTE the argument convention: this returns ``n!!``, NOT ``(2n+1)!!``.
    To get the ``(2L+1)!!`` factor of the Einstein-A prefactor, call
    ``_double_factorial(2*L + 1)`` (as :func:`einstein_A_eL` does).  Examples:
    ``_double_factorial(3) == 3``, ``_double_factorial(5) == 15``,
    ``_double_factorial(7) == 105``.
    """
    result = 1
    for i in range(1, n + 1, 2):
        result *= i
    return result


def einstein_A_eL(omega, line_strength, L, tj_i):
    """Einstein A-coefficient for electric 2^L-pole transition.

    A_EL = 2 * alpha^(2L+1) * omega^(2L+1) * (L+1)(2L+1)
           / [L * ((2L+1)!!)^2] * S_EL / (tj_i + 1)

    This is Grant (2007) Eq. 8.4.10 in the long-wavelength, Babushkin
    (length) gauge. For L=1 the prefactor collapses to (4/3)*alpha^3,
    giving the textbook Einstein coefficient
    A(E1) = (4/3)*alpha^3*omega^3*S/(2j_i+1).

    ``line_strength`` is |<f || r^L * C^L || i>|^2 in atomic units
    (i.e. the squared reduced matrix element of r^L*C^L^q, exactly
    what :mod:`atomx.properties.transitions.multipole.radial_eL` times
    :func:`~atomx.properties.transitions.angular.angular_eL` produces).

    Reference validation: H 2p_{1/2}->1s_{1/2} with omega=3/8 Ha and
    S = 1.110 returns A = 6.27 x 10^8 s^-1, matching NIST/FAC.

    Parameters
    ----------
    omega : float
        Transition energy in Hartree (positive).
    line_strength : float
        Line strength S_EL = |<f||r^L C^L||i>|^2 in atomic units (a_0^{2L}).
    L : int
        Multipole rank (1=E1, 2=E2, ...).
    tj_i : int
        Twice the initial-state total angular momentum (= 2*j_upper for
        spontaneous emission).

    Returns
    -------
    float
        A-coefficient in s^-1.
    """
    if omega <= 0 or line_strength <= 0:
        return 0.0

    dbl_fact = _double_factorial(2 * L + 1)
    prefactor = (2.0 * ALPHA ** (2 * L + 1) * omega ** (2 * L + 1)
                 * (L + 1) * (2 * L + 1)
                 / (L * dbl_fact ** 2))
    A_au = prefactor * line_strength / (tj_i + 1)
    return A_au * AU_RATE_PER_S


def einstein_A_mL(omega, line_strength, L, tj_i):
    """Einstein A-coefficient for magnetic 2^L-pole transition.

    A_ML = 2 * alpha^(2L+1) * omega^(2L+1) * (L+1)(2L+1)
           / [L * ((2L+1)!!)^2] * S_ML / (tj_i + 1)

    Same Grant (2007) Eq. 8.4.10 prefactor as the electric multipole
    case.  The alpha-suppression of magnetic vs electric rates is
    encoded in ``S_ML``, which uses the magnetic radial integral
    ``R_ML = ∫(P_f Q_i + Q_f P_i) r^L dr`` (no explicit 1/c — the
    suppression comes from Q ~ alpha Z P) combined with the kappa
    angular factor ``(kappa_i + kappa_f)/(2L+1)`` from
    :func:`~atomx.properties.transitions.angular.angular_mL`.

    Reference validation: H-like ``2s_{1/2} -> 1s_{1/2}`` M1 rate
    matches Drake (1971) ``2.5e-6 Z^10 s^-1`` to <1 % for Z <= 10 and
    to ~5 % at Z = 26 (mild drift from relativistic corrections beyond
    the leading Z^10 term).  See ``tests/test_M1_E2_rates.py``.

    Parameters
    ----------
    omega, line_strength, L, tj_i : same as :func:`einstein_A_eL`.

    Returns
    -------
    float
        A-coefficient in s^-1.
    """
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
