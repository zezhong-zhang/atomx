"""Angular coupling coefficients for radiative transitions.

Selection rules and reduced matrix elements for electric (EL) and
magnetic (ML) multipole transitions in jj-coupling.

All angular momentum arguments follow the "twice-j" convention used
throughout AtomX: j=1/2 -> tj=1, j=1 -> tj=2, etc.

Key selection rules:
- EL: l_i + l_f + L must be odd (parity change), |j_i - j_f| <= L <= j_i + j_f
- ML: l_i + l_f + L must be even (no parity change), |j_i - j_f| <= L <= j_i + j_f
"""

import numpy as np

from atomx.angular.wigner import wigner_3j


def _kappa_to_lj(kappa):
    """Convert kappa to (l, j) as integers and twice-j."""
    l = kappa if kappa > 0 else -kappa - 1
    tj = 2 * abs(kappa) - 1  # twice j
    return l, tj


def selection_rules(kappa_i, kappa_f, L, multipole_type='E'):
    """Check if a transition is allowed by selection rules.

    Parameters
    ----------
    kappa_i, kappa_f : int
        Initial and final state kappa quantum numbers.
    L : int
        Multipole rank (1=dipole, 2=quadrupole, ...).
    multipole_type : str
        'E' for electric, 'M' for magnetic.

    Returns
    -------
    bool
        True if transition is allowed.
    """
    l_i, tj_i = _kappa_to_lj(kappa_i)
    l_f, tj_f = _kappa_to_lj(kappa_f)

    # Triangle condition: |j_i - j_f| <= L <= j_i + j_f
    tL = 2 * L
    if tL < abs(tj_i - tj_f) or tL > tj_i + tj_f:
        return False

    # Parity selection rule
    # Electric EL operator has parity (-1)^L; requires pi_i * (-1)^L * pi_f = +1
    # => (-1)^(l_i + l_f + L) = +1 => l_i + l_f + L must be EVEN
    # Magnetic ML operator has parity (-1)^(L+1); => l_i + l_f + L must be ODD
    parity_sum = l_i + l_f + L
    if multipole_type == 'E':
        return parity_sum % 2 == 0
    elif multipole_type == 'M':
        return parity_sum % 2 == 1
    else:
        raise ValueError(f"Unknown multipole type: {multipole_type}")


def angular_eL(kappa_i, kappa_f, L):
    """Reduced angular matrix element for electric 2^L-pole transition.

    Computes the angular factor for the EL transition i -> f:
        C_EL = (-1)^(j_f+1/2) * sqrt((2j_i+1)(2j_f+1))
               * 3j(j_i, L, j_f; -1/2, 0, 1/2)

    This is the standard reduced matrix element in the Babushkin gauge.

    Parameters
    ----------
    kappa_i, kappa_f : int
    L : int

    Returns
    -------
    float
        Angular factor (0 if forbidden).
    """
    if not selection_rules(kappa_i, kappa_f, L, 'E'):
        return 0.0

    l_i, tj_i = _kappa_to_lj(kappa_i)
    l_f, tj_f = _kappa_to_lj(kappa_f)

    # 3j symbol: (j_i  L  j_f; -1/2  0  1/2)
    # In twice-j convention: (tj_i  2L  tj_f; -1  0  1)
    tL = 2 * L
    threej = wigner_3j(tj_i, tL, tj_f, -1, 0, 1)

    if abs(threej) < 1e-20:
        return 0.0

    # Phase: (-1)^(j_f + 1/2) = (-1)^((tj_f+1)/2)
    phase_exp = (tj_f + 1) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0

    return phase * np.sqrt((tj_i + 1.0) * (tj_f + 1.0)) * threej


def angular_mL(kappa_i, kappa_f, L):
    """Reduced angular matrix element for magnetic 2^L-pole transition.

    For M_L the reduced matrix element factorises as

        ⟨f‖t^M_L‖i⟩ = ⟨-kappa_f‖C^L‖kappa_i⟩ · (kappa_i + kappa_f)/(L+1) · R_ML

    where ``R_ML`` is the long-wavelength radial integral returned by
    :func:`~atomx.properties.transitions.multipole.radial_mL` (no 1/c factor).

    This function returns the angular part ``⟨f‖C^L‖i⟩ · (kappa_i +
    kappa_f)/(L+1)``, so that the line strength used by
    :func:`~atomx.properties.transitions.rates.einstein_A_mL` is simply
    ``(angular_mL * radial_mL) ** 2``.

    Convention notes:

    * The factorisation comes from Grant (2007) Eq. 8.4.17 / FAC
      ``TRRateMultipole`` Eq. 38 in the long-wavelength limit, where the
      magnetic radial operator reduces to ``r^L`` times the kappa
      structure shown above. The combination ``2(2L+1)(L+1)/[L((2L+1)!!)^2]
      · alpha^(2L+1) · omega^(2L+1)`` in ``einstein_A_mL`` is the same
      prefactor as ``einstein_A_eL`` (Grant Eq. 8.4.10), and is shared
      between E and M multipoles when matrix elements use this
      convention.
    * Inhibited M1 transitions require photon retardation even at small Z:
      the leading long-wavelength amplitude cancels. Matching one such
      rate cannot calibrate the angular factor for other transitions.

    Parameters
    ----------
    kappa_i, kappa_f : int
    L : int

    Returns
    -------
    float
        Angular factor (0 if forbidden).
    """
    if not selection_rules(kappa_i, kappa_f, L, 'M'):
        return 0.0

    l_i, tj_i = _kappa_to_lj(kappa_i)
    l_f, tj_f = _kappa_to_lj(kappa_f)

    tL = 2 * L
    threej = wigner_3j(tj_i, tL, tj_f, -1, 0, 1)

    if abs(threej) < 1e-20:
        return 0.0

    phase_exp = (tj_f + 1) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0

    # FAC's amplitude contains (2L+1)/sqrt(L(L+1))*j_L(kr)/k^L.
    # With our electric-rate normalization and radial (2L+1)!! j_L/k^L,
    # its remaining magnetic angular factor is (kappa_i+kappa_f)/(L+1).
    kappa_factor = (kappa_i + kappa_f) / (L + 1.0)

    return phase * np.sqrt((tj_i + 1.0) * (tj_f + 1.0)) * threej * kappa_factor


def allowed_final_kappas(kappa_i, L, multipole_type='E'):
    """Return list of allowed final-state kappa values.

    Generalizes the hard-coded E1 kappa list in continuum.py.

    Parameters
    ----------
    kappa_i : int
        Initial state kappa.
    L : int
        Multipole rank.
    multipole_type : str
        'E' or 'M'.

    Returns
    -------
    list of int
        Allowed final kappa values.
    """
    l_i, tj_i = _kappa_to_lj(kappa_i)

    results = []
    # Scan over possible l_f values: |l_i - L| to l_i + L
    for l_f in range(max(0, l_i - L), l_i + L + 1):
        # Check parity (EL: l_i + l_f + L even; ML: odd)
        parity_sum = l_i + l_f + L
        if multipole_type == 'E' and parity_sum % 2 != 0:
            continue
        if multipole_type == 'M' and parity_sum % 2 != 1:
            continue

        # For each l_f, there are up to two j_f values: l_f +/- 1/2
        # kappa for j = l + 1/2: kappa = -(l+1)
        # kappa for j = l - 1/2: kappa = l (only if l > 0)
        for kf in [-(l_f + 1), l_f]:
            if kf == 0:
                continue  # kappa=0 is not physical
            l_f_check, tj_f = _kappa_to_lj(kf)
            if l_f_check != l_f:
                continue

            # Triangle condition on j
            tL = 2 * L
            if tL < abs(tj_i - tj_f) or tL > tj_i + tj_f:
                continue

            if kf not in results:
                results.append(kf)

    return sorted(results, key=lambda k: (abs(k), k))


def line_strength_factor(kappa_i, kappa_f, L, multipole_type='E'):
    """Compute the full angular line-strength factor S.

    For a single-electron transition, the line strength is:
        S = |angular_factor|^2 * |radial_me|^2

    This function returns |angular_factor|^2.

    Parameters
    ----------
    kappa_i, kappa_f : int
    L : int
    multipole_type : str

    Returns
    -------
    float
    """
    if multipole_type == 'E':
        ang = angular_eL(kappa_i, kappa_f, L)
    elif multipole_type == 'M':
        ang = angular_mL(kappa_i, kappa_f, L)
    else:
        raise ValueError(f"Unknown multipole type: {multipole_type}")
    return ang ** 2
