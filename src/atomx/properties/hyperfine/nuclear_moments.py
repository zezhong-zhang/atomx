"""Tabulated nuclear spins and magnetic moments.

Values taken from:
    N. J. Stone, "Table of nuclear magnetic dipole and electric quadrupole
    moments", IAEA INDC(NDS)-0658 (2014) and IAEA INDC(NDS)-0794 (2019).
    NIST Compilation of Atomic Spectra Database.

Conventions
-----------
* Nuclear spin ``I`` is a rational number (integer or half-integer).
* Magnetic moment ``mu_I`` is in units of the nuclear magneton μ_N,
  following the convention μ_I = ⟨I, m=I | μ_z | I, m=I⟩ (the maximally
  stretched state), so the nuclear g-factor is ``g_I = mu_I / I``.
* Electric quadrupole moment ``Q`` is in barns (10⁻²⁴ cm²) and refers to
  the spectroscopic (intrinsic-in-the-body-frame) quadrupole moment.

Only a representative set of frequently-measured isotopes is shipped
initially; users can extend via :data:`NUCLEAR_MOMENTS` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class NuclearMoment:
    """Static nuclear properties for a specific isotope.

    Attributes
    ----------
    Z : int
        Nuclear charge (number of protons).
    A : int
        Mass number.
    I : float
        Nuclear spin (integer or half-integer).
    mu_I : float
        Magnetic dipole moment in units of the nuclear magneton μ_N.
    Q : float
        Electric quadrupole moment in barns (10⁻²⁴ cm²).  Zero for I ≤ 1/2.
    name : str
        Human-readable label, e.g. ``"Cs-133"``.
    mass_u : float
        Atomic mass in unified atomic-mass units (1 u = 1/12 ¹²C mass).
        When zero (default) :meth:`mass_me` falls back to
        ``A × m_p/m_e``, adequate for isotope-shift estimates to ~0.1 %.
    r_rms_fm : float
        Nuclear charge RMS radius in femtometres (Angeli–Marinova 2013
        tabulation for light isotopes; empirical ``0.836 A^(1/3) + 0.570``
        for heavier).  When zero, callers should use
        :func:`atomx.potentials.nuclear._get_rrms_fm` as a fallback.
    """

    Z: int
    A: int
    I: float
    mu_I: float
    Q: float
    name: str
    mass_u: float = 0.0
    r_rms_fm: float = 0.0

    @property
    def g_I(self) -> float:
        """Nuclear g-factor g_I = μ_I / I (nuclear-magneton units)."""
        if self.I == 0:
            return 0.0
        return self.mu_I / self.I

    @property
    def mass_me(self) -> float:
        """Nuclear (not atomic) mass in electron-mass units.

        If ``mass_u`` is set, uses the atomic mass minus Z electron masses
        (nuclear mass = atomic mass − Z × m_e in energy-equivalent terms,
        but in u × PROTON_ELECTRON_MASS_RATIO for our purposes, with the
        Z × 1 electron-mass subtraction small).  Otherwise the simple
        ``A × m_p/m_e`` proxy is used.
        """
        from atomx.constants import PROTON_ELECTRON_MASS_RATIO
        if self.mass_u > 0:
            # 1 u ≈ 1822.888 m_e.  Subtract Z electrons.
            U_IN_ME = 1822.888486209
            return self.mass_u * U_IN_ME - self.Z
        return self.A * PROTON_ELECTRON_MASS_RATIO


# ---------------------------------------------------------------------------
# Representative isotope table
# ---------------------------------------------------------------------------
# Key: (Z, A).  Values drawn from Stone 2014/2019 and NIST.

NUCLEAR_MOMENTS: Dict[Tuple[int, int], NuclearMoment] = {
    # (Z, A): NuclearMoment
    # Precision isotopes (with atomic mass) for isotope-shift work
    (1, 1):   NuclearMoment(Z=1,  A=1,   I=0.5, mu_I=+2.7928474,  Q=0.0,
                            mass_u=1.00782503207, r_rms_fm=0.8783, name="H-1"),
    (1, 2):   NuclearMoment(Z=1,  A=2,   I=1.0, mu_I=+0.857438,   Q=+0.0028,
                            mass_u=2.01410177812, r_rms_fm=2.1421, name="H-2 (D)"),
    (2, 3):   NuclearMoment(Z=2,  A=3,   I=0.5, mu_I=-2.127625,   Q=0.0,
                            mass_u=3.01602932197, r_rms_fm=1.9661, name="He-3"),
    (3, 6):   NuclearMoment(Z=3,  A=6,   I=1.0, mu_I=+0.822047,   Q=-0.000806, name="Li-6"),
    (3, 7):   NuclearMoment(Z=3,  A=7,   I=1.5, mu_I=+3.256427,   Q=-0.0406,name="Li-7"),
    (4, 9):   NuclearMoment(Z=4,  A=9,   I=1.5, mu_I=-1.17749,    Q=+0.0529,name="Be-9"),
    (5, 11):  NuclearMoment(Z=5,  A=11,  I=1.5, mu_I=+2.688649,   Q=+0.0407,name="B-11"),
    (6, 13):  NuclearMoment(Z=6,  A=13,  I=0.5, mu_I=+0.702412,   Q=0.0,    name="C-13"),
    (7, 14):  NuclearMoment(Z=7,  A=14,  I=1.0, mu_I=+0.4037610,  Q=+0.0200,name="N-14"),
    (8, 17):  NuclearMoment(Z=8,  A=17,  I=2.5, mu_I=-1.89380,    Q=-0.0255,name="O-17"),
    (9, 19):  NuclearMoment(Z=9,  A=19,  I=0.5, mu_I=+2.628868,   Q=0.0,    name="F-19"),
    (11, 23): NuclearMoment(Z=11, A=23,  I=1.5, mu_I=+2.217656,   Q=+0.1006,name="Na-23"),
    (13, 27): NuclearMoment(Z=13, A=27,  I=2.5, mu_I=+3.641507,   Q=+0.1466,name="Al-27"),
    (15, 31): NuclearMoment(Z=15, A=31,  I=0.5, mu_I=+1.13160,    Q=0.0,    name="P-31"),
    (19, 39): NuclearMoment(Z=19, A=39,  I=1.5, mu_I=+0.3914662,  Q=+0.0585,name="K-39"),
    (29, 63): NuclearMoment(Z=29, A=63,  I=1.5, mu_I=+2.2273456,  Q=-0.220, name="Cu-63"),
    (37, 85): NuclearMoment(Z=37, A=85,  I=2.5, mu_I=+1.3533515,  Q=+0.2766,name="Rb-85"),
    (37, 87): NuclearMoment(Z=37, A=87,  I=1.5, mu_I=+2.7512932,  Q=+0.1335,name="Rb-87"),
    (47, 107):NuclearMoment(Z=47, A=107, I=0.5, mu_I=-0.11368,    Q=0.0,    name="Ag-107"),
    (55, 133):NuclearMoment(Z=55, A=133, I=3.5, mu_I=+2.5828327,  Q=-0.00355,name="Cs-133"),
    (79, 197):NuclearMoment(Z=79, A=197, I=1.5, mu_I=+0.148158,   Q=+0.547, name="Au-197"),
    (80, 199):NuclearMoment(Z=80, A=199, I=0.5, mu_I=+0.5058855,  Q=0.0,    name="Hg-199"),
    (81, 205):NuclearMoment(Z=81, A=205, I=0.5, mu_I=+1.638214,   Q=0.0,    name="Tl-205"),
    (82, 207):NuclearMoment(Z=82, A=207, I=0.5, mu_I=+0.592583,   Q=0.0,    name="Pb-207"),
}


# ---------------------------------------------------------------------------
# Default-isotope convenience map (most abundant stable isotope per element)
# ---------------------------------------------------------------------------

_DEFAULT_A: Dict[int, int] = {
    1: 1, 2: 3, 3: 7, 4: 9, 5: 11, 6: 13, 7: 14, 8: 17, 9: 19,
    11: 23, 13: 27, 15: 31, 19: 39, 29: 63, 37: 87, 47: 107,
    55: 133, 79: 197, 80: 199, 81: 205, 82: 207,
}


def default_isotope(Z: int) -> int:
    """Return the default mass number A for element Z (most common with I≠0)."""
    if Z not in _DEFAULT_A:
        raise KeyError(
            f"No default isotope tabulated for Z={Z}. "
            f"Known: {sorted(_DEFAULT_A.keys())}"
        )
    return _DEFAULT_A[Z]


def nuclear_moment(Z: int, A: int | None = None) -> NuclearMoment:
    """Look up tabulated nuclear properties for isotope (Z, A).

    If ``A`` is ``None``, :func:`default_isotope` picks the standard choice.
    """
    if A is None:
        A = default_isotope(Z)
    key = (Z, A)
    if key not in NUCLEAR_MOMENTS:
        raise KeyError(
            f"No tabulated moments for (Z={Z}, A={A}). "
            f"Available: {sorted(NUCLEAR_MOMENTS.keys())}"
        )
    return NUCLEAR_MOMENTS[key]


def g_factor(Z: int, A: int | None = None) -> float:
    """Nuclear g-factor g_I = μ_I/I for isotope (Z, A)."""
    return nuclear_moment(Z, A).g_I
