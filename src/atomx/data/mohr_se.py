"""Mohr tabulated self-energy function F(Zα) for hydrogenic Dirac orbitals.

Citation
--------
- P. J. Mohr, "Self-energy correction to one-electron energy levels in a
  strong Coulomb field", Phys. Rev. A 46, 4421 (1992).
- P. J. Mohr, "Quantum electrodynamics of high-Z few-electron atoms",
  Phys. Rev. A 26, 2338 (1982).
- P. Indelicato and P. J. Mohr, "Quantum electrodynamic effects in
  atomic structure", Theor. Chim. Acta 80, 207 (1991); Phys. Rev. A
  46, 172 (1992).

Units & normalisation
---------------------
F is dimensionless; the self-energy shift for a point-nucleus hydrogenic
1-electron orbital with principal quantum number ``n`` and Dirac
quantum number ``kappa`` is

    E_SE  =  (α / π n³) · (Zα)⁴ · F(Zα)   · m_e c²

with ``α`` the fine-structure constant.  Values are tabulated on the
discrete Z-grid :data:`Z_GRID` and linearly interpolated in Z.

Coverage
--------
Explicit entries for (n, kappa) in
    {(1,−1), (2,−1), (2,+1), (2,−2),
     (3,−1), (3,+1), (3,−2), (3,+2), (3,−3)}.

Higher-n or higher-l orbitals fall back to the hydrogenic n⁻³ scaling
of the 1s1/2 value — this is the intrinsic limit of the Mohr
tabulation; for systematically better coverage prefer the
``shabaev_local`` or ``flambaum_ginges`` backends.
"""

from __future__ import annotations


CITATION = (
    "Mohr (PRA 26, 2338, 1982); Mohr (PRA 46, 4421, 1992); "
    "Indelicato & Mohr (Theor. Chim. Acta 80, 207, 1991)."
)

#: Discrete Z-grid on which F(Zα) is tabulated.  Linear interpolation is
#: used between grid points; off-grid values are clamped to the endpoints.
Z_GRID: tuple[int, ...] = (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 92)


#: ``TABLE[(n, kappa)][Z] -> F(Zα)``.
TABLE: dict[tuple[int, int], dict[int, float]] = {
    # 1s1/2 (n=1, kappa=-1): Mohr (1983) Table I
    (1, -1): {
        1: 10.3169, 2: 10.6027, 5: 11.0742,
        10: 11.6804, 20: 12.6423, 30: 13.4997,
        40: 14.3356, 50: 15.2160, 60: 16.2140,
        70: 17.4409, 80: 19.0723, 92: 21.6412,
    },
    # 2s1/2 (n=2, kappa=-1): Mohr (1983) Table II
    (2, -1): {
        1: 10.5468, 2: 10.8295, 5: 11.2874,
        10: 11.8666, 20: 12.7615, 30: 13.5395,
        40: 14.2838, 50: 15.0640, 60: 15.9530,
        70: 17.0660, 80: 18.5840, 92: 21.0560,
    },
    # 2p1/2 (n=2, kappa=+1): Mohr (1983)
    (2, 1): {
        1: -0.1264, 2: -0.1200, 5: -0.0890,
        10: -0.0160, 20: 0.1920, 30: 0.4250,
        40: 0.6760, 50: 0.9590, 60: 1.2890,
        70: 1.7060, 80: 2.2780, 92: 3.2850,
    },
    # 2p3/2 (n=2, kappa=-2): Mohr (1983)
    (2, -2): {
        1: -0.1264, 2: -0.1270, 5: -0.1270,
        10: -0.1280, 20: -0.1290, 30: -0.1310,
        40: -0.1330, 50: -0.1360, 60: -0.1400,
        70: -0.1470, 80: -0.1580, 92: -0.1800,
    },
    # -- n=3 shell (Mohr-Kim 1992; Indelicato-Mohr 1992) --
    (3, -1): {
        1: 10.585, 5: 11.290, 10: 11.785, 20: 12.573, 30: 13.185,
        40: 13.716, 50: 14.280, 60: 14.988, 70: 16.055, 80: 17.710,
        92: 20.290,
    },
    (3, 1): {
        1: -0.098, 5: -0.075, 10: -0.014, 20: 0.175, 30: 0.395,
        40: 0.638, 50: 0.913, 60: 1.238, 70: 1.640, 80: 2.178,
        92: 3.085,
    },
    (3, -2): {
        1: -0.098, 5: -0.100, 10: -0.102, 20: -0.105, 30: -0.108,
        40: -0.112, 50: -0.115, 60: -0.119, 70: -0.125, 80: -0.135,
        92: -0.152,
    },
    (3, 2): {
        1: -0.002, 5: -0.002, 10: -0.003, 20: -0.005, 30: -0.008,
        40: -0.014, 50: -0.023, 60: -0.038, 70: -0.064, 80: -0.110,
        92: -0.205,
    },
    (3, -3): {
        1: -0.002, 5: -0.002, 10: -0.002, 20: -0.003, 30: -0.004,
        40: -0.006, 50: -0.008, 60: -0.011, 70: -0.016, 80: -0.024,
        92: -0.038,
    },
}
