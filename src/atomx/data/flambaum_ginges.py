"""Fit coefficients for the Flambaum–Ginges local radiative potential.

Citation
--------
V. V. Flambaum and J. S. M. Ginges, "The radiative potential method for
calculations of QED radiative corrections to energy levels and
electromagnetic amplitudes in many-electron atoms", Phys. Rev. A 72,
052115 (2005); arXiv:physics/0507067.

Equation references in the attribute docstrings follow the arXiv
preprint numbering.
"""

from __future__ import annotations

CITATION = "Flambaum & Ginges, Phys. Rev. A 72, 052115 (2005)."

# -- low-frequency SE (eq. 9) --------------------------------------------
#: Constant term of  B(Z) = B0 + B1 · Zα.
B_LOW_FREQ_CONST: float = 0.074
#: Linear-in-Zα term of the low-frequency coefficient.
B_LOW_FREQ_LINEAR: float = 0.35

# -- high-frequency electric form factor (eq. 10) ------------------------
#: Expansion coefficients of  A_scale(Z) = a0 + a2 x² + a3 x³ + a4 x⁴,
#: with  x = (Z − 80) α.
A_HIGH_FREQ_COEFFS: tuple[float, float, float, float] = (
    1.071,    # a0
   -1.976,    # a2
   -2.128,    # a3
    0.169,    # a4
)
#: Scale ``x0`` in the expansion variable x = (Z − x0) α.
A_HIGH_FREQ_Z_REF: int = 80
#: Short-range cutoff ``A(Z,r) = A_scale · r / (r + A_CUTOFF_COEFF · Z² α²)``.
A_CUTOFF_COEFF: float = 0.07

# -- simplified Wichmann–Kroll (eq. 12) ----------------------------------
#: Leading numerical coefficient  V_WK ∝ WK_LEADING · Z²α².
WK_LEADING: float = 0.092
#: Denominator scale  (1 + (WK_DENOM_SCALE · c · r)⁴).
WK_DENOM_SCALE: float = 1.62


TABLE: dict[str, object] = {
    "B_LOW_FREQ_CONST": B_LOW_FREQ_CONST,
    "B_LOW_FREQ_LINEAR": B_LOW_FREQ_LINEAR,
    "A_HIGH_FREQ_COEFFS": A_HIGH_FREQ_COEFFS,
    "A_HIGH_FREQ_Z_REF": A_HIGH_FREQ_Z_REF,
    "A_CUTOFF_COEFF": A_CUTOFF_COEFF,
    "WK_LEADING": WK_LEADING,
    "WK_DENOM_SCALE": WK_DENOM_SCALE,
}
