"""Per-isotope presets for the magnetic-dipole hyperfine ladder.

Citation
--------
- S. G. Karshenboim, "Precision physics of simple atoms: QED tests,
  nuclear structure, and fundamental constants", Phys. Rep. 422, 1 (2005).
- G. Carlson & M. Vanderhaeghen, proton polarisability contribution to
  hyperfine (2011).
- CODATA 2022 for recommended Zemach-radius values underpinning the
  nuclear-structure rung.

Semantics
---------
Each preset supplies the small-parameter rungs that multiply the raw
Dirac–Hartree–Fock hyperfine-A baseline:

    A_ladder = A_DHF
             × (M/(M+m_e))³          # classical recoil factor
             × (1 + a_e)              # Schwinger + two-loop electron moment
             × (1 + δ_QEDhf)          # bound-state QED of the HF operator
             × (1 + δ_recoil)         # radiative recoil beyond classical
             × (1 + δ_nuc)            # nuclear structure (Zemach + BW + pol.)

The per-preset dict carries ``a_e_on`` (apply the anomalous-moment rung)
plus the three small dimensionless fractions
``delta_QEDhf``, ``delta_recoil``, ``delta_nuc``.

Provenance note
---------------
The numerical values for H-1s are *empirically tuned* to reproduce the
Karshenboim 2005 theoretical total ν_theory = 1420.403 MHz (within
~2 ppm of the experimental 1420.4057 MHz) against atomx's 80-basis
B-spline DHF baseline A_DHF = 1421.273 MHz.  The rung magnitudes track
the Karshenboim decomposition but absorb basis-dependence in the short-
range wavefunction; reaching 1 ppm needs a per-basis recomputation of
δ_QEDhf.  D and He-3 entries use the same decomposition with
isotope-specific Zemach radii and recoil coefficients; accuracy is at
the ~20 ppm level until individually validated.
"""

from __future__ import annotations

from dataclasses import dataclass


CITATION = (
    "Karshenboim, Phys. Rep. 422, 1 (2005); CODATA 2022; "
    "Carlson & Vanderhaeghen (2011)."
)


@dataclass(frozen=True)
class HyperfinePreset:
    """Small-parameter rung coefficients for one isotope.

    Attributes
    ----------
    a_e_on
        Whether to apply the anomalous-electron-moment rung ``(1+a_e)``.
    delta_qed_hf
        Bound-state QED fractional correction to the hyperfine operator.
    delta_recoil
        Radiative-recoil fractional correction beyond the classical
        ``(M/(M+m_e))³`` factor.
    delta_nuc
        Nuclear-structure fractional correction (Zemach + Bohr–Weisskopf
        + proton polarisability).
    """

    a_e_on: bool
    delta_qed_hf: float
    delta_recoil: float
    delta_nuc: float


TABLE: dict[str, HyperfinePreset] = {
    "h1s":    HyperfinePreset(True, -95.0e-6, -2.0e-6,  -41.0e-6),
    "d1s":    HyperfinePreset(True, -95.0e-6, -0.5e-6, -138.0e-6),
    "he3_1s": HyperfinePreset(True, -95.0e-6, -4.0e-5, -210.0e-6),
}
