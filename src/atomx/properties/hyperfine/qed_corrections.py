"""QED corrections to the hyperfine operator itself.

Beyond the overall ``(1 + a_e)`` Schwinger multiplicative factor, the
hyperfine photon line and vertex receive radiative dressings from QED.
Below a sketch of the leading contributions and their status in atomx.

Status summary
--------------
* Uehling-vertex (vacuum polarization of the hyperfine photon):
  NOT YET IMPLEMENTED.  Absorbed empirically into the
  ``qed_hf_factor`` coefficient of the Karshenboim presets; see
  :data:`atomx.methods.dirac_fock._HYPERFINE_PRESETS`.
* Wichmann–Kroll vertex:  NOT YET IMPLEMENTED (sub-ppm for H 1s).
* Electron self-energy correction to the HF operator:  NOT YET
  IMPLEMENTED; similarly absorbed.
* Bethe–Gerhards logarithm:  NOT YET IMPLEMENTED; requires the
  Coulomb Green's function.
* Higher-order electron-vertex α²(Zα)², α³(Zα):  covered by the
  Karshenboim-tuned ``qed_hf_factor``.

Why the "naïve" local matrix element fails
------------------------------------------
One might be tempted to approximate the Uehling-vertex correction as

    δ_vertex  ≈  ⟨ψ | V_U(r) · T_HF(r) | ψ⟩ / ⟨ψ | T_HF(r) | ψ⟩       (✗)

but this **does not** correspond to any physical QED diagram: the
Uehling potential V_U(r) diverges near the origin faster than the bare
``1/r²`` hyperfine kernel, so the local ratio is dominated by a small
spatial region without physical significance.  The correct Uehling-
vertex correction treats the vacuum polarization as a modification of
the hyperfine **photon propagator**, equivalent to a convolution

    T_HF^{VP}(r)  =  ∫ dr' K_U(|r − r'|) · T_HF(r')                   (2)

with K_U the Uehling kernel in momentum space.  That integral can be
evaluated in closed form for hydrogenic states (Mickelwait 1954,
Drell-Dalgarno 1957) but is expensive on a general B-spline basis;
its magnitude for H 1s is −8 ppm.

To reach 1 ppm for heavy systems we need to evaluate eq. (2) properly;
an MV implementation of the full integral is a TODO flagged here.

Precision constants (usable today)
----------------------------------

:data:`PROTON_ZEMACH_RADIUS_FM`
    CODATA 2022 Zemach radius  r_Z(p) = 1.054 fm.

:data:`DEUTERON_ZEMACH_RADIUS_FM`
    2.593 fm, Friar-Payne-Prelin 2005.

:data:`HELIUM3_ZEMACH_RADIUS_FM`
    2.528 fm, Nesvizhevsky 1992 / Friar 2013 compilations.

:data:`PROTON_POLARIZABILITY_H1S`
    Carlson-Vanderhaeghen 2011 polarizability contribution to the H 1s
    hyperfine splitting,  δ_pol = +1.3(3) × 10⁻⁶.

:data:`DEUTERON_POLARIZABILITY_D1S`, :data:`HELIUM3_POLARIZABILITY`
    Analogous values for deuterium and ³He.

References
----------
S. G. Karshenboim, *Phys. Rep.* **422**, 1 (2005).
A. N. Artemyev, V. M. Shabaev, V. A. Yerokhin, G. Plunien, G. Soff,
    *Phys. Rev. A* **71**, 062104 (2005).
C. E. Carlson, V. Nazaryan, K. Griffioen, *Phys. Rev. A* **83**,
    042509 (2011).  Proton polarizability contribution.
J. Friar, *Phys. Rev. C* **88**, 034003 (2013).  ³He Zemach moment.
CODATA 2022 recommended values.
"""

from __future__ import annotations

# Precision constants live in :mod:`atomx.data.nuclear_structure` with
# full citations.  Re-export them here for backwards compatibility with
# code that imports ``atomx.properties.hyperfine.qed_corrections.PROTON_ZEMACH_RADIUS_FM``.
from atomx.data.nuclear_structure import TABLE as _NUC


PROTON_ZEMACH_RADIUS_FM = _NUC["h1s"].zemach_radius_fm
DEUTERON_ZEMACH_RADIUS_FM = _NUC["d1s"].zemach_radius_fm
HELIUM3_ZEMACH_RADIUS_FM = _NUC["he3_1s"].zemach_radius_fm

PROTON_POLARIZABILITY_H1S = _NUC["h1s"].polarisability_fraction
DEUTERON_POLARIZABILITY_D1S = _NUC["d1s"].polarisability_fraction
HELIUM3_POLARIZABILITY = _NUC["he3_1s"].polarisability_fraction


# ---------------------------------------------------------------------------
# Zemach correction (closed form deferred)
# ---------------------------------------------------------------------------
#
# A ``zemach_fractional_correction(Z, r_Zemach_fm)`` function was
# attempted here but failed quantitative validation: the simple
# ``-2·Z·α·r_Zemach(Bohr)`` expression returns ~-0.3 ppm for H 1s,
# whereas the published Karshenboim value is -41 ppm — off by ~1/α.
# The correct formula involves integral moments of the charge and
# magnetization distributions (Zemach 1956 eq. 6, Karshenboim 2005
# §IV.F), with a specific combination of Bohr radius, reduced mass,
# and nuclear form factors that does not reduce to a single
# ``r_Zemach_fm``-scaled expression.
#
# Implementing this properly (targeting the sub-ppm precision milestone)
# requires:
#   * proton (and magnetization-weighted) charge-density form factors,
#   * numerical evaluation of the two-dimensional moment integral,
#   * separate treatment of the elastic and inelastic (polarizability)
#     proton-structure contributions per Carlson et al. 2011.
#
# For now, users should apply the published per-isotope δ_Zemach
# coefficients directly through the ``nuclear_structure_factor`` rung
# of :meth:`atomx.methods.dirac_fock.SCFResult.hyperfine_constants`, using the
# tabulated constants above as a reference.
