"""Semi-empirical Slater-parameter fitting (Cowan-style).

For open-shell atomic configurations, the ab-initio Slater integrals
``F^k``, ``G^k`` (direct and exchange Coulomb) and spin-orbit
``ζ_nl`` routinely over-estimate experimental level-energy splittings
by 10–30 %.  Cowan's classic workflow is to *scale* each parameter by
a multiplicative factor and let a least-squares fit determine the
scaling that best reproduces a list of measured levels.

This module implements the simplest variant of that workflow:

    E_i^{calc}  =  E_0  +  Σ_p  s_p · A_{i,p} · P_p^{ab-initio}       (1)

where

    * ``P_p^{ab-initio}``  are the ab-initio values of the parameters
      (F² of d-d, G² of p-d, ζ of d, …),
    * ``s_p``              are the scale factors to be fitted,
    * ``A_{i,p}``          is the level-parameter angular coefficient
      (ideally computed from an open-shell CI matrix; in Cowan's
      semi-empirical implementation the off-diagonal coefficients come
      from the diagonalisation of a model Hamiltonian).
    * ``E_0``              is an overall offset.

The unknowns are ``(s_p, E_0)``; the fit minimises

    χ²  =  Σ_i  [ (E_i^{exp} − E_i^{calc}) / σ_i ]²                   (2)

The implementation is a thin wrapper around
``scipy.optimize.least_squares``.  For production Cowan-style fits
users typically supply the angular matrix ``A`` from Cowan's RCE code
or an equivalent; atomx's fitter accepts ``A`` as an input array.

References
----------
R. D. Cowan, *The Theory of Atomic Structure and Spectra*
    (Univ. California Press, 1981).  Chapter on parameter fitting.
"""

from atomx.fitting.slater import (
    SlaterFitResult,
    fit_slater_parameters,
)

__all__ = [
    "SlaterFitResult",
    "fit_slater_parameters",
]
