"""Angular momentum algebra for both LS and jj coupling schemes.

Core coefficients (Racah formula with log-factorial caching):
    wigner_3j, wigner_6j, wigner_9j, clebsch_gordan

LS-coupling (non-relativistic):
    term_symbol, ls_expectation, allowed_terms, slater_condon_f

Recoupling (LS ↔ jj transformations):
    ls_jj_transformation, kappa_from_lj, lj_from_kappa
"""

from atomx.angular.wigner import wigner_3j, wigner_6j, wigner_9j
from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.ls_coupling import (
    term_symbol, ls_expectation, allowed_terms, slater_condon_f,
)
from atomx.angular.recoupling import (
    ls_jj_transformation, kappa_from_lj, lj_from_kappa,
)
