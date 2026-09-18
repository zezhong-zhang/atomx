"""Angular momentum algebra for both LS and jj coupling schemes.

Core coefficients (Racah formula with log-factorial caching):
    wigner_3j, wigner_6j, wigner_9j, clebsch_gordan

LS-coupling (non-relativistic):
    term_symbol, ls_expectation, allowed_terms, slater_condon_f

Recoupling (LS ↔ jj transformations):
    ls_jj_transformation, kappa_from_lj, lj_from_kappa
"""

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.csf_recoupling import (
    ChainCoupledCSFState,
    RecoupledCSFState,
    TwoBodyCSFCoefficient,
    chain_one_body_excitation_reduced,
    chain_projection_amplitudes,
    chain_state_from_simple_csf_auto,
    chain_two_body_scalar_coefficients,
    chain_w_reduced,
    chain_wxw_scalar,
    discover_chain_two_body_scalar_coefficients,
    make_chain_coupled_state,
)
from atomx.angular.ls_coupling import (
    allowed_terms,
    ls_expectation,
    slater_condon_f,
    term_symbol,
)
from atomx.angular.recoupling import (
    kappa_from_lj,
    lj_from_kappa,
    ls_jj_transformation,
)
from atomx.angular.wigner import wigner_3j, wigner_6j, wigner_9j
