"""Tests for shared CSF angular utilities."""

import numpy as np
import pytest

from atomx.angular.csf import (
    coulomb_pair_k_values,
    csf_twice_j,
    fac_uta_population_factor,
    multipole_allows_state_transition,
    one_body_csf_coupling,
    orbital_tensor_coupling_coeff,
    two_body_csf_coupling,
)
from atomx.config.csf import CSF
from atomx.methods.dirac_fock import Subshell


def test_explicit_j_zero_is_preserved():
    csf = CSF([], total_J2=0, parity=1, label="closed J=0")
    assert csf_twice_j(csf) == 0


def test_many_electron_multipole_selection_rules():
    assert multipole_allows_state_transition(0, 1, 2, -1, 1, "E")
    assert not multipole_allows_state_transition(0, 1, 2, 1, 1, "E")
    assert multipole_allows_state_transition(2, 1, 2, 1, 1, "M")
    assert not multipole_allows_state_transition(0, 1, 0, 1, 1, "M")


def test_fac_uta_population_factor():
    assert fac_uta_population_factor(4, 0, -2, -3) == pytest.approx(1.0)
    assert fac_uta_population_factor(2, 1, -2, -3) == pytest.approx(
        np.sqrt(10.0 / 24.0)
    )
    assert fac_uta_population_factor(2, 6, -2, -3) == 0.0


def test_one_body_csf_coupling_extracts_source_destination_and_population():
    lower = CSF([
        Subshell(n=2, kappa=-2, occupation=2),
        Subshell(n=3, kappa=-3, occupation=1),
    ], total_J2=2, parity=1, label="lower J=1")
    upper = CSF([
        Subshell(n=2, kappa=-2, occupation=1),
        Subshell(n=3, kappa=-3, occupation=2),
    ], total_J2=4, parity=-1, label="upper J=2")

    coupling = one_body_csf_coupling(upper, lower, L=1, operator_type="E")

    assert coupling is not None
    assert coupling.source == (2, -2)
    assert coupling.destination == (3, -3)
    assert coupling.source_occupation == 2
    assert coupling.destination_occupation == 1
    assert coupling.population_factor == pytest.approx(np.sqrt(10.0 / 24.0))


def test_one_body_csf_coupling_blocks_forbidden_explicit_j():
    lower = CSF([
        Subshell(n=2, kappa=-2, occupation=1),
        Subshell(n=3, kappa=-3, occupation=1),
    ], total_J2=0, parity=1, label="lower J=0")
    upper = CSF([
        Subshell(n=2, kappa=-2, occupation=0),
        Subshell(n=3, kappa=-3, occupation=2),
    ], total_J2=0, parity=-1, label="upper J=0")

    assert one_body_csf_coupling(upper, lower, L=1, operator_type="E") is None


def test_coulomb_pair_k_values_intersects_one_electron_tensor_ranges():
    # s1/2 -> p1/2 and s1/2 -> p1/2 require odd Coulomb rank k=1.
    assert coulomb_pair_k_values((1, -1), (1, -1), (2, 1), (2, 1)) == (1,)
    # s1/2 -> s1/2 and p1/2 -> p1/2 allow scalar and vector ranks common to both.
    assert coulomb_pair_k_values((1, -1), (2, 1), (3, -1), (3, 1)) == (0,)


def test_orbital_tensor_coupling_coeff_is_rank_filtered_and_normalized():
    coeff = orbital_tensor_coupling_coeff((1, -1), (2, 1), (1, -1), (2, 1), 1)
    assert coeff > 0.0
    assert coeff <= 1.0
    assert orbital_tensor_coupling_coeff((1, -1), (2, 1), (1, -1), (2, 1), 0) == 0.0


def test_two_body_csf_coupling_detects_direct_and_exchange_ranks():
    ket = CSF([
        Subshell(n=1, kappa=-1, occupation=2),
    ], total_J2=0, parity=1, label="1s2 J=0")
    bra = CSF([
        Subshell(n=2, kappa=1, occupation=2),
    ], total_J2=0, parity=1, label="2p-2 J=0")

    coupling = two_body_csf_coupling(bra, ket)

    assert coupling is not None
    assert coupling.sources == ((1, -1), (1, -1))
    assert coupling.destinations == ((2, 1), (2, 1))
    assert coupling.direct_k == (1,)
    assert coupling.exchange_k == (1,)


def test_two_body_csf_coupling_blocks_scalar_jpi_mismatch():
    ket = CSF([
        Subshell(n=1, kappa=-1, occupation=2),
    ], total_J2=0, parity=1, label="1s2 J=0")
    bra = CSF([
        Subshell(n=2, kappa=1, occupation=2),
    ], total_J2=2, parity=1, label="2p-2 J=1")

    assert two_body_csf_coupling(bra, ket) is None
