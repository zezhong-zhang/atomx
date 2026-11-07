"""Embedding E1 (atomic limit) — L0 anchors (design docs/QUANTUM_EMBEDDING_DESIGN.md).

- occupation sum rule: sum_a n_a = N exactly (thermal mixture of the N sector)
- spectral-weight sum rule: each spin-orbital carries unit weight
- multiplet poles: the d8 excitation ladder matches the validated Racah
  structure (single-ion 1D/1G/1S families with SOC and baricenter monopole)
- payload: impurity-greens@1 with mandatory provenance
"""
import numpy as np
import pytest

from atomx.embedding import (
    ImpuritySpec,
    impurity_greens_payload,
    occupation,
    solve_atomic_limit,
    spectral_function,
)

# CPC-2019 screened set (same numbers the hyperx anchors #19/#23 pin);
# baricenter dd monopole convention.
SPEC = ImpuritySpec(
    n_electrons=8,
    slater_eV={"F0": 0.409231746031746, "F2": 7.9521, "F4": 4.9387},
    zeta_d_eV=0.083,
)


@pytest.fixture(scope="module")
def sol():
    return solve_atomic_limit(SPEC)


def test_occupation_sum_rule_exact(sol):
    n = occupation(sol)
    assert float(n.sum()) == pytest.approx(8.0, abs=1e-12)
    assert (n > 0).all() and (n < 1).all()


def test_spectral_weight_unit_per_orbital(sol):
    omega = np.linspace(-15, 5, 1200)
    A, _ = spectral_function(sol, omega, eta=0.1)
    weights = A.sum(axis=1) * (omega[1] - omega[0])
    assert np.allclose(weights, 1.0, atol=0.02)


def test_multiplet_poles_match_racah_ladder(sol):
    """Full-SOC ladder (spin-flip terms included — the diagonal-only SOC
    that once slipped through produced a different, wrong fine structure;
    corrected windows pinned after the fix)."""
    ev = sol["N=8"]["evals"]
    ex = np.unique(np.round(ev - ev[0], 4))
    assert ((ex > 0.0) & (ex < 0.3)).sum() >= 2   # 3F_J ground fine structure
    assert ((ex > 1.3) & (ex < 1.9)).sum() >= 3   # 1D family
    assert ((ex > 2.0) & (ex < 2.4)).sum() >= 1   # 1G family
    assert ((ex > 5.0) & (ex < 5.5)).sum() >= 1   # 1S


def test_green_poles_equal_sector_level_differences(sol):
    """The removal edge sits at E_0^(N-1) - E_0^(N) (Hubbard-I identity)."""
    eN = sol["N=8"]["evals"]
    em = sol["N=7"]["evals"]
    omega = np.linspace(-20, 5, 4001)
    A, _ = spectral_function(sol, omega, eta=0.05)
    tot = A.sum(axis=0)
    # the REMOVAL EDGE: strongest pole in the near-edge window (deep d7
    # multiplet tails extend to ~-10 eV and are not the edge)
    expected = float(em[0] - eN[0])
    mask = (omega > expected - 1.5) & (omega < expected + 0.5)
    peak = omega[mask][np.argmax(tot[mask])]
    assert abs(peak - expected) < 0.05


def test_payload_schema_and_provenance(sol):
    omega = np.linspace(-15, 5, 200)
    A, G = spectral_function(sol, omega)
    payload = impurity_greens_payload(
        species="Ni", n_electrons=8, omega_eV=omega, G=G,
        occupation=occupation(sol), temperature_K=SPEC.temperature_K,
        backend="atomx-embedding-reference", eta_eV=0.1,
    )
    assert payload["schema"] == "impurity-greens@1"
    prov = payload["provenance"]
    assert prov["model"] == "atomx-embedding-reference"
    assert prov["level"] == "atomic-limit(Hubbard-I)"
    assert len(payload["G_orb"]) == 10
