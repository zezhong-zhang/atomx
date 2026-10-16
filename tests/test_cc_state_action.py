"""Matched complete finite CC/EA physics without a dense generator bank."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from atomx.methods import coupled_cluster, eom_attachment


def _anchor(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("_state_action_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_core_state_action_matches_independent_nonzero_t3_fci(monkeypatch):
    fixture = _anchor("test_finite_ccsdt.py")
    masks = fixture._masks(6, 3)
    reference = sum(1 << p for p in (0, 2, 4))
    h = fixture._hamiltonian(masks, *fixture._integrals(6))
    opts = dict(spinor_count=6, reference_identity="state-action:independent-core", method="CCSDT")
    dense = coupled_cluster.compute(h, masks, reference, **opts)

    def forbidden(*args):
        raise AssertionError("dense excitation generator allocation in state_action")

    monkeypatch.setattr(coupled_cluster, "_cluster_generators", forbidden)
    state = coupled_cluster.compute(h, masks, reference, **opts,
                                    action_representation="state_action")
    energies, vectors = np.linalg.eigh(h)
    assert state.total_energy_Ha == pytest.approx(energies[0], abs=1e-11)
    np.testing.assert_allclose(state.amplitudes, dense.amplitudes, atol=1e-11, rtol=0)
    np.testing.assert_allclose(np.outer(state.coefficients, state.coefficients),
                               np.outer(vectors[:, 0], vectors[:, 0]), atol=1e-11, rtol=0)
    assert abs(state.amplitudes[-1]) > 1e-5
    assert state.amplitude_residual_Ha < 1e-10
    assert state.full_eigenstate_residual_Ha < 1e-10
    assert state.provenance["generator_bytes"] == 0
    assert state.provenance["dense_generator_bytes_if_materialized"] == 60_800
    assert state.provenance["action_representation"] == "state_action"


def test_attachment_active_triples_and_omitted_rank_survive_state_action(monkeypatch):
    fixture = _anchor("test_eom_attachment.py")
    obj, _, _ = fixture.integrals(spinors=7)
    dense = eom_attachment.compute(obj)

    def forbidden(*args):
        raise AssertionError("dense excitation generator allocation in state_action")

    monkeypatch.setattr(coupled_cluster, "_cluster_generators", forbidden)
    monkeypatch.setattr(eom_attachment, "_cluster_generators", forbidden)
    state = eom_attachment.compute(obj, action_representation="state_action")
    np.testing.assert_allclose(state.projected_operator_Ha, dense.projected_operator_Ha,
                               atol=1e-11, rtol=0)
    np.testing.assert_allclose(state.attachment_energies_Ha, dense.attachment_energies_Ha,
                               atol=1e-11, rtol=0)
    for actual, expected in zip(state.coefficients.T, dense.coefficients.T, strict=True):
        np.testing.assert_allclose(np.outer(actual, actual.conj()),
                                   np.outer(expected, expected.conj()), atol=1e-10, rtol=0)
    assert state.provenance["projected_determinants"] == 34
    assert np.max(state.projected_residuals_Ha) < 1e-10
    assert np.max(state.full_eigenstate_residuals_Ha) > 1e-6
    np.testing.assert_allclose(state.full_eigenstate_residuals_Ha,
                               dense.full_eigenstate_residuals_Ha, atol=1e-10, rtol=0)
    assert state.provenance["generator_bytes"] == 0
    assert state.provenance["dense_generator_bytes_if_materialized"] == 666_400


def test_state_action_cap_rejects_before_phase_operator_or_dense_allocation(monkeypatch):
    from atomx.numerics import cluster_actions

    core_fixture = _anchor("test_finite_ccsdt.py")
    masks = core_fixture._masks(6, 3)
    reference = sum(1 << p for p in (0, 2, 4))
    h = core_fixture._hamiltonian(masks, *core_fixture._integrals(6))
    obj, _, _ = _anchor("test_eom_attachment.py").integrals(spinors=7)

    def forbidden(*args, **kwargs):
        raise AssertionError("operator entered before full action work admission")

    monkeypatch.setattr(cluster_actions, "apply_terms", forbidden)
    monkeypatch.setattr(eom_attachment, "_hamiltonian_terms", forbidden)
    with pytest.raises(ValueError, match="exponential term application cap"):
        coupled_cluster.compute(h, masks, reference, spinor_count=6,
                                reference_identity="pre-work-admission", method="CCSDT",
                                action_representation="state_action", cluster_term_application_cap=1000)
    with pytest.raises(ValueError, match="exponential term application cap"):
        eom_attachment.compute(obj, action_representation="state_action",
                               cluster_term_application_cap=3000)
    with pytest.raises(ValueError, match="representation"):
        eom_attachment.compute(obj, action_representation="unknown")


def test_no_virtuals_state_action_retains_complete_one_state_result():
    result = coupled_cluster.compute([[-.5]], (3,), 3, spinor_count=2,
                                     reference_identity="filled-sector", action_representation="state_action")
    assert result.total_energy_Ha == -.5
    assert len(result.amplitudes) == 0
    assert result.provenance["generator_bytes"] == 0
