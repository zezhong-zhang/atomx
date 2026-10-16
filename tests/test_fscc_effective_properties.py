"""Independent normalized signed properties and strict wave admission gates."""

import numpy as np
import pytest

from atomx.methods.ci_effective import FiniteCIModel, FiniteCITransitionOperator
from atomx.properties.transitions.fscc_effective import FiniteCorrelatedWave


def model(h, p, name):
    return FiniteCIModel(
        h,
        p,
        reference_identity=name,
        csf_identities=tuple(range(len(h))),
        radial_binding=(name,),
    )


def test_rectangular_signed_two_endpoint_metrics_against_full_ci():
    hb = np.array([[-0.7, 0.1, 0.3], [0.1, -0.4, -0.2], [0.3, -0.2, 1.0]])
    hk = np.array([[-0.6, 0.3], [0.3, 1.2]])
    b, k = model(hb, (0, 1), "bra"), model(hk, (0,), "ket")
    bc, kc = b.bloch_reduction((0, 1)), k.bloch_reduction((0,))
    bw = FiniteCorrelatedWave(
        b, bc.wave_operator, bc.directed_Ha, source_identity="actual bra"
    )
    kw = FiniteCorrelatedWave(
        k, kc.wave_operator, kc.directed_Ha, source_identity="actual ket"
    )
    full = np.array([[0.2, 0.4j], [-0.3j, 0.6], [0.8, -0.2]])
    op = FiniteCITransitionOperator(
        full, b, k, operator_identity="complex T+", unit="au"
    )
    expected = bc.orthonormal_wave_operator.T @ full @ kc.orthonormal_wave_operator
    np.testing.assert_allclose(
        bw.effective_transition(op, kw, representation="orthonormal"),
        expected,
        atol=1e-14,
    )
    eb, vk = np.linalg.eigh(hb), np.linalg.eigh(hk)[1][:, :1]
    vb = eb[1][:, :2]
    # Independent eigenvectors supply all signed amplitudes, not diagonal strengths.
    coordinates_b = bw.orthonormal_wave_operator.conj().T @ vb
    coordinates_k = kw.orthonormal_wave_operator.conj().T @ vk
    np.testing.assert_allclose(
        coordinates_b.conj().T @ expected @ coordinates_k,
        vb.conj().T @ full @ vk,
        atol=1e-14,
    )
    bloch = bw.effective_transition(op, kw, representation="bloch")
    reverse = FiniteCITransitionOperator(
        full.conj().T, k, b, operator_identity="T-", unit="au"
    )
    rb = kw.effective_transition(reverse, bw, representation="bloch")
    np.testing.assert_allclose(kw.metric @ rb, bloch.conj().T @ bw.metric, atol=1e-14)
    weak = FiniteCITransitionOperator(
        1e-18 * full, b, k, operator_identity="weak", unit="au"
    )
    np.testing.assert_allclose(
        bw.effective_transition(weak, kw, representation="orthonormal"),
        1e-18 * expected,
        atol=0,
        rtol=1e-13,
    )
    assert np.max(abs(expected - full[:2, :1])) > 0.05


def test_complex_wave_covariance_and_owned_metric():
    parent = model(-np.eye(3), (0, 1), "degenerate complete H")
    wave = np.array([[1.0, 0], [0, 1], [0.2j, 0.3]], complex)
    actual = FiniteCorrelatedWave(
        parent, wave, -np.eye(2), source_identity="complex wave"
    )
    g = wave.conj().T @ wave
    v, u = np.linalg.eigh(g)
    w = wave @ ((u / np.sqrt(v)) @ u.conj().T)
    op = FiniteCITransitionOperator(
        np.array([[0, 1j, 0.1], [-1j, 0, 0.3], [0.1, 0.3, 0.4]]),
        parent,
        parent,
        operator_identity="Hermitian control",
        unit="au",
    )
    np.testing.assert_allclose(actual.metric, g, atol=1e-15)
    np.testing.assert_allclose(
        actual.effective_transition(op, actual, representation="orthonormal"),
        w.conj().T @ op.matrix @ w,
        atol=1e-15,
    )
    assert abs(g[0, 1].imag) > 0.01
    phase = np.diag([-1.0, 1.0, -1.0])
    phase_p = phase[:2, :2]
    changed = FiniteCorrelatedWave(
        parent, phase @ wave @ phase_p, -np.eye(2), source_identity="full basis phase"
    )
    changed_op = FiniteCITransitionOperator(
        phase @ op.matrix @ phase,
        parent,
        parent,
        operator_identity="same physical phase",
        unit="au",
    )
    np.testing.assert_allclose(
        changed.effective_transition(changed_op, changed, representation="orthonormal"),
        phase_p @ (w.conj().T @ op.matrix @ w) @ phase_p,
        atol=1e-15,
    )
    wave[:] = 10
    assert actual.wave_operator[0, 0] == 1
    with pytest.raises(ValueError):
        actual.metric.setflags(write=True)


def test_gate_rejects_wrong_metric_wrong_hamiltonian_and_resources():
    parent = model(np.diag([-0.5, 0.2, 1.0]), (0,), "test")
    wave = np.array([[1.0], [0], [0]])
    for opts, message in (
        (dict(metric=np.array([[2.0]])), "metric"),
        (dict(memory_bytes=1), "cap"),
        (dict(dimension_cap=2), "cap"),
    ):
        with pytest.raises((ValueError, MemoryError), match=message):
            FiniteCorrelatedWave(
                parent, wave, np.array([[-0.5]]), source_identity="input", **opts
            )
    with pytest.raises(ValueError, match="equation"):
        FiniteCorrelatedWave(parent, wave, np.array([[-0.4]]), source_identity="bad H")
    actual = FiniteCorrelatedWave(
        parent, wave, np.array([[-0.5]]), source_identity="good"
    )
    clone = model(parent.hamiltonian_Ha, (0,), "test")
    op = FiniteCITransitionOperator(
        np.eye(3), clone, clone, operator_identity="wrong snapshot", unit="au"
    )
    with pytest.raises(ValueError, match="snapshot"):
        actual.effective_transition(op, actual, representation="orthonormal")
