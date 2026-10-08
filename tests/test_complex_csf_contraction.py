"""Complex radial data reuse exact signed jj angular algebra without casts."""

import hashlib

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import csf_signature, exact_block_matrix
from atomx.config.csf import enumerate_jj_orbital_space


def signatures(keys, N, J2=0, parity=1):
    return [csf_signature(c) for c in enumerate_jj_orbital_space(keys, N)
            if (c.total_J2, c.parity) == (J2, parity)]


def test_closed_s_shell_complex_energy_equals_two_h_plus_direct_F0():
    sigs = signatures(((1, -1),), 2)
    def integral(key):
        return -5-.1j if key[0] == 'h' else .7-.2j
    H = exact_block_matrix(sigs, integral, integral_dtype=complex)
    np.testing.assert_allclose(H, [[-9.3-.4j]], atol=1e-14)
    # NumPy complex scalars must also reject instead of issuing only a cast warning.
    with pytest.raises(ValueError, match='complex radial'):
        exact_block_matrix(sigs, lambda k: np.complex128(integral(k)))


@pytest.mark.parametrize('N,J2,parity', [(2, 0, 1), (3, 1, 1), (3, 3, -1)])
def test_complex_projection_is_linear_and_operator_adjoint_is_correct(N, J2, parity):
    sigs = signatures(((1, -1), (2, -1), (2, 1), (2, -2)), N, J2, parity)
    assert len(sigs) > 1
    def values(key):
        d = hashlib.sha256(repr(key).encode()).digest()
        return (int.from_bytes(d[:4], 'little')/2**32,
                int.from_bytes(d[4:8], 'little')/2**32)
    real = exact_block_matrix(sigs, lambda k: values(k)[0])
    imag = exact_block_matrix(sigs, lambda k: values(k)[1])
    def evaluator(k):
        return complex(*values(k))
    H = exact_block_matrix(sigs, evaluator, integral_dtype=complex)
    np.testing.assert_allclose(H, real+1j*imag, atol=1e-14)
    np.testing.assert_array_equal(H, H.T)
    assert np.linalg.norm(H-H.conj().T) > .1
    op = exact_block_matrix(sigs, evaluator, integral_dtype=complex, as_operator=True)
    x = np.arange(len(sigs))+1j*np.arange(len(sigs))[::-1]
    np.testing.assert_allclose(op@x, H@x, atol=1e-13)
    np.testing.assert_allclose(op.rmatvec(x), H.conj().T@x, atol=1e-13)
    np.testing.assert_allclose(op.diagonal(), np.diag(H), atol=1e-14)
    np.testing.assert_allclose(op@np.eye(len(sigs)), H, atol=1e-14)


def test_complex_nonfinite_integrals_reject_before_projection():
    sigs = signatures(((1, -1),), 2)
    with pytest.raises(ValueError, match='nonfinite'):
        exact_block_matrix(sigs, lambda k: complex(1, np.nan), integral_dtype=complex)
    with pytest.raises(ValueError, match='integral_dtype'):
        exact_block_matrix(sigs, lambda k: 1, integral_dtype=np.float32)
