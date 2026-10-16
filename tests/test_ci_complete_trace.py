"""Full-Fock trace identities independently fix the meaning of CI ensembles.

For M spin orbitals, uniform averaging of C(M,N) determinants gives
gamma_ij=(N/M)delta_ij and the pair count N(N-1)/2. These are combinatorial
identities, independent of any radial solver, CI Hamiltonian or experiment.
"""
from math import comb

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.ci import CIResult
from atomx.physics.csf.levels import ci_level_blocks
from atomx.physics.csf.radial_density import CIRadialDensity
from atomx.physics.csf.state_average import CIStateAverage


@pytest.mark.parametrize("number", [1, 2, 3])
@pytest.mark.parametrize("rotate", [False, True])
def test_complete_magnetic_trace_has_uniform_spinor_density_and_exact_pair_count(number, rotate):
    keys = ((1, -1), (2, -2), (2, -1), (2, 1))
    spinors = sum(2*abs(kappa) for _, kappa in keys)
    csfs = enumerate_jj_orbital_space(keys, number, magnetic_state_cap=512)
    assert sum(c.total_J2+1 for c in csfs) == comb(spinors, number)
    # Any orthonormal basis in each J block spans the same uniform ensemble.
    result = CIResult(np.arange(len(csfs), dtype=float), np.eye(len(csfs)))
    if rotate:
        _, blocks = ci_level_blocks(csfs, result, require_full=True)
        rng = np.random.default_rng(614)
        for columns in blocks.values():
            q, _ = np.linalg.qr(rng.normal(size=(len(columns), len(columns))))
            result.mixing_coefficients[np.ix_(columns, columns)] = q
    target = CIStateAverage(statistical=True)
    contracted = CIRadialDensity(csfs).contract(result, target.weights(csfs, result))
    for na, kappa in keys:
        for nc, kc in keys:
            if kc != kappa:
                continue
            expected = number/spinors*(2*abs(kappa)) if na == nc else 0.
            assert contracted.get(('h', na, kappa, nc), 0.) == pytest.approx(expected, abs=2e-13)
    pairs = sum(value for key, value in contracted.items()
                if key[0] == 'r' and key[1] == 0 and key[2] == key[3] and key[4] == key[5])
    assert pairs == pytest.approx(number*(number-1)/2, abs=2e-13)


def test_equal_J_roots_are_a_different_objective_than_complete_magnetic_trace():
    keys = ((1, -1), (2, -2), (2, -1), (2, 1))
    csfs = enumerate_jj_orbital_space(keys, 3, magnetic_state_cap=512)
    result = CIResult(np.arange(len(csfs), dtype=float), np.eye(len(csfs)))
    density = CIRadialDensity(csfs)
    roots = density.contract(result, CIStateAverage().weights(csfs, result))
    trace = density.contract(result, CIStateAverage(statistical=True).weights(csfs, result))
    assert roots[('h', 2, 1, 2)]/2 != pytest.approx(roots[('h', 2, -2, 2)]/4, abs=1e-4)
    assert trace[('h', 2, 1, 2)]/2 == pytest.approx(trace[('h', 2, -2, 2)]/4, abs=2e-13)


def test_sparse_density_preserves_signed_bilinear_against_complete_hamiltonian(monkeypatch):
    from scipy.sparse import isspmatrix_csr

    from atomx.angular import csf_hamiltonian as exact

    def reject_dense(*_args, **_kwargs):
        raise AssertionError('density contraction requested a dense CSF projection')

    monkeypatch.setattr(exact, '_block_csf_matrix', reject_dense)
    keys = ((1, -1), (2, -2), (2, -1), (2, 1))
    csfs = list(reversed(enumerate_jj_orbital_space(keys, 3, magnetic_state_cap=512)))
    density = CIRadialDensity(csfs)
    assert all(isspmatrix_csr(projection) for _, projection, _ in density.blocks)
    rng = np.random.default_rng(716)
    left, right = rng.normal(size=(2, len(csfs)))
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)

    def evaluate(term):
        if term[0] == 'h':
            _, na, kappa, nb = term
            return -1.0 / (na * nb * (1 + abs(kappa)))
        _, rank, *orbitals = term
        return 1.0 / (1 + rank + sum(n for n, _ in orbitals))

    weights = density.contract_bilinear(left, right)
    from_density = sum(value * evaluate(term) for term, value in weights.items())
    from_hamiltonian = 0.0
    sectors = {(csf.total_J2, csf.parity) for csf in csfs}
    for sector in sectors:
        rows = [i for i, csf in enumerate(csfs)
                if (csf.total_J2, csf.parity) == sector]
        signatures = [exact.csf_signature(csfs[i]) for i in rows]
        block = exact.exact_block_matrix(signatures, evaluate)
        from_hamiltonian += left[rows] @ block @ right[rows]
    assert from_density == pytest.approx(from_hamiltonian, rel=0, abs=2e-12)
    assert any(value < 0 for value in weights.values())
    assert any(value > 0 for value in weights.values())
    unsigned = sum(abs(value) * evaluate(term) for term, value in weights.items())
    assert abs(unsigned - from_hamiltonian) > 1e-5
