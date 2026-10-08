"""Independent superoperator oracle for absolute radiative error certification."""

import numpy as np
import pytest
from scipy.linalg import expm

from atomx.xfel import CoherentGaussianPulse, CoherentSystem
from atomx.xfel.radiative_reservoir import (
    arithmetic_difference_bound,
    cluster_radiative_baths,
    finite_time_error_bound,
)


def toy(scale=1):
    operators = []
    for level, energy in ((1, 1.0), (2, 1.2)):
        op = np.zeros((3, 3), complex)
        op[0, level] = (1 if level == 1 else -1j) * np.sqrt(scale * energy**3)
        operators.append((energy, [np.zeros_like(op), op, np.zeros_like(op)], 1.0))
    return operators


def independent_generator(baths, arithmetic):
    """Construct by acting on matrix units, without atomx's kron kernel."""
    n = baths[0][1][0].shape[0]
    columns = []
    for col in range(n):
        for row in range(n):
            rho = np.zeros((n, n), complex)
            rho[row, col] = 1
            out = np.zeros_like(rho)
            for ea, ga, _ in baths:
                for eb, gb, _ in baths:
                    coefficient = (ea**3 + eb**3) / (2 * np.sqrt(ea**3 * eb**3)) if arithmetic else 1
                    for a, b in zip(ga, gb, strict=True):
                        product = b.conj().T @ a
                        out += coefficient * (a @ rho @ b.conj().T - .5 * (product @ rho + rho @ product))
            columns.append(out.ravel(order='F'))
    return np.column_stack(columns)


def test_bound_covers_independent_nonsecular_density_oracle():
    baths = toy()
    eta = arithmetic_difference_bound(baths)
    reference = independent_generator(baths, True)
    geometric = independent_generator(baths, False)
    rng = np.random.default_rng(33)
    for elapsed_s in (.001, .02, .1):
        limit = finite_time_error_bound(eta, elapsed_s * 1e15)
        for _ in range(8):
            x = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            density = x @ x.conj().T
            density /= np.trace(density)
            a = (expm(reference * elapsed_s) @ density.ravel(order='F')).reshape((3, 3), order='F')
            b = (expm(geometric * elapsed_s) @ density.ravel(order='F')).reshape((3, 3), order='F')
            difference = np.linalg.svd(a-b, compute_uv=False).sum()
            assert 0 < difference < limit
            assert np.linalg.eigvalsh(b).min() > -1e-14
            assert np.trace(b) == pytest.approx(1, abs=1e-14)
    # Independent analytic eta for this pair; catches factor-of-two/units.
    expected = 2 * (1 + 1.2**3 - 2 * 1.2**1.5)
    assert eta == pytest.approx(expected, rel=2e-14)


def test_legacy_coefficient_gate_preserved_bounded_mode_has_absolute_budget():
    baths = toy(1e7)
    with pytest.raises(ValueError, match='vacuum spectral approximation'):
        cluster_radiative_baths(baths, 'geometric_clustered')
    merged, factors, records = cluster_radiative_baths(baths, 'geometric_bounded')
    assert factors == {1.0: 1.0, 1.2: 1.0}
    system = CoherentSystem({'g': 0, 'a': 1, 'b': 1.2}, dict.fromkeys(('g', 'a', 'b'), 0),
        np.zeros((3, 3)), charges=dict.fromkeys(('g', 'a', 'b'), 1), decay_operators=merged[0][1],
        provenance={'radiative_arithmetic_generator_difference_bound_per_s':
                    records[0]['arithmetic_generator_difference_bound_per_s'],
                    'radiative_arithmetic_trace_norm_error_budget': 1e-3})
    pulse = CoherentGaussianPulse(100, 0, 10)
    density = np.diag([0, .5, .5])
    result = system.solve(pulse, density, after_pulse_fs=100)
    assert 0 < result.provenance['radiative_arithmetic_trace_norm_error_bound'] < 1e-3
    with pytest.raises(ValueError, match='trace-norm budget'):
        system.solve(pulse, density, after_pulse_fs=1e7)


def test_error_bound_preserves_complex_phase_and_degeneracy():
    baths = toy(1e8)
    transformed = []
    phases = np.diag([1, 1j, -1])
    for energy, group, width in baths:
        transformed.append((energy, [phases @ op @ phases.conj().T for op in group], width))
    assert arithmetic_difference_bound(transformed) == arithmetic_difference_bound(baths)
    baths[1] = (1.0, baths[1][1], baths[1][2])
    assert arithmetic_difference_bound(baths) == 0
    assert finite_time_error_bound(0, 1e20) == 0
    assert finite_time_error_bound(1e300, 1e10) == np.inf
    with pytest.raises(ValueError, match='nonnegative'):
        finite_time_error_bound(-1, 1)


def test_sparse_loss_representation_preserves_signed_full_trajectory():
    from scipy.sparse import csc_matrix, issparse

    from atomx.xfel.multielectron_coherent import MultiElectronCoherentModel

    # Shared complex paths: dropping signs/off-diagonals must fail this oracle.
    decay = np.zeros((3, 3), complex)
    decay[0, 1], decay[0, 2] = 1e6, -2j * 1e6
    photo = np.zeros((3, 3), complex)
    photo[0, 1], photo[0, 2] = 1e-10, 2j * 1e-10
    # Equal photon ladders admit the shared signed paths without a time-dependent bath.
    fixed = dict.fromkeys(('ion', 'g', 'e'), 0)
    args = ({'ion': 0, 'g': 1, 'e': 1.2}, fixed, np.zeros((3, 3), complex))
    dense = CoherentSystem(*args, charges={'ion': 1, 'g': 0, 'e': 0},
        decay_operators=[decay], pi_operators=[photo], provenance={'oracle': 'dense'})
    sparse = CoherentSystem(*args, charges={'ion': 1, 'g': 0, 'e': 0},
        decay_operators=[csc_matrix(decay)], pi_operators=[csc_matrix(photo)], provenance={'oracle': 'sparse'})
    np.testing.assert_array_equal(dense.decay_liouvillian.toarray(), sparse.decay_liouvillian.toarray())
    pulse = CoherentGaussianPulse(100, 1e16, 10)
    state = np.array([0, 1, 1j]) / np.sqrt(2)
    rho = np.outer(state, state.conj())
    a, b = dense.solve(pulse, rho), sparse.solve(pulse, rho)
    np.testing.assert_array_equal(a.density_matrices, b.density_matrices)
    observable = decay.conj().T @ decay
    np.testing.assert_allclose(a.expectation(observable), b.expectation(csc_matrix(observable)), rtol=2e-15)
    # Production threshold changes representation only, retaining complex entries.
    op = MultiElectronCoherentModel._new_jump(201)
    op[3, 71], op[3, 72] = 1, -2j
    ready = MultiElectronCoherentModel._finish_jump(op)
    assert issparse(ready) and ready.nnz == 2
    assert ready[3, 72] == -2j
    np.testing.assert_array_equal(MultiElectronCoherentModel._column_rates([ready], 201),
                                  np.asarray(abs(ready.toarray())**2).sum(axis=0))
    with pytest.raises(ValueError, match='explicit state basis'):
        CoherentSystem(*args, charges={'ion': 1, 'g': 0, 'e': 0},
            decay_operators=[csc_matrix(decay * np.nan)], provenance={'oracle': 'nonfinite'})


def test_large_basis_sparse_hamiltonian_against_analytic_pulse_unitary():
    from scipy.special import erf

    from atomx.constants import AU_RATE_PER_S

    # Two interfering signed E1 paths, embedded in a larger inert basis.
    n = 129
    energies = dict.fromkeys(range(n), 0.)
    energies[1] = energies[2] = 100.
    ladders = dict.fromkeys(range(n), 0)
    ladders[1] = ladders[2] = 1
    dipole = np.zeros((n, n), complex)
    dipole[1, 0], dipole[2, 0] = .2, -.13j
    system = CoherentSystem(energies, ladders, dipole, charges=dict.fromkeys(range(n), 0),
                            provenance={'oracle': 'Gaussian commuting unitary'})
    pulse = CoherentGaussianPulse(100, 1e17, 10, phase_rad=.7)
    rho = np.zeros((n, n), complex)
    rho[0, 0] = 1
    times = np.linspace(*pulse.support_fs, 51)
    result = system.solve(pulse, rho, times_fs=times, rtol=1e-11, atol=1e-14)
    assert result.provenance['hamiltonian_action'] == 'diagonal plus sparse E1'
    sigma = pulse.fwhm_fs / np.sqrt(8*np.log(2))
    peak = abs(pulse.electric_field_au(0))
    h = np.exp(1j*.7)*dipole[:3, :3]+np.exp(-1j*.7)*dipole[:3, :3].conj().T
    for t, density in zip(times, result.density_matrices, strict=True):
        area = sigma*np.sqrt(np.pi)*(erf(t/(2*sigma))-erf(times[0]/(2*sigma)))
        theta = -.5*AU_RATE_PER_S*1e-15*peak*area
        state = expm(-1j*theta*h)[:, 0]
        expected = np.zeros((n,n), complex)
        expected[:3,:3] = np.outer(state, state.conj())
        np.testing.assert_allclose(density, expected, rtol=0, atol=2e-10)


def test_signed_electron_operator_bound_against_independent_cptp_reference():
    from atomx.xfel.reservoir_errors import lindblad_difference_bound

    c, b = np.zeros((3,3), complex), np.zeros((3,3), complex)
    c[0, 1], c[0, 2] = 1, -1.1j
    b[0, 1], b[0, 2] = 1.0003, -1.099j
    eta = lindblad_difference_bound(c, b)
    def generator(op):
        columns = []
        cc = op.conj().T @ op
        for j in range(3):
            for i in range(3):
                x = np.zeros((3,3), complex)
                x[i,j] = 1
                columns.append((op @ x @ op.conj().T - .5*(cc @ x+x @ cc)).ravel(order='F'))
        return np.column_stack(columns)
    reference, approximate = generator(c), generator(b)
    density = np.outer([0, 1/np.sqrt(2), 1j/np.sqrt(2)], [0, 1/np.sqrt(2), -1j/np.sqrt(2)])
    for elapsed in (.01, .1, 1):
        a = (expm(reference*elapsed) @ density.ravel(order='F')).reshape((3,3),order='F')
        d = (expm(approximate*elapsed) @ density.ravel(order='F')).reshape((3,3),order='F')
        assert 0 < np.linalg.svd(a-d, compute_uv=False).sum() < eta*elapsed
    np.testing.assert_allclose(lindblad_difference_bound(c[:1,1:],b[:1,1:]), eta, rtol=1e-14)
    phases = np.diag([1, 1j, -1])
    assert lindblad_difference_bound(phases@c@phases.conj().T, phases@b@phases.conj().T) == eta


def test_combined_reservoir_budget_rechecks_fluence_and_elapsed_time():
    system = CoherentSystem({'g': 0}, {'g': 0}, np.zeros((1,1)), charges={'g': 0},
        provenance={'electron_pi_generator_difference_bound_cm2': 1e-20,
                    'electron_auger_generator_difference_bound_per_s': 1e5,
                    'radiative_arithmetic_generator_difference_bound_per_s': 0.,
                    'radiative_arithmetic_trace_norm_error_budget': 1e-3,
                    'reservoir_trace_norm_error_budget': 1e-3})
    pulse = CoherentGaussianPulse(100, 1e16, 10)
    result = system.solve(pulse, np.ones((1,1)), after_pulse_fs=100)
    expected = np.expm1(1e-4+1e5*(pulse.support_fs[1]-pulse.support_fs[0]+100)*1e-15)
    assert result.provenance['reservoir_reference_trace_norm_error_bound'] == pytest.approx(expected)
    with pytest.raises(ValueError, match='combined reservoir'):
        system.solve(CoherentGaussianPulse(100, 1e18, 10), np.ones((1,1)))
    with pytest.raises(ValueError, match='combined reservoir'):
        system.solve(pulse, np.ones((1,1)), after_pulse_fs=1e9)


def test_large_sparse_drive_and_loss_with_detuning_against_independent_density_ode():
    from scipy.integrate import solve_ivp
    from scipy.sparse import csc_matrix

    from atomx.constants import AU_RATE_PER_S, HARTREE_EV

    n = 129
    energies = dict.fromkeys(range(n), 0.)
    energies[1], energies[2] = 100.1, 100.3
    ladders = dict.fromkeys(range(n), 0)
    ladders[1] = ladders[2] = 1
    dipole, decay = np.zeros((n,n), complex), np.zeros((n,n), complex)
    dipole[1,0], dipole[2,0] = .2, -.13j
    decay[0,1], decay[0,2] = 1e6, -2j*1e6
    system = CoherentSystem(energies, ladders, dipole, charges=dict.fromkeys(range(n), 0),
        decay_operators=[csc_matrix(decay)], provenance={'oracle': 'detuned direct density ODE'})
    pulse = CoherentGaussianPulse(100, 1e17, 10, phase_rad=.7)
    rho = np.zeros((n,n), complex)
    rho[0,0] = 1
    times = np.linspace(*pulse.support_fs, 51)
    result = system.solve(pulse, rho, times_fs=times, rtol=1e-11, atol=1e-14)
    rate_fs = AU_RATE_PER_S*1e-15
    diagonal = np.diag([0, .1, .3])*rate_fs/HARTREE_EV
    c = decay[:3,:3]*np.sqrt(1e-15)
    cc = c.conj().T@c
    def rhs(t, vector):
        density = vector.reshape((3,3))
        v = -.5*pulse.electric_field_au(t)*rate_fs*dipole[:3,:3]
        h = diagonal+v+v.conj().T
        return (-1j*(h@density-density@h)+c@density@c.conj().T
                -.5*(cc@density+density@cc)).ravel()
    reference = solve_ivp(rhs, pulse.support_fs, rho[:3,:3].ravel(), t_eval=times,
                          method='DOP853',rtol=1e-12,atol=1e-15,max_step=.05)
    assert reference.success
    expected = np.zeros_like(result.density_matrices)
    expected[:,:3,:3] = reference.y.T.reshape((-1,3,3))
    np.testing.assert_allclose(result.density_matrices, expected, rtol=0, atol=2e-9)
