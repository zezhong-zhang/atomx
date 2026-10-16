"""Physical and work-count checks for occupied same-kappa rotations."""

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, Subshell


def test_pair_rotation_restores_variational_li_energy_with_six_fock_builds(
        monkeypatch):
    solver = DFSolver(
        Z=3,
        subshells=[Subshell(1, -1, 2), Subshell(2, -1, 1)],
        ns=45, ks=9, exchange="hf",
    )
    result = solver.solve(max_iter=30)
    assert result.converged
    orbitals = {
        kappa: [(p.copy(), q.copy(), occupation) for p, q, occupation in shells]
        for kappa, shells in result.orbitals_raw.items()
    }

    # Rotate the two occupied s orbitals away from the stationary solution.
    (p0, q0, n0), (p1, q1, n1) = orbitals[-1]
    cosine, sine = np.cos(0.05), np.sin(0.05)
    orbitals[-1][0] = (cosine*p0 + sine*p1, cosine*q0 + sine*q1, n0)
    orbitals[-1][1] = (cosine*p1 - sine*p0, cosine*q1 - sine*q0, n1)
    before = solver._compute_total_energy(orbitals)[0]

    calls = 0
    original = solver._build_hf_matrix_dbsr

    def count_fock_builds(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_build_hf_matrix_dbsr", count_fock_builds)
    angle = solver._optimize_same_kappa_rotations(orbitals)
    after = solver._compute_total_energy(orbitals)[0]

    assert calls == 6  # three density samples for each of two target fields
    assert angle == pytest.approx(0.05, abs=1e-4)
    assert after < before - 0.005
    assert after == pytest.approx(result.total_energy, abs=1e-7)


def test_local_core_hole_rotation_removes_false_sulfur_stationarity(monkeypatch):
    import atomx.methods.dirac_fock as module
    from atomx import Atom, Basis
    from atomx.methods.dirac_fock import DiracFock

    captured = []
    original = module.DFSolver

    class LocalRotationSolver(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(module, 'DFSolver', LocalRotationSolver)
    energies = []
    for start in ('gsz', 'tfda'):
        result = Atom(Z=16, config='1s2 2s2 2p6 3s2 3p3 5p1',
                      engine='bspline', basis=Basis(r_max=75, grid_hmax=1)).solve(
            DiracFock(starting_potential=start),
            tol=1e-8, max_iter=200).result
        assert result.converged
        solver = captured[-1]
        orbitals = result.orbitals_raw
        energies.append(solver._compute_total_energy(orbitals)[0])
        # Finite differences test the actual energy functional rather than
        # duplicating the interpolation used by the rotation implementation.
        for kappa in (1, -2):
            pi, qi, ni = orbitals[kappa][1]
            pj, qj, nj = orbitals[kappa][2]
            values = []
            for angle in (-1e-4, 1e-4):
                c, s = np.cos(angle), np.sin(angle)
                orbitals[kappa][1] = (c*pi+s*pj, c*qi+s*qj, ni)
                orbitals[kappa][2] = (c*pj-s*pi, c*qj-s*qi, nj)
                values.append(solver._compute_total_energy(orbitals)[0])
            orbitals[kappa][1] = (pi, qi, ni)
            orbitals[kappa][2] = (pj, qj, nj)
            assert abs((values[1]-values[0])/2e-4) < 1e-6
    assert abs(energies[0]-energies[1]) < 1e-8
