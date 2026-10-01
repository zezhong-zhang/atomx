"""DBSR radial/energy comparison independent of orbital optimization."""
import json
from pathlib import Path

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, neon_subshells
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices

DATA = Path(__file__).parent/'data'


@pytest.fixture(scope='module')
def common_state():
    reference = json.loads((DATA/'dbsr_ne_reference.json').read_text())
    with np.load(DATA/'dbsr_ne_spinors.npz') as arrays:
        knots, pa, qa, labels = [arrays[key].copy() for key in ('knots','pa','qa','labels')]
    ns, ks = reference['ns'], reference['ks']
    solver = DFSolver(10, neon_subshells(), ns=ns, ks=ks, nuclear='fermi',
        nuclear_mass_number=20, kinetic_balance=True, include_breit=False, include_qed=False)
    solver.t = knots
    solver.bsp_data = prepare_bspline_data(knots, ns, ks, kinetic_balance=True)
    solver._precompute()
    orbitals, schedule = {}, []
    for shell in solver.subshells:
        index = list(labels).index(shell.label)
        group = orbitals.setdefault(shell.kappa, [])
        schedule.append((shell, shell.kappa, len(group)))
        group.append((pa[index], qa[index], shell.occupation))
    solver._precompute_angular_coefficients(schedule)
    return solver, orbitals, schedule, reference


def test_same_spinor_one_and_two_electron_energies(common_state):
    solver, orbitals, _, reference = common_state
    energy, components = solver._compute_total_energy(orbitals)
    # On common spinors, isotope/model/SCF differences cannot conceal an
    # operator convention or integral defect. The corrected reference's
    # one-body, two-body and total energies are independently printed.
    assert energy == pytest.approx(reference['reference']['total_energy_Ha'], abs=2e-10, rel=0)
    for key in ('E_1e','E_2e'):
        assert components[key] == pytest.approx(reference['components'][key], abs=2e-10, rel=0)


def test_same_spinor_final_fock_expectations(common_state):
    solver, orbitals, schedule, reference = common_state
    ns = solver.ns
    for i, (shell, kappa, index) in enumerate(schedule):
        p, q, _ = orbitals[kappa][index]
        vector = np.concatenate((p,q))
        h, s = build_dirac_matrices(ns, kappa, 10, solver.bsp_data, solver._nuclear_potential)
        pp, qq, _, qp = solver._build_hf_matrix_dbsr(i, orbitals, skip_pq_exchange=True)
        h[:ns,:ns] += pp
        h[ns:,ns:] += qq
        h[ns:,:ns] += qp
        h = np.tril(h)+np.tril(h,-1).T
        assert vector @ s @ vector == pytest.approx(1., abs=5e-13, rel=0)
        assert vector @ h @ vector == pytest.approx(
            reference['reference']['orbital_fock_expectations_Ha'][shell.label], abs=2e-10, rel=0)
