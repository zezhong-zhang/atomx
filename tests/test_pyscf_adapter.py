"""PySCF molecular adapter round-trip tests (HYP-904, stage G1).

Skipped when PySCF is not installed (optional ``atomx[pyscf]`` extra).
Verifies that a converged small-molecule RHF is faithfully translated into
an ElectronicStructurePayload: AO/MO coefficients, non-orthogonal overlap,
density, occupations, provenance, and the orthonormality consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from pyscf import gto, scf  # noqa: E402

from atomx.providers.contract import Capability  # noqa: E402
from atomx.providers.pyscf_adapter import (  # noqa: E402
    PySCFProvider,
    from_core_hole_delta_scf,
    from_scf,
)
from atomx.scattering.autoionization.molecular import (  # noqa: E402
    molecular_auger_initial_state,
)


def _h2(basis="sto-3g"):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _h2o(basis="sto-3g"):
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis=basis,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def test_h2_roundtrip_basic_shapes():
    mol, mf = _h2()
    payload = from_scf(mf)
    assert payload.Z == 2
    assert payload.orbitals is not None
    assert payload.orbitals.n_ao == mol.nao
    assert payload.orbitals.n_mo == mf.mo_coeff.shape[1]
    np.testing.assert_allclose(
        payload.orbitals.coefficients, mf.mo_coeff, atol=0
    )
    np.testing.assert_allclose(payload.orbitals.energies, mf.mo_energy, atol=0)


def test_h2_capabilities_and_provenance():
    _, mf = _h2()
    prov = PySCFProvider(mf)
    caps = prov.capabilities()
    assert Capability.MO_COEFFICIENTS in caps
    assert Capability.OVERLAP_S in caps
    assert Capability.DENSITY_MATRIX in caps
    assert Capability.ORBITAL_ENERGIES in caps
    assert Capability.OCCUPATIONS in caps
    payload = prov.build_payload()
    assert payload.provenance.provider == "pyscf"
    assert payload.provenance.basis == "sto-3g"
    assert "nonrel-gto" in payload.provenance.approximations
    assert payload.provenance.code_version.startswith("pyscf ")


def test_h2_overlap_is_nonorthogonal_but_mos_orthonormal():
    _, mf = _h2()
    payload = from_scf(mf)
    # AO overlap has non-trivial off-diagonal (non-orthogonal GTO basis).
    assert payload.overlap.is_symmetric()
    assert not payload.overlap.is_orthonormal(tol=1e-3)
    assert payload.overlap.max_deviation_from_identity() > 0.1
    # ... yet the MOs are orthonormal in the S metric: C^T S C = I.
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-8)


def test_h2_density_recovers_electron_count():
    mol, mf = _h2()
    payload = from_scf(mf)
    assert payload.electron_count() == pytest.approx(mol.nelectron, abs=1e-8)


def test_h2_with_eri_exposes_two_electron_integrals():
    mol, mf = _h2()
    payload = from_scf(mf, with_eri=True)
    assert Capability.TWO_ELECTRON_INTEGRALS in payload.capabilities()
    eri = payload.two_electron.eri
    n = mol.nao
    assert eri.shape == (n, n, n, n)
    # (ij|kl) 8-fold symmetry: (ij|kl) == (kl|ij).
    np.testing.assert_allclose(
        eri, np.transpose(eri, (2, 3, 0, 1)), atol=1e-10
    )


def test_h2o_roundtrip_validates():
    mol, mf = _h2o()
    payload = from_scf(mf)
    assert payload.Z == int(sum(mol.atom_charges()))  # 10
    assert payload.validate() is not None
    assert payload.electron_count() == pytest.approx(mol.nelectron, abs=1e-7)
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-7)


def test_unsolved_meanfield_rejected():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)  # not kernel()-ed
    with pytest.raises(Exception):
        PySCFProvider(mf)


def test_uhf_rejected_fail_closed():
    mol = gto.M(
        atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0
    )
    mf = scf.UHF(mol)
    mf.kernel()
    with pytest.raises(Exception):
        PySCFProvider(mf)


def test_h2o_o_1s_core_hole_delta_scf_payload():
    mol, mf = _h2o()
    payload = from_core_hole_delta_scf(mf, "O 1s", max_cycle=80)
    assert Capability.CORE_HOLE_SPEC in payload.capabilities()
    assert payload.provenance.core_hole == "O 1s"
    assert payload.provenance.energy_zero == "delta_scf_total_energy"
    assert payload.metadata["n_electrons"] == mol.nelectron - 1
    assert payload.metadata["core_hole"]["occupation_before"] == pytest.approx(2.0)
    assert payload.metadata["core_hole"]["occupation_after"] == pytest.approx(1.0)
    assert payload.metadata["delta_scf_energy_hartree"] == pytest.approx(
        payload.metadata["core_hole"]["initial_state_energy_hartree"]
    )
    assert payload.metadata["delta_scf_energy_hartree"] > 0.0
    assert payload.electron_count() == pytest.approx(mol.nelectron - 1, abs=1e-7)
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-7)

    state = molecular_auger_initial_state(payload)
    assert state.orbital_label == "O 1s"
    assert state.occupation_delta == pytest.approx(-1.0)
    assert state.initial_energy == pytest.approx(
        payload.metadata["delta_scf_energy_hartree"]
    )
