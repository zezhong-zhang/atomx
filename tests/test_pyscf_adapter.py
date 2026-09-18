"""PySCF molecular adapter round-trip tests (HYP-904, stage G1).

Skipped when PySCF is not installed (optional ``atomx[pyscf]`` extra).
Verifies that a converged small-molecule RHF is faithfully translated into
an ElectronicStructurePayload: AO/MO coefficients, non-orthogonal overlap,
density, occupations, provenance, and the orthonormality consistency.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from atomx.providers.contract import (  # noqa: E402
    Capability,
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    ProviderContractError,
)
from atomx.providers.pyscf_adapter import (  # noqa: E402
    PySCFProvider,
    _core_hole_metadata,
    _mom_core_hole_occ,
    from_core_hole_delta_scf,
    from_scf,
)
from atomx.scattering.autoionization.molecular import (  # noqa: E402
    molecular_auger_initial_state,
)


def _h2(basis="sto-3g"):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _h2o(basis="sto-3g"):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

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
    assert int(sum(mol.atom_charges())) == payload.Z  # 10
    assert payload.validate() is not None
    assert payload.electron_count() == pytest.approx(mol.nelectron, abs=1e-7)
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-7)


def test_unsolved_meanfield_rejected():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)  # not kernel()-ed
    with pytest.raises(ProviderContractError, match="not solved yet"):
        PySCFProvider(mf)


def test_unconverged_meanfield_with_orbitals_rejected():
    _, mf = _h2()
    assert mf.mo_coeff is not None
    mf.converged = False

    with pytest.raises(
        ProviderContractError,
        match="input PySCF mean-field did not converge",
    ):
        PySCFProvider(mf)


def test_build_payload_revalidates_convergence_before_consuming_orbitals():
    _, mf = _h2()
    provider = PySCFProvider(mf)
    mf.converged = False

    with pytest.raises(
        ProviderContractError,
        match="input PySCF mean-field did not converge",
    ):
        provider.build_payload()


def test_unconverged_meanfield_rejected_before_mo_coeff_probe():
    pytest.importorskip("pyscf")

    class UnconvergedMeanField:
        converged = False
        mo_coeff_accessed = False

        @property
        def mo_coeff(self):
            self.mo_coeff_accessed = True
            raise AssertionError("mo_coeff should not be accessed")

    mf = UnconvergedMeanField()

    with pytest.raises(
        ProviderContractError,
        match="input PySCF mean-field did not converge",
    ):
        PySCFProvider(mf)
    assert mf.mo_coeff_accessed is False


def test_meanfield_missing_converged_status_rejected_fail_closed():
    _, mf = _h2()
    no_status = SimpleNamespace(mol=mf.mol, mo_coeff=mf.mo_coeff)

    with pytest.raises(
        ProviderContractError,
        match="convergence status is unavailable",
    ):
        PySCFProvider(no_status)


def test_throwing_converged_descriptor_rejected_without_leaking_detail():
    pytest.importorskip("pyscf")

    class ThrowingConvergedMeanField:
        @property
        def converged(self):
            raise RuntimeError("attacker-controlled detail")

        @property
        def mo_coeff(self):
            raise AssertionError("mo_coeff should not be accessed")

    with pytest.raises(ProviderContractError) as exc_info:
        PySCFProvider(ThrowingConvergedMeanField())
    message = str(exc_info.value)
    assert message == "input PySCF mean-field convergence status is unavailable"
    assert "attacker-controlled" not in message
    assert exc_info.value.__suppress_context__ is True


def test_truthy_non_bool_converged_rejected_fail_closed():
    _, mf = _h2()
    truthy_status = SimpleNamespace(converged=1, mol=mf.mol, mo_coeff=mf.mo_coeff)

    with pytest.raises(
        ProviderContractError,
        match="input PySCF mean-field did not converge",
    ):
        PySCFProvider(truthy_status)


def test_core_hole_delta_scf_rejects_unconverged_ground_state():
    _, mf = _h2o()
    assert mf.mo_coeff is not None
    mf.converged = False

    with pytest.raises(
        ProviderContractError,
        match="ground-state PySCF mean-field did not converge",
    ):
        from_core_hole_delta_scf(mf, "O 1s", max_cycle=1)


def test_core_hole_delta_scf_rejects_unconverged_final_state_before_orbitals(
    monkeypatch,
):
    pytest.importorskip("pyscf")
    from pyscf import scf

    _, ground_mf = _h2o()
    final_states = []

    class UnconvergedFinalState:
        def __init__(self, mol):
            self.mol = mol
            self.conv_tol = 1e-9
            self.max_cycle = 50
            self.converged = False
            self.mo_coeff_accessed = False

        @property
        def mo_coeff(self):
            self.mo_coeff_accessed = True
            raise AssertionError("final-state mo_coeff should not be accessed")

        def make_rdm1(self, mo_coeff, occ):
            return ground_mf.make_rdm1(mo_coeff, occ)

        def kernel(self, dm0=None):
            return 0.0

    def fake_rohf(mol):
        mf = UnconvergedFinalState(mol)
        final_states.append(mf)
        return mf

    monkeypatch.setattr(scf, "ROHF", fake_rohf)

    with pytest.raises(
        ProviderContractError,
        match="core-hole final-state PySCF mean-field did not converge",
    ):
        from_core_hole_delta_scf(ground_mf, "O 1s")
    assert final_states
    assert final_states[0].mo_coeff_accessed is False


def test_uhf_rejected_fail_closed():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = gto.M(
        atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0
    )
    mf = scf.UHF(mol)
    mf.kernel()
    with pytest.raises(ProviderContractError, match="restricted references"):
        PySCFProvider(mf)


def test_mom_core_hole_occ_tracks_orbital_character_after_column_swap():
    S = np.eye(4)
    C_ground = np.eye(4)
    core_index = 1
    final_index = 3
    ref_ao = C_ground[:, core_index]
    mo_coeff = C_ground[:, [0, 2, 3, 1]]
    mo_energy = np.array([-0.5, -1.0, -0.6, -20.0])

    occ, tracked_index, max_overlap = _mom_core_hole_occ(
        mo_coeff,
        mo_energy,
        S,
        ref_ao,
        n_electrons=3,
        spin=1,
    )

    # The old fixed-index occupation would leave the hole at core_index=1 even
    # though the O 1s character moved to column 3; MOM follows the character.
    assert tracked_index == final_index
    assert tracked_index == int(np.argmax(np.abs((ref_ao @ S) @ mo_coeff)))
    assert max_overlap == pytest.approx(1.0)
    assert occ[tracked_index] == pytest.approx(1.0)
    assert occ[core_index] != pytest.approx(1.0)

    ground_occ = np.array([2.0, 2.0, 0.0, 0.0])
    core_hole = _core_hole_metadata(
        "O 1s",
        ground_occ,
        occ,
        core_index,
        final_index,
        max_overlap,
        delta_e=10.0,
    )
    payload = ElectronicStructurePayload(
        Z=8,
        provenance=Provenance(
            provider="mock",
            code_version="mock 0",
            method="ROHF",
            basis="synthetic",
            theory_level="ROHF",
            core_hole="O 1s",
            energy_zero="delta_scf_total_energy",
        ),
        orbital_energies={f"MO{i}": float(e) for i, e in enumerate(mo_energy)},
        orbitals=OrbitalCoefficients(
            coefficients=mo_coeff,
            energies=mo_energy,
            occupations=occ,
        ),
        overlap=OverlapBlock(S),
        density=DensityMatrix(np.diag(occ)),
        metadata={"core_hole": core_hole},
    )
    state = molecular_auger_initial_state(payload)
    assert core_hole["occupation_before"] == pytest.approx(2.0)
    assert payload.orbitals.occupations[final_index] == pytest.approx(1.0)
    assert core_hole["occupation_after"] == pytest.approx(1.0)
    assert core_hole["occupation_delta"] == pytest.approx(-1.0)
    assert state.orbital_label == "O 1s"
    assert state.occupation_before == pytest.approx(2.0)
    assert state.occupation_after == pytest.approx(1.0)
    assert state.occupation_delta == pytest.approx(-1.0)


def test_h2o_o_1s_core_hole_delta_scf_payload():
    mol, mf = _h2o()
    payload = from_core_hole_delta_scf(mf, "O 1s", max_cycle=80)
    core_hole = payload.metadata["core_hole"]
    assert Capability.CORE_HOLE_SPEC in payload.capabilities()
    assert payload.provenance.core_hole == "O 1s"
    assert payload.provenance.energy_zero == "delta_scf_total_energy"
    assert payload.metadata["n_electrons"] == mol.nelectron - 1
    assert core_hole["occupation_before"] == pytest.approx(2.0)
    assert core_hole["occupation_after"] == pytest.approx(1.0)
    assert core_hole["max_overlap"] >= core_hole["overlap_threshold"]
    assert payload.orbitals.occupations[core_hole["tracked_mo_index"]] == pytest.approx(
        1.0
    )
    assert payload.metadata["delta_scf_energy_hartree"] == pytest.approx(
        core_hole["initial_state_energy_hartree"]
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
