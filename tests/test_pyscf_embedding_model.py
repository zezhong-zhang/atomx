"""PySCF-free contract tests for the embedding model layer (HYP-909, R4).

These tests intentionally carry **no** ``pytest.importorskip("pyscf")``
guard: they run on the minimal dependency set (CI never installs PySCF) and
pin the pure-numpy layer — ``EmbeddingModel`` validation, rocksalt
geometry/Evjen neutrality, ``scaled_mm_charges``, study-grid fail-closed
validation and ``to_table`` — plus the AC2 lazy-import property: importing
:mod:`atomx.providers.pyscf_embedding` must never require PySCF.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.providers.contract import ProviderContractError  # noqa: E402
from atomx.providers.pyscf_embedding import (  # noqa: E402
    ClusterConvergenceStudy,
    CoreOrbitalIdentity,
    EmbeddingModel,
    EmbeddingRecord,
    rocksalt_embedding,
    study_cluster_convergence,
)

MGO_NN_ANGSTROM = 2.106  # rock-salt MgO nearest-neighbor distance


def _mgo_model(quantum_shells=0, mm_shells=1, evjen=True):
    return rocksalt_embedding(
        "Mg",
        "O",
        2.0,
        -2.0,
        MGO_NN_ANGSTROM,
        quantum_shells=quantum_shells,
        mm_shells=mm_shells,
        evjen=evjen,
    )


def test_lazy_import_binds_no_pyscf():
    # AC2 pin: the module must not bind a top-level pyscf reference; on the
    # minimal dependency set (CI) the import at collection time is itself
    # the proof that no top-level `import pyscf` exists.
    import atomx.providers.pyscf_embedding as mod

    assert not any(
        name == "pyscf" or name.startswith("pyscf.") for name in vars(mod)
    )


def test_rocksalt_embedding_structure_and_evjen_neutrality():
    model = _mgo_model(quantum_shells=0, mm_shells=1, evjen=True)
    assert model.central_atom_index == 0
    assert model.quantum_atoms.startswith("Mg 0.0000000000 0.0000000000 0.0000000000")
    assert model.n_quantum_atoms == 1
    assert model.n_mm_charges == 26  # full first Chebyshev shell around Mg
    # Evjen fractional boundary charges: 6 faces (1/2) + 12 edges (1/4)
    # + 8 corners (1/8) -> the truncated lattice is neutral overall, so the
    # MM region carries exactly -q_cation to balance the quantum cube.
    assert model.net_mm_charge == pytest.approx(-2.0, abs=1e-12)
    # Nearest neighbours of the central cation are the 6 face anions.
    distances = np.linalg.norm(model.mm_coords, axis=1)
    nearest = np.sort(distances)[:6]
    np.testing.assert_allclose(nearest, MGO_NN_ANGSTROM, atol=1e-12)
    np.testing.assert_allclose(model.mm_charges[np.argsort(distances)[:6]], -1.0)


def test_rocksalt_embedding_without_evjen_records_net_charge():
    model = _mgo_model(quantum_shells=0, mm_shells=1, evjen=False)
    assert model.n_mm_charges == 26
    # The full 27-site cube carries one extra anion-parity site; removing the
    # central cation leaves the MM region at net -2 * q_cation, recorded in
    # the model metadata rather than silently renormalized.
    assert model.net_mm_charge == pytest.approx(-4.0, abs=1e-12)
    assert "evjen-boundary-charges" not in model.approximations


def test_rocksalt_qm1_net_mm_balances_quantum_formal_charge():
    # With Evjen boundary charges the truncated lattice is exactly neutral:
    # the 27-site quantum cube is Mg13O14 (formal charge -q_cation), so the
    # MM region carries +q_cation. Net MM = -(quantum-region formal charge).
    model = _mgo_model(quantum_shells=1, mm_shells=2, evjen=True)
    assert model.n_quantum_atoms == 27
    assert model.quantum_atoms.count("Mg") == 13
    assert model.quantum_atoms.count("O") == 14
    assert model.net_mm_charge == pytest.approx(2.0, abs=1e-12)


def test_embedding_model_validation_fail_closed():
    model = _mgo_model()
    with pytest.raises(ProviderContractError, match="mm_coords"):
        EmbeddingModel(
            quantum_atoms=model.quantum_atoms,
            central_atom_index=0,
            core_shell="1s",
            mm_coords=np.zeros((3, 2)),
            mm_charges=np.zeros(3),
        )
    with pytest.raises(ProviderContractError, match="mm_charges"):
        EmbeddingModel(
            quantum_atoms=model.quantum_atoms,
            central_atom_index=0,
            core_shell="1s",
            mm_coords=np.zeros((3, 3)),
            mm_charges=np.zeros(2),
        )
    with pytest.raises(ProviderContractError, match="finite"):
        EmbeddingModel(
            quantum_atoms=model.quantum_atoms,
            central_atom_index=0,
            core_shell="1s",
            mm_coords=np.array([[np.inf, 0.0, 0.0]]),
            mm_charges=np.array([1.0]),
        )
    with pytest.raises(ProviderContractError, match="central_atom_index"):
        EmbeddingModel(
            quantum_atoms=model.quantum_atoms,
            central_atom_index=7,
            core_shell="1s",
            mm_coords=np.zeros((0, 3)),
            mm_charges=np.zeros(0),
        )
    with pytest.raises(ProviderContractError, match="quantum_shells"):
        rocksalt_embedding("Mg", "O", 2.0, -2.0, MGO_NN_ANGSTROM,
                           quantum_shells=1, mm_shells=1)


def test_scaled_mm_charges_records_scale():
    model = _mgo_model()
    scaled = model.scaled_mm_charges(0.5)
    np.testing.assert_allclose(scaled.mm_charges, model.mm_charges * 0.5)
    assert scaled.net_mm_charge == pytest.approx(model.net_mm_charge * 0.5)
    assert "scale=0.5" in scaled.label
    assert "mm-charge-scale=0.5" in scaled.approximations
    # The base model is untouched (frozen dataclass semantics).
    assert "mm-charge-scale=0.5" not in model.approximations
    with pytest.raises(ProviderContractError, match="scale"):
        model.scaled_mm_charges(-0.5)
    with pytest.raises(ProviderContractError, match="scale"):
        model.scaled_mm_charges(np.nan)


def test_convergence_study_fail_closed_on_empty_grid():
    # Grid validation runs before any PySCF import, so this fail-closed
    # contract holds on the minimal dependency set too.
    with pytest.raises(ProviderContractError, match="no configurations"):
        study_cluster_convergence(
            cation="Mg",
            anion="O",
            q_cation=2.0,
            q_anion=-2.0,
            nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
            basis="sto-3g",
            quantum_shells=(2,),
            mm_shells=(1,),
        )
    with pytest.raises(ProviderContractError, match="mm_shells"):
        study_cluster_convergence(
            cation="Mg",
            anion="O",
            q_cation=2.0,
            q_anion=-2.0,
            nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
            basis="sto-3g",
            quantum_shells=(0,),
            mm_shells=(0,),
        )


def _synthetic_identity(energy=-48.0):
    return CoreOrbitalIdentity(
        label="Mg 1s",
        environment="synthetic",
        mo_index=0,
        energy_hartree=energy,
        occupation=2.0,
        dominance=0.99,
        runner_up=0.01,
    )


def test_study_to_table_roundtrip():
    identity = _synthetic_identity()
    record = EmbeddingRecord(
        label="MgO:qm0/mm1/evjen",
        quantum_shells=0,
        mm_shells=1,
        evjen=True,
        charge_scale=1.0,
        n_quantum_atoms=1,
        n_mm_charges=26,
        net_mm_charge=-2.0,
        identity=identity,
        total_energy_hartree=-197.0,
        binding_energy_hartree=47.0,
        orbital_shift_hartree=0.7,
        binding_shift_hartree=-0.7,
    )
    study = ClusterConvergenceStudy(
        lattice="rocksalt-MgO",
        basis="sto-3g",
        core_shell="1s",
        nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
        reference_identity=_synthetic_identity(-48.7),
        reference_binding_hartree=47.7,
        records=(record,),
    )
    table = study.to_table()
    assert len(table) == 1
    row = table[0]
    assert {
        "label",
        "quantum_shells",
        "mm_shells",
        "evjen",
        "charge_scale",
        "n_quantum_atoms",
        "n_mm_charges",
        "net_mm_charge",
        "core_mo_index",
        "core_dominance",
        "core_epsilon_hartree",
        "binding_energy_hartree",
        "orbital_shift_hartree",
        "binding_shift_hartree",
    } == set(row)
    assert row["label"] == "MgO:qm0/mm1/evjen"
    assert row["core_dominance"] == pytest.approx(0.99)
    assert row["orbital_shift_hartree"] == pytest.approx(0.7)


def test_identity_and_record_reject_bad_energy():
    with pytest.raises(Exception, match="Hartree|finite"):
        _synthetic_identity(energy=np.nan)
    with pytest.raises(ProviderContractError, match="non-empty"):
        CoreOrbitalIdentity(
            label="",
            environment="synthetic",
            mo_index=0,
            energy_hartree=-48.0,
            occupation=2.0,
            dominance=0.99,
            runner_up=0.01,
        )
