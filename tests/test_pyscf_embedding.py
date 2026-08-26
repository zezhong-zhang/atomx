"""PySCF cluster/embedding + local-orbital bridge tests (HYP-909, stage G2).

Skipped when PySCF is not installed (optional ``atomx[pyscf]`` extra).
Covers the acceptance surface: embedded-cluster SCF with a traceable
point-charge environment, core-orbital identity bridging across
isolated-atom / molecule / embedded-cluster environments, declared energy
zeros (orbital energy and ΔSCF binding energy), cluster-size convergence
and embedding-charge sensitivity, and fail-closed contract violations.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "pyscf",
    reason="PySCF is an optional dependency (atomx[pyscf]); the whole module "
    "is skipped on the minimal dependency set",
)

from atomx.providers.contract import (  # noqa: E402
    Capability,
    ProviderContractError,
)
from atomx.providers.pyscf_embedding import (  # noqa: E402
    CORE_IDENTITY_DOMINANCE_THRESHOLD,
    EmbeddingModel,
    bridge_core_orbitals,
    build_embedded_scf,
    core_orbital_identity,
    embedded_core_hole_delta_scf,
    rocksalt_embedding,
    study_cluster_convergence,
)

HARTREE_TO_EV = 27.211386245988
MGO_NN_ANGSTROM = 2.106  # rock-salt MgO nearest-neighbor distance


def _isolated_mg(basis="sto-3g"):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = gto.M(atom="Mg 0 0 0", basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _mgo_model(quantum_shells=0, mm_shells=1, evjen=True):
    pytest.importorskip("pyscf")
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


# ---------------------------------------------------------------------------
# Embedding model construction & validation
# ---------------------------------------------------------------------------


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
    with pytest.raises(ProviderContractError, match="scale"):
        model.scaled_mm_charges(-0.5)


def test_odd_electron_quantum_region_fail_closed():
    # LiF qm=1 cube has 14 Li + 13 F = 159 electrons: open shell, restricted
    # embedding must fail closed rather than silently average spin channels.
    model = rocksalt_embedding(
        "Li", "F", 1.0, -1.0, 2.014, quantum_shells=1, mm_shells=2
    )
    with pytest.raises(ProviderContractError, match="closed-shell"):
        model.build_mol("sto-3g")


# ---------------------------------------------------------------------------
# Embedded SCF and the local-orbital bridge
# ---------------------------------------------------------------------------


def test_embedded_scf_converges_and_stamps_metadata():
    model = _mgo_model()
    mf = build_embedded_scf(model, "sto-3g")
    assert mf.converged is True
    meta = model.metadata()
    assert meta["embedding_n_mm_charges"] == 26
    assert meta["embedding_coord_unit"] == "angstrom"


def test_core_orbital_identity_tracks_mg_1s():
    _, iso_mf = _isolated_mg()
    identity = core_orbital_identity(iso_mf, 0, "1s", environment="isolated-atom")
    assert identity.label == "Mg 1s"
    assert identity.occupation == pytest.approx(2.0)
    assert identity.dominance > CORE_IDENTITY_DOMINANCE_THRESHOLD
    assert identity.dominance > identity.runner_up
    assert identity.energy_zero == "mo_energy"


def test_core_orbital_identity_fail_closed():
    _, iso_mf = _isolated_mg()
    with pytest.raises(ProviderContractError, match="no AO labels"):
        core_orbital_identity(iso_mf, 0, "4f")
    with pytest.raises(ProviderContractError, match="no AO labels"):
        core_orbital_identity(iso_mf, 5, "1s")  # atom index out of range
    with pytest.raises(ProviderContractError, match="dominance"):
        core_orbital_identity(iso_mf, 0, "1s", dominance_threshold=1.5)


def test_madelung_potential_raises_cation_core_level():
    # Physics gate: the Madelung potential at the cation site is dominated by
    # the nearest anions, so the Mg 1s level (and hence its binding energy)
    # is *destabilized* relative to the isolated atom — the classic ionic-
    # crystal cation core-level shift.
    _, iso_mf = _isolated_mg()
    iso_id = core_orbital_identity(iso_mf, 0, "1s", environment="isolated-atom")
    model = _mgo_model()
    emb_mf = build_embedded_scf(model, "sto-3g")
    emb_id = core_orbital_identity(mf=emb_mf, atom_index=0, shell="1s",
                                   environment=model.label)
    bridge = bridge_core_orbitals(iso_id, emb_id)
    assert bridge.label == "Mg 1s"
    assert bridge.delta_epsilon_hartree == pytest.approx(
        emb_id.energy_hartree - iso_id.energy_hartree, abs=1e-12
    )
    shift_ev = bridge.delta_epsilon_hartree * HARTREE_TO_EV
    assert 5.0 < shift_ev < 40.0  # toward the ~24 eV infinite-lattice limit


def test_bridge_fail_closed_on_mismatched_identity():
    _, iso_mf = _isolated_mg()
    mg_1s = core_orbital_identity(iso_mf, 0, "1s")
    mg_2s = core_orbital_identity(iso_mf, 0, "2s")
    with pytest.raises(ProviderContractError, match="different core orbitals"):
        bridge_core_orbitals(mg_1s, mg_2s)
    other_zero = core_orbital_identity(iso_mf, 0, "1s")
    object.__setattr__(other_zero, "energy_zero", "other_zero")
    with pytest.raises(ProviderContractError, match="energy zeros"):
        bridge_core_orbitals(mg_1s, other_zero)


def test_three_way_environment_comparison_isolated_molecule_embedded():
    # Acceptance: the same named core orbital is comparable across the
    # isolated atom, a molecule, and the embedded cluster.
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    _, iso_mf = _isolated_mg()
    mol = gto.M(
        atom=f"Mg 0 0 0; O {MGO_NN_ANGSTROM} 0 0", basis="sto-3g", verbose=0
    )
    mol_mf = scf.RHF(mol)
    mol_mf.kernel()
    model = _mgo_model()
    emb_mf = build_embedded_scf(model, "sto-3g")

    ids = [
        core_orbital_identity(iso_mf, 0, "1s", environment="isolated-atom"),
        core_orbital_identity(mol_mf, 0, "1s", environment="molecule-MgO"),
        core_orbital_identity(emb_mf, 0, "1s", environment=model.label),
    ]
    assert {i.label for i in ids} == {"Mg 1s"}
    bridges = [bridge_core_orbitals(ids[0], ids[i]) for i in (1, 2)]
    shifts_ev = [b.delta_epsilon_hartree * HARTREE_TO_EV for b in bridges]
    assert all(np.isfinite(shifts_ev))
    # Both chemical environments shift the level away from the bare atom.
    assert all(abs(s) > 0.5 for s in shifts_ev)


# ---------------------------------------------------------------------------
# Embedded ΔSCF: traceable energy zero
# ---------------------------------------------------------------------------


def test_embedded_delta_scf_payload_contract():
    model = _mgo_model()
    payload, identity, binding = embedded_core_hole_delta_scf(model, "sto-3g")
    assert Capability.CORE_HOLE_SPEC in payload.capabilities()
    assert payload.provenance.core_hole == "Mg 1s"
    assert payload.provenance.energy_zero == "delta_scf_total_energy"
    assert "point-charge-embedding" in payload.provenance.approximations
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-7)
    assert payload.electron_count() == pytest.approx(11.0, abs=1e-6)
    assert identity.label == "Mg 1s"
    assert binding > 0.0
    meta = payload.metadata
    assert meta["embedding_label"] == model.label
    assert meta["embedding_net_mm_charge"] == pytest.approx(-2.0)
    assert meta["delta_scf_binding_energy_hartree"] == pytest.approx(binding)
    assert (
        meta["core_hole_total_energy_hartree"]
        - meta["ground_state_total_energy_hartree"]
        == pytest.approx(binding, abs=1e-10)
    )


def test_embedded_delta_scf_binding_shift_matches_madelung_sign():
    _, iso_mf = _isolated_mg()
    from atomx.providers.pyscf_adapter import build_core_hole_delta_scf_payload

    iso_id = core_orbital_identity(iso_mf, 0, "1s")
    ref = build_core_hole_delta_scf_payload(iso_mf, iso_id.mo_index)
    ref_binding = ref.metadata["delta_scf_energy_hartree"]

    model = _mgo_model()
    _, _, binding = embedded_core_hole_delta_scf(model, "sto-3g")
    shift_ev = (binding - ref_binding) * HARTREE_TO_EV
    # Cation core binding decreases in the Madelung potential.
    assert -40.0 < shift_ev < -5.0


def test_zero_charge_scale_recovers_isolated_atom():
    _, iso_mf = _isolated_mg()
    iso_id = core_orbital_identity(iso_mf, 0, "1s")
    model = _mgo_model()
    scaled = model.scaled_mm_charges(0.0)
    assert "mm-charge-scale=0" in scaled.approximations
    emb_mf = build_embedded_scf(scaled, "sto-3g")
    emb_id = core_orbital_identity(emb_mf, 0, "1s", environment=scaled.label)
    assert emb_id.energy_hartree == pytest.approx(iso_id.energy_hartree, abs=1e-8)


def test_charge_scale_sensitivity_is_linear():
    model = _mgo_model()
    _, iso_mf = _isolated_mg()
    iso_id = core_orbital_identity(iso_mf, 0, "1s")
    shifts = []
    for scale in (1.0, 2.0):
        scaled = model.scaled_mm_charges(scale)
        emb_mf = build_embedded_scf(scaled, "sto-3g")
        emb_id = core_orbital_identity(emb_mf, 0, "1s")
        shifts.append(emb_id.energy_hartree - iso_id.energy_hartree)
    # First-order electrostatics: doubling the embedding charges doubles
    # the Madelung shift.
    assert shifts[1] == pytest.approx(2.0 * shifts[0], rel=0.05)


# ---------------------------------------------------------------------------
# Convergence study (cluster size / embedding charge)
# ---------------------------------------------------------------------------


def test_convergence_study_small_grid():
    study = study_cluster_convergence(
        cation="Mg",
        anion="O",
        q_cation=2.0,
        q_anion=-2.0,
        nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
        basis="sto-3g",
        quantum_shells=(0,),
        mm_shells=(1, 2),
        charge_scales=(1.0,),
    )
    assert study.lattice == "rocksalt-MgO"
    assert study.reference_identity.label == "Mg 1s"
    assert study.reference_binding_hartree > 0.0
    assert len(study.records) == 2
    by_mm = {rec.mm_shells: rec for rec in study.records}
    assert set(by_mm) == {1, 2}
    for rec in study.records:
        assert rec.identity.label == "Mg 1s"
        assert rec.identity.dominance > CORE_IDENTITY_DOMINANCE_THRESHOLD
        assert np.isfinite(rec.binding_energy_hartree)
        assert np.isfinite(rec.net_mm_charge)
        # Madelung destabilization of the cation core level, both metrics.
        assert rec.orbital_shift_hartree > 0.0
        assert rec.binding_shift_hartree < 0.0
    # Convergence toward the infinite-lattice limit is monotonic here: the
    # mm=2 embedding moves the shift further toward the asymptote.
    assert abs(by_mm[2].binding_shift_hartree) > abs(by_mm[1].binding_shift_hartree)
    table = study.to_table()
    assert len(table) == 2
    assert {
        "label",
        "mm_shells",
        "charge_scale",
        "core_dominance",
        "orbital_shift_hartree",
        "binding_shift_hartree",
    } <= set(table[0])


def test_convergence_study_charge_sensitivity_records_scale():
    study = study_cluster_convergence(
        cation="Mg",
        anion="O",
        q_cation=2.0,
        q_anion=-2.0,
        nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
        basis="sto-3g",
        quantum_shells=(0,),
        mm_shells=(1,),
        charge_scales=(0.5, 1.0),
    )
    assert [rec.charge_scale for rec in study.records] == [0.5, 1.0]
    half, full = study.records
    assert "scale=0.5" in half.label
    assert abs(half.binding_shift_hartree) < abs(full.binding_shift_hartree)


def test_convergence_study_fail_closed_on_empty_grid():
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
