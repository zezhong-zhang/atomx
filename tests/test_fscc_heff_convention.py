"""Physical attachment matrices require signed complex wave evidence."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from atomx.methods import fscc

EVIDENCE = (Path(__file__).resolve().parents[1] /
            "docs/yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/"
            "paper_reproduction001/reference_tensor_compare004/actual")


def test_actual_complex_attachment_matrix_and_all_physical_waves():
    archive = EVIDENCE / "complete_phase_comparison.npz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "1bc5a0e1b9531596bd14c7533b9b79975a4b9b5b1c29a5320e36af03e9668a27")
    result = json.loads((EVIDENCE / "result.json").read_text())
    heff = EVIDENCE / "fscc/scratch/HEFF"
    assert hashlib.sha256(heff.read_bytes()).hexdigest() == (
        "7b54e29f72e5ed25b4158e1aea510fe5a2ad61d08b713c4a14f7258463c69998")
    backend = {"source_sha256": dict(fscc._ATTACHMENT_HEFF_SOURCE_SHA256)}
    with np.load(archive) as a:
        blocks = fscc.read_heff(heff)
        raw = fscc._model_matrix(blocks, "0h1p", (3, 4, 5, 6), None)
        np.testing.assert_array_equal(raw, a["external_raw_Heff"])
        physical = fscc._attachment_ket_matrix(raw, backend)
        omega = a["external_model_omega"]
        model = a["model_rows"]
        hbar = a["external_attachment_hbar"] - result["independent_core_energy_Ha"] * np.eye(35)
        # Independent directed wave projection; no Hermitian averaging.
        expected_from_wave = (hbar @ omega)[model]
        np.testing.assert_allclose(physical, expected_from_wave, atol=1e-10, rtol=0)
        np.testing.assert_allclose(physical, a["expected_ket_Heff"], atol=1e-10, rtol=0)
        for wrong in (raw, raw.T, raw.conj()):
            assert np.max(abs(wrong - expected_from_wave)) > 1e-6
        assert np.max(abs((hbar @ omega - omega @ physical)[a["retained_R_rows"]])) < 1e-10
        # The omitted rank4 defect must survive the convention repair.
        assert np.max(abs(hbar @ omega - omega @ physical)) > 1e-6
        values, vectors = np.linalg.eig(physical)
        order = np.argsort(values.real)
        np.testing.assert_allclose(values[order], result["frozen_roots_Ha"], atol=1e-10, rtol=0)
        wave = a["external_attachment_T"]
        exponential = np.eye(35, dtype=complex)
        power = exponential.copy()
        for rank in range(1, 4):
            power = power @ wave / rank
            exponential += power
        for column, index in enumerate(order):
            actual = exponential @ omega @ vectors[:, index]
            actual /= np.linalg.norm(actual)
            expected = a["expected_physical_wave"][:, column]
            np.testing.assert_allclose(np.outer(actual, actual.conj()),
                                       np.outer(expected, expected.conj()), atol=1e-10, rtol=0)
        with pytest.raises(ValueError):
            physical.setflags(write=True)
        np.testing.assert_array_equal(blocks["0h1p", 1], a["external_raw_Heff"])


@pytest.mark.parametrize("changed", list(fscc._ATTACHMENT_HEFF_SOURCE_SHA256))
@pytest.mark.parametrize("kind", ["missing", "changed"])
def test_unknown_attachment_matrix_convention_rejects_before_export(
        tmp_path, monkeypatch, changed, kind):
    from test_fscc_backend_physics_capability import request

    integrals, executable = request(tmp_path, 0.)
    manifest_path = tmp_path / "atomx_backend.json"
    manifest = json.loads(manifest_path.read_text())
    if kind == "missing":
        del manifest["source_sha256"][changed]
    else:
        manifest["source_sha256"][changed] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        pytest.fail("unknown physical matrix convention reached export or execution")

    monkeypatch.setattr(fscc, "export_integrals", forbidden)
    monkeypatch.setattr(fscc.subprocess, "run", forbidden)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="qualified HEFF bra-to-ket convention"):
        fscc.compute(integrals, executable=executable, directory=output,
                     sector="0h1p", active_indices=(2, 3))
    assert not output.exists()


def test_live_public_complex_nonhermitian_attachment_convention(tmp_path):
    executable = os.environ.get("ATOMX_EXPT_EXECUTABLE")
    if not executable:
        pytest.skip("set ATOMX_EXPT_EXECUTABLE for the actual complex attachment replay")
    from atomx.physics.csf.spinor_integrals import SpinorIntegrals

    manifest = json.loads((EVIDENCE / "fscc/integrals.json").read_text())
    with np.load(EVIDENCE / "complete_phase_comparison.npz") as a:
        integrals = SpinorIntegrals(tuple(tuple(v) for v in manifest["labels"]), 3,
            a["one_body_Ha"], a["Fock_Ha"], a["packed_eri_Ha"],
            manifest["frozen_energy_Ha"], manifest["reference_energy_Ha"], manifest["identity"])
        result = fscc.compute(integrals, executable=executable, directory=tmp_path / "run",
            model="CCSDT", sector="0h1p", active_indices=(3, 4, 5, 6), tolerance=1e-13,
            residual_bound_tolerance_Ha=1e-11, memory_bytes=512 * 1024**2,
            tile_size=4, threads=1, timeout_seconds=60)
        np.testing.assert_allclose(result.model_space_matrix_Ha, a["expected_ket_Heff"],
                                   atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.effective_matrices["0h1p", 1], a["external_raw_Heff"],
                                   atol=1e-10, rtol=0)
        assert "physical ket" in result.provenance["model_space_matrix_convention"]
        assert "raw EXP-T" in result.provenance["effective_matrices_convention"]
        with np.load(tmp_path / "run/model_space_matrix.npz") as saved:
            np.testing.assert_array_equal(saved["matrix_Ha"], result.model_space_matrix_Ha)
