"""MCDHF cards expose original stationarity failures and separate references."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "docs/atomx_hourly_progress_2026_10_10"
sys.path.insert(0, str(HERE))
import sync_card  # noqa: E402


@pytest.fixture
def record(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_card, "REPO", tmp_path)
    source = tmp_path / "actual_run.json"
    source.write_text(json.dumps({"converged": False}))
    rows = []
    for i, (label, j, parity) in enumerate((
        ("6s_1/2", 1, 1), ("5d_3/2", 3, 1), ("5d_5/2", 5, 1),
        ("6p_1/2", 1, -1), ("6p_3/2", 3, -1),
    )):
        reference, error = float(10000 * i), float(500 * i)
        rows.append(dict(label=label, twoJ=j, parity=parity,
                         computed_cm1=reference + error, experimental_cm1=reference,
                         error_cm1=error, relative_error=.05 if i else None,
                         gate_pass=False if i else None))
    data = dict(method="MCDHF -> RCI", fitted_corrections=False,
                highest_level_reached=False, production_certified=False,
                computed_run="native-five-state", CSFs=5, radial_orbitals=24,
                electrons=69, rows=rows, experimental_gate_passed=False,
                card_summary="裸参考缺少核价动态关联；不能冒充空间收敛。",
                stationarity=dict(converged=False, orbital_residual=5e-6,
                                  orbital_tolerance=1e-7, estimated_energy_error=3e-11,
                                  energy_tolerance=1e-8),
                inputs=dict(run=dict(path=source.name,
                                     sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
    return data, source


def test_failed_stationarity_stays_next_to_the_actual_diagnostic_rows(record):
    data, _ = record
    citations = []
    text = sync_card.mcdhf_summary(data, citations)
    assert "驻值未通过，能级仅用于诊断" in text
    assert "5.000e-06 / 门 1.0e-07" in text
    assert "裸参考缺少核价动态关联" in text
    assert "5d_3/2：10500.00 / 10000.00" in text
    assert citations == [data["inputs"]["run"]]


def test_failed_original_gate_cannot_be_promoted_by_flipping_status(record):
    data, _ = record
    data["stationarity"]["converged"] = True
    with pytest.raises(ValueError, match="stationarity certification"):
        sync_card.mcdhf_summary(data, [])


@pytest.mark.parametrize("field", ["error_cm1", "experimental_cm1", "gate_pass"])
def test_changed_reference_arithmetic_or_experiment_gate_rejects(record, field):
    data, _ = record
    data["rows"][1][field] = True if field == "gate_pass" else 123.
    with pytest.raises(ValueError, match="arithmetic|experimental gate"):
        sync_card.mcdhf_summary(data, [])


def test_changed_bound_run_rejects_old_measurement(record):
    data, source = record
    source.write_text("changed physical input")
    with pytest.raises(ValueError, match="input identity changed"):
        sync_card.mcdhf_summary(data, [])
