"""Cards must recover later native results without losing reference identity."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "docs/atomx_hourly_progress_2026_10_10"
sys.path.insert(0, str(HERE))
import sync_card  # noqa: E402


def harness(tmp_path, monkeypatch, reader, data):
    monkeypatch.setattr(sync_card, "REPO", tmp_path)
    (tmp_path / "comparison.json").write_text(
        json.dumps(dict(rows=[], splittings=[], computed_run="fixture"))
    )
    (tmp_path / "native.json").write_text(json.dumps(data))
    progress = dict(yb_comparison="comparison.json", tasks=[], updated_at="fixture",
                    spectra=[dict(title="native", comparison="obsolete early result",
                                  status="declared model", source_artifact="native.json",
                                  reader=reader)])
    return progress


def test_ne_card_reads_later_actual_positions_and_open_gates(tmp_path, monkeypatch):
    # Numeric regression fixture from the saved later Ne record; the tracked
    # test remains runnable without the large local scientific evidence tree.
    data = dict(transitions=dict(
        q=dict(energy_eV=908.193003809802, minus_experiment_meV=-8.896190198015574),
        r=dict(energy_eV=908.0657189178187, minus_experiment_meV=-13.881082181342208)),
        experiment=dict(q_eV=908.2019, r_eV=908.0796,
                        source="https://doi.org/10.1103/PhysRevA.110.L030802"),
        full_correlation_converged=False, physical_pole_certified=False)
    progress = harness(tmp_path, monkeypatch, "native_ne_positions", data)
    card, evidence = sync_card.render(progress, datetime.now(timezone.utc))
    text = "\n".join(row["content"] for row in card["body"]["elements"])
    assert "−实验 -8.896190 meV" in text and "−实验 -13.881082 meV" in text
    assert "908.193003810" in text and "PhysRevA.110.L030802" in text
    assert "完整相关收敛=False" in text and "物理极点认证=False" in text
    assert "obsolete early result" not in text
    assert any(row["path"] == "native.json" for row in evidence)
    # A reference update cannot retain a score from a different source row.
    data["experiment"]["q_eV"] += .001
    (tmp_path / "native.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Ne position/reference identity"):
        sync_card.render(progress, datetime.now(timezone.utc))


def test_eu_card_verifies_calibration_audit_inputs(tmp_path, monkeypatch):
    source = tmp_path / "historical.log"
    source.write_text("historical calibrated model\n")
    data = dict(inputs=dict(log=dict(path=source.name,
                                    sha256=hashlib.sha256(source.read_bytes()).hexdigest())),
                card_summary="标定 RMS 8.2 cm⁻¹；NIST 半经验条目，未知不确定度。")
    progress = harness(tmp_path, monkeypatch, "native_eu_baseline", data)
    card, evidence = sync_card.render(progress, datetime.now(timezone.utc))
    text = "\n".join(row["content"] for row in card["body"]["elements"])
    assert data["card_summary"] in text
    assert any(row["path"] == source.name for row in evidence)
    source.write_text("different model\n")
    with pytest.raises(ValueError, match="Eu baseline input identity"):
        sync_card.render(progress, datetime.now(timezone.utc))
