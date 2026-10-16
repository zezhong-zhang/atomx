"""Current full-norm consumer uses an independent signed finite fermion anchor."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

OWNER = Path(__file__).resolve().parents[1] / "docs/atomic_pci_sd_native_2026_10_09"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_signed_complete_all_M_anchor():
    sys.path.insert(0, str(Path(__file__).parent))
    norm = load(OWNER / "monovalent_norm_control.py", "current_live_whole_norm")
    anchor = load(
        OWNER / "parallel_six_norm001/finite_anchor.py", "fresh_signed_norm_anchor"
    )
    result = anchor.run(norm)
    assert result["status"] == "pass"
    assert result["cases"] == 104 and result["component_checks"] == 832
    assert max(result["maxima"].values()) < 1e-11
    assert any(
        row["expected"]["core_singles_valence_doubles_interference"] != 0
        for row in result["rows"]
    )
    signs = {row["sign"] for row in result["rows"]}
    assert signs == {-1, 1}
    assert any(row["key"][1] == -5 for row in result["rows"])


def test_anchor_detects_lost_signed_interference():
    norm = load(OWNER / "monovalent_norm_control.py", "negative_control_live_norm")
    anchor = load(
        OWNER / "parallel_six_norm001/finite_anchor.py", "signed_negative_control"
    )

    def lose_phase(*args, **kwargs):
        result = norm.norm_components(*args, **kwargs)
        result["core_singles_valence_doubles_interference"] = 0.0
        return result

    defective = SimpleNamespace(
        projections=norm.projections, norm_components=lose_phase
    )
    with pytest.raises(AssertionError, match="signed finite anchor failed"):
        anchor.run(defective)
