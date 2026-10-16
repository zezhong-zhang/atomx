"""The public optional builder must reproduce the actually validated C source."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "noncanonical_patch", REPO / "tools/expt_noncanonical_patch.py"
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def test_public_patch_matches_live_candidate_source_and_preserves_reference(tmp_path):
    owner = REPO / "docs/atomic_pci_sd_native_2026_10_09"
    relative = "src/rcc/models/sector01.c"
    reference = owner / "expt_compact_public_source001" / relative
    candidate = owner / "parallel_fscc012/candidate001/source" / relative
    original = reference.read_bytes()
    copied = tmp_path / relative
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original)
    diff = patch.patch_noncanonical_attachment(tmp_path)
    assert copied.read_bytes() == candidate.read_bytes()
    assert reference.read_bytes() == original
    assert '-    perm("r2", "(34)");' in diff
    assert 'mult("vh", "r1", "r2", 1);' in copied.read_text()
    with pytest.raises(ValueError, match="diagram differs"):
        patch.patch_noncanonical_attachment(tmp_path)


def test_changed_or_duplicate_diagram_contract_rejects_without_writing(tmp_path):
    path = tmp_path / "src/rcc/models/sector01.c"
    path.parent.mkdir(parents=True)
    for value in (patch.OLD.replace('"vh"', '"ph"'), patch.OLD + patch.OLD):
        path.write_text(value)
        with pytest.raises(ValueError, match="diagram differs"):
            patch.patch_noncanonical_attachment(tmp_path)
        assert path.read_text() == value
