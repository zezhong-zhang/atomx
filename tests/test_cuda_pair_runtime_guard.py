"""Fixed CUDA dependencies reject changed bytes before optional imports."""

import importlib.util
import json
from pathlib import Path

import pytest

OWNER = (
    Path(__file__).resolve().parents[1]
    / "docs/atomic_pci_sd_native_2026_10_09/parallel_six_gpu001"
)
spec = importlib.util.spec_from_file_location(
    "pair_runtime_guard", OWNER / "runtime_guard.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_runtime_dependency_size_and_same_size_change_are_rejected(tmp_path):
    library = tmp_path / "declared.so"
    library.write_bytes(b"fixed CUDA bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dependencies": {
                    str(library): {
                        "bytes": library.stat().st_size,
                        "sha256": guard.digest(library),
                    }
                }
            }
        )
    )
    assert guard.verify(manifest)["matched"]
    library.write_bytes(b"changed CUDA xxx")
    with pytest.raises(RuntimeError, match="runtime dependency changed"):
        guard.verify(manifest)
    library.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="runtime dependency changed"):
        guard.verify(manifest)
