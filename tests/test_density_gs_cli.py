"""The density CLI must solve the model whose occupations it exports."""

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.mark.parametrize("averaging", ["hamiltonian", "statistical"])
def test_open_shell_cli_exports_solved_density(tmp_path, averaging):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(name, None)
    # Switching model must rebuild an existing output even without --force.
    with h5py.File(tmp_path / "C.h5", "w") as f:
        meta = f.create_group("metadata").attrs
        meta["averaging"] = "statistical" if averaging == "hamiltonian" else "ci"
        meta["correlation"] = "none"
        meta["engine"] = "bspline"
    # No averaging flag for Hamiltonian averaging: gate the default model.
    options = [] if averaging == "hamiltonian" else ["--averaging", averaging]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/density_gs.py"), "6",
         "--out", str(tmp_path), "--no-compare", *options],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with h5py.File(tmp_path / "C.h5") as f:
        meta = f["metadata"].attrs
        assert meta["engine"] == "bspline"
        assert meta["averaging"] == averaging
        assert meta["breit"] and meta["qed"]
        groups = [f[k] for k in f if k != "metadata"]
        occupations = {g.attrs["orbital"]: g.attrs["nq"] for g in groups}
        if averaging == "statistical":
            assert sorted(occupations.values()) == pytest.approx([2/3, 4/3, 2, 2])
        else:
            assert occupations['2p1/2'] == pytest.approx(4/3, abs=.01)
            assert occupations['2p3/2'] == pytest.approx(2/3, abs=.01)
        count = sum(np.trapezoid(g["density"][:], g["r"][:]) for g in groups)
        assert count == pytest.approx(6, abs=2e-3)


def test_statistical_expansion_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/density_gs.py"), "6",
         "--out", str(tmp_path), "--averaging", "statistical",
         "--correlation", "s", "--no-compare"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2
    assert "--correlation requires --averaging ci" in completed.stderr
    assert not (tmp_path / "C.h5").exists()
