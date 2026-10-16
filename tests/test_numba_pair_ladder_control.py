"""Complete CPU IEEE control gates; CUDA is explicitly optional."""

import sys
from pathlib import Path

import numba as nb
import numpy as np
import pytest

OWNER = Path(__file__).resolve().parents[1] / "docs/atomic_pci_sd_native_2026_10_09"
sys.path[:0] = [
    str(OWNER),
    str(OWNER / "parallel_six_gpu001"),
    str(OWNER / "parallel_six_sd_diagrams001"),
]
from pair_ladder_numba import NumbaPairLadderControl
from payload import load_hamiltonian

from atomx.methods.sd_pair_diagrams import CorePairLadders
from atomx.methods.sd_transverse import build_j_coupled_sd_interaction


@pytest.mark.parametrize("mode", ["dc", "static", "dynamic"])
def test_complete_all_directions_and_scoped_worker_control(mode):
    data = np.load(OWNER / "parallel_six_sd_diagrams001/inputs.npz")
    ladders = CorePairLadders(
        build_j_coupled_sd_interaction(load_hamiltonian(data, mode))
    )
    x = np.ascontiguousarray(
        np.column_stack(
            (
                data["amplitudes_basis"],
                data["amplitudes_signed"].real,
                data["amplitudes_signed"].imag,
            )
        )
    )
    control = NumbaPairLadderControl(
        ladders, backend="cpu", batch=x.shape[1], memory_bytes=32 * 1024**2
    )
    p, h = np.empty_like(x), np.empty_like(x)
    original_workers = nb.get_num_threads()
    control.apply_into(x, p, h)
    assert nb.get_num_threads() == original_workers
    for b in range(x.shape[1]):
        expected = ladders.evaluate(x[:, b])
        np.testing.assert_allclose(
            p[:, b], expected.particle_particle_Ha, atol=1e-11, rtol=0
        )
        np.testing.assert_allclose(h[:, b], expected.hole_hole_Ha, atol=1e-11, rtol=0)
    with pytest.raises(MemoryError):
        NumbaPairLadderControl(
            ladders, backend="cuda", batch=x.shape[1], memory_bytes=1
        )
    with pytest.raises(ValueError):
        control.apply_into(x.astype(np.complex128), p, h)
    with pytest.raises(ValueError):
        control.apply_into(x[:, :-1], p, h)
    with pytest.raises(ValueError):
        control.apply_into(x, p, p)
    with pytest.raises(ValueError):
        control.apply_resident()


def test_complex_physics_input_rejects_before_real_cast():
    data = np.load(OWNER / "parallel_six_sd_diagrams001/inputs.npz")
    ladders = CorePairLadders(
        build_j_coupled_sd_interaction(load_hamiltonian(data, "complex_phase"))
    )
    with pytest.raises(ValueError, match="complex interaction"):
        NumbaPairLadderControl(
            ladders, backend="cpu", batch=10, memory_bytes=32 * 1024**2
        )
