"""Unique-pair occupation signs and retained versus omitted CCSDT ranks."""

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

OWNER = (
    Path(__file__).resolve().parents[1]
    / "docs/atomic_pci_sd_native_2026_10_09/parallel_six_fscc_scope001"
)


def occupation_action(occupied, operations):
    """Sorted occupation-list action, independent of the bit oracle."""
    state, phase = list(occupied), 1
    for bit, create in reversed(operations):
        if create:
            if bit in state:
                return None, 0
            position = sum(x < bit for x in state)
            state.insert(position, bit)
        else:
            if bit not in state:
                return None, 0
            position = state.index(bit)
            state.pop(position)
        phase *= (-1) ** position
    return tuple(state), phase


def unique_cluster(tensors, masks, occupied, attachment=False):
    n = tensors[0].shape[0]
    states = [tuple(i for i in range(n) if int(mask) & (1 << i)) for mask in masks]
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), complex)
    for rank, tensor in enumerate(tensors, 1):
        core_holes = itertools.combinations(range(occupied), rank - int(attachment))
        incoming = (
            ((v, *holes) for holes in core_holes for v in range(occupied, n))
            if attachment
            else core_holes
        )
        for holes in incoming:
            for particles in itertools.combinations(range(occupied, n), rank):
                value = tensor[(*holes, *particles)].conjugate()
                if value == 0:
                    continue
                operations = tuple((p, True) for p in particles) + tuple(
                    (h, False) for h in reversed(holes)
                )
                for column, state in enumerate(states):
                    final, sign = occupation_action(state, operations)
                    if sign:
                        matrix[index[final], column] += sign * value
    return matrix


def polynomial_exp(t, rank=4):
    value, power = np.eye(len(t), dtype=complex), np.eye(len(t), dtype=complex)
    for order in range(1, rank + 1):
        power = power @ t
        value += power / math.factorial(order)
    assert np.max(abs(power @ t)) < 1e-14
    return value


def evidence():
    directory = OWNER / "core4_n12_002"
    result = json.loads((directory / "result.json").read_text())
    path = directory / "full_arrays.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["arrays_sha256"]
    return np.load(path), result


def test_full_signed_unique_pair_cluster_and_attachment_lift():
    arrays, _ = evidence()
    t = [arrays[name] for name in ("t1c", "t2c", "t3c")]
    core = unique_cluster(t, arrays["core_masks"], 4)
    attached = unique_cluster(t, arrays["attachment_masks"], 4)
    np.testing.assert_allclose(core, arrays["core_cluster"], atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        attached, arrays["attachment_cluster"], atol=1e-14, rtol=0
    )
    s = unique_cluster(
        [arrays[name] for name in ("s1c", "s2c", "s3c")],
        arrays["attachment_masks"],
        4,
        attachment=True,
    )
    expected = (np.eye(len(s)) + s)[:, arrays["model_indices"]]
    np.testing.assert_allclose(expected, arrays["omega"], atol=1e-14, rtol=0)


def test_retained_projection_pass_does_not_hide_true_connected_quadruples():
    arrays, result = evidence()
    exp_t = polynomial_exp(arrays["core_cluster"])
    hbar = (
        polynomial_exp(-arrays["core_cluster"])
        @ arrays["full_core_hamiltonian_Ha"]
        @ exp_t
    )
    ref = list(arrays["core_masks"]).index(15)
    residual = hbar[:, ref].copy()
    residual[ref] -= hbar[ref, ref]
    rank = arrays["core_ranks"]
    np.testing.assert_allclose(residual, arrays["core_residual_Ha"], atol=1e-13, rtol=0)
    assert np.max(abs(residual[(rank > 0) & (rank <= 3)])) <= 1e-11
    assert np.max(abs(residual[rank == 4])) > 1e-4
    assert result["core_fci_method_error_Ha"] > 1e-7
    assert result["attachment_fci_method_error_max_Ha"] > 1e-6
    assert result["fock_occupied_virtual_max_Ha"] > 0.01
    assert result["highest_level_reached"] is False
