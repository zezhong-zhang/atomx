from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from fac_runner import parse_debug_hamiltonian_log, parse_hamiltonian_binary  # noqa: E402


def test_parse_hamiltonian_binary_expands_lower_triangle(tmp_path):
    ham = tmp_path / "mini.ham.b"
    payload = bytearray()

    def put_i(value: int):
        payload.extend(struct.pack("i", value))

    def put_d(value: float):
        payload.extend(struct.pack("d", value))

    put_i(2)       # ng0
    put_i(1)       # ng
    put_i(7)       # kg
    put_i(0)       # ngp
    put_i(5)       # symmetry
    put_i(2)       # dim
    put_i(2)       # orig_dim
    put_i(2)       # n_basis
    put_i(3)       # basis[0]
    put_i(4)       # basis[1]
    put_i(3)       # n_values
    put_i(0); put_i(0); put_d(-10.0)
    put_i(0); put_i(1); put_d(0.25)
    put_i(1); put_i(1); put_d(-9.5)
    put_i(9)       # empty symmetry block
    put_i(0)
    ham.write_bytes(payload)

    parsed = parse_hamiltonian_binary(ham)

    assert parsed["ng0"] == 2
    assert parsed["kg"] == [7]
    assert len(parsed["blocks"]) == 1
    block = parsed["blocks"][0]
    assert block["symmetry"] == 5
    assert block["basis"] == [3, 4]
    np.testing.assert_allclose(block["hamiltonian"], [[-10.0, 0.25], [0.25, -9.5]])



def test_parse_debug_hamiltonian_log_groups_hpair_sections(tmp_path):
    log = tmp_path / "debug.log"
    log.write_text(
        "noise before\n"
        "HPAIR 0 0\n"
        "3 2 -2 4 4\n"
        "3 2 -2 4 4\n"
        "1 electron: -10.0\n"
        "3 2 -2 4 4\n"
        "3 2 -2 4 4\n"
        "2 2 1 2 2\n"
        "2 2 1 2 2\n"
        "2 electron: 1.5\n"
        "HPAIR 0 1\n"
        "5 3 -3 1 0\n"
        "4 3 2 0 1\n"
        "3 2 -2 3 3\n"
        "3 2 -2 3 3\n"
        "2 electron: -0.25\n"
        "HPAIR 1 1\n"
        "4 3 2 1 1\n"
        "4 3 2 1 1\n"
        "1 electron: -9.0\n"
        "HPAIR 0 0\n"
        "1 electron: -2.0\n"
    )

    groups = parse_debug_hamiltonian_log(log)

    assert len(groups) == 2
    np.testing.assert_allclose(groups[0]["hamiltonian"], [[-8.5, -0.25], [-0.25, -9.0]])
    np.testing.assert_allclose(groups[1]["hamiltonian"], [[-2.0]])
    first_detail = groups[0]["sections"][0]["details"][1]
    assert first_detail["kind"] == "two"
    assert first_detail["shells"] == (
        (3, 2, -2, 4, 4),
        (3, 2, -2, 4, 4),
        (2, 2, 1, 2, 2),
        (2, 2, 1, 2, 2),
    )
