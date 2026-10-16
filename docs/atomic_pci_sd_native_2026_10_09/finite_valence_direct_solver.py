"""Exact finite-sector LU anchor, retaining complete signed output tensors.

Only strict total-M/parity zeros are excluded from the numerical factorization.
This explicitly bounded diagnostic solver is never a production dense engine.
"""

import itertools

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator

from atomx.angular.recoupling import l_of_kappa


def magnetic_solver(labels, nc, column, energy, oracle, *, linear=None):
    shape = (len(labels), len(labels), nc)
    coordinates = []
    for m, n, b in itertools.product(
        range(nc, len(labels)), range(nc, len(labels)), range(nc)
    ):
        if (
            labels[m][2] + labels[n][2] - labels[b][2] == labels[column][2]
            and (
                l_of_kappa(labels[m][1])
                + l_of_kappa(labels[n][1])
                - l_of_kappa(labels[b][1])
                - l_of_kappa(labels[column][1])
            )
            % 2
            == 0
        ):
            coordinates.append((m, n, b))
    index = tuple(np.asarray(coordinates).T)
    allowed = np.zeros(shape, dtype=bool)
    allowed[index] = True
    constant = oracle(energy, np.zeros(shape), column)[1]
    if np.any(constant[~allowed]):
        raise AssertionError("source violates exact magnetic-sector contract")

    def unpack(vector):
        full = np.zeros(shape, dtype=constant.dtype)
        full[index] = vector
        return full

    def action(vector):
        full = unpack(vector)
        homogeneous = (
            oracle(energy, full, column)[1] - constant
            if linear is None else linear(energy, full, column)[1]
        )
        applied = full - homogeneous
        if np.any(applied[~allowed]):
            raise AssertionError("operator mixes exact magnetic sectors")
        return applied[index]

    count = len(coordinates)
    if constant.itemsize * count**2 * 5 > 64 * 1024**2:
        raise MemoryError("complete finite-sector direct anchor exceeds cap")
    matrix = np.column_stack([action(np.eye(1, count, i)[0]) for i in range(count)])
    lu = lu_factor(np.asarray(matrix, dtype=np.float64))

    def inverse(rhs):
        rhs = np.asarray(rhs).reshape(shape)
        if not np.all(np.isfinite(rhs)) or np.any(rhs[~allowed]):
            raise ValueError("complete finite same-sector RHS required")
        solution = np.asarray(
            lu_solve(lu, np.asarray(rhs[index], dtype=np.float64)),
            dtype=constant.dtype,
        )
        # Iterative refinement uses the actual operator, not just stored K.
        for _ in range(3):
            residual = rhs[index] - action(solution)
            solution += lu_solve(lu, np.asarray(residual, dtype=np.float64))
        return unpack(solution)

    def full_action(vector):
        full = np.asarray(vector).reshape(shape)
        homogeneous = (
            oracle(energy, full, column)[1] - constant
            if linear is None else linear(energy, full, column)[1]
        )
        return (full - homogeneous).reshape(-1)

    operator = LinearOperator((constant.size,) * 2, matvec=full_action, dtype=float)
    solved = inverse(constant)
    residual = solved - oracle(energy, solved, column)[1]
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    info = dict(
        full_output_coordinates=int(constant.size),
        complete_sector_coordinates=count,
        excluded="strict total-M/parity and occupied padding zeros only; full tensor preserved",
        smallest_singular_value=float(singular[-1]),
        condition_2=float(singular[0] / singular[-1]),
    )
    return solved, residual, operator, inverse, info


def reduced_solver(energy, count, evaluate, *, linear=None):
    constant = evaluate(energy, np.zeros(count))[1]

    def action(vector):
        homogeneous = (
            evaluate(energy, vector)[1] - constant
            if linear is None else linear(energy, vector)[1]
        )
        return vector - homogeneous

    if 8 * count**2 * 5 > 64 * 1024**2:
        raise MemoryError("complete reduced diagnostic matrix exceeds cap")
    matrix = np.column_stack([action(np.eye(1, count, i)[0]) for i in range(count)])
    lu = lu_factor(matrix)

    def inverse(rhs):
        solution = lu_solve(lu, rhs)
        for _ in range(3):
            solution += lu_solve(lu, rhs - action(solution))
        return solution

    solved = inverse(constant)
    op = LinearOperator(matrix.shape, matvec=action, dtype=float)
    return solved, op, solved - evaluate(energy, solved)[1], inverse, matrix
