"""Local full-equation Fock-Jacobi/DIIS experiment; no production API change."""

import numpy as np

from atomx.methods.tensor_coupled_cluster import TensorCoreResult


def solve(op, initial=None, *, scalar_sector=True, max_calls=60):
    active = op.scalar_core_indices() if scalar_sector else np.arange(op.core.size)
    vector = np.zeros(op.core.size) if initial is None else np.array(initial, copy=True)
    outside = np.setdiff1d(np.arange(op.core.size), active)
    if np.any(vector[outside] != 0):
        raise ValueError("initial leaves declared exact core sector")
    eps = np.diag(op.f)
    d = np.concatenate(
        [
            np.sum(eps[op.o + p], axis=1) - np.sum(eps[h], axis=1)
            for _, p, h, _ in op.core.blocks
        ]
    )[active]
    if np.any(d == 0) or not np.all(np.isfinite(d)):
        raise ValueError("nonzero finite unshifted Fock denominators required")
    candidates, errors, history = [], [], []
    for count in range(1, max_calls + 1):
        residual = op.core_residual(vector)
        maximum = float(np.max(abs(residual)))
        if not np.isfinite(maximum):
            raise FloatingPointError("nonfinite complete SDT residual")
        row = dict(evaluation=count, maximum_Ha=maximum)
        history.append(row)
        if maximum <= 1e-10:
            energy = op.core_energy(vector)
            return TensorCoreResult(
                vector,
                energy,
                energy - op.reference_energy_Ha,
                maximum,
                count,
                op.operator_sha256,
            ), history
        if len(outside) and np.max(abs(residual[outside])) > 1e-10:
            raise ValueError("original equations leave declared exact core sector")
        error = residual[active] / d
        candidates.append(vector[active] - error)
        errors.append(error.copy())
        if len(errors) > 6:
            candidates.pop(0)
            errors.pop(0)
        if len(errors) == 1:
            next_point = candidates[0]
        else:
            e = np.column_stack(errors)
            gram = e.T @ e
            # One common scale leaves the constrained minimizer unchanged.
            gram /= np.max(np.diag(gram))
            n = len(errors)
            pencil = np.block(
                [[gram, -np.ones((n, 1))], [-np.ones((1, n)), np.zeros((1, 1))]]
            )
            rhs = np.r_[np.zeros(n), -1.0]
            coefficients = np.linalg.solve(pencil, rhs)[:n]
            row.update(
                history_size=n,
                Gram_condition=float(np.linalg.cond(gram)),
                coefficients=coefficients.tolist(),
            )
            if not np.all(np.isfinite(coefficients)):
                raise FloatingPointError("nonfinite DIIS coefficients")
            next_point = np.column_stack(candidates) @ coefficients
        vector = np.zeros(op.core.size)
        vector[active] = next_point
    raise RuntimeError("predeclared60 complete residual call cap")
