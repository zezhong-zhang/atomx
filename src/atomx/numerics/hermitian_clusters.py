"""Explicit Rayleigh--Ritz refinement of close real Hermitian CI roots.

An accurate invariant subspace can contain poorly resolved individual roots.
Remove the common energy in extended precision before re-diagonalizing only
that subspace. This improves the representation of the SAME supplied matrix;
it neither changes a physical Hamiltonian nor chooses a spectroscopic state.
"""

from numbers import Real

import numpy as np
from scipy import linalg


def refine_real_symmetric_clusters(
    matrix, energies, coefficients, *, gap, max_cluster_size=64, residual_iterations=0
):
    """Refine a complete real spectrum in predeclared adjacent-gap clusters.

    ``gap`` is an absolute Hartree resolution used to group numerical work,
    not a physical degeneracy tolerance or a state-average weight. The full
    subspace is retained, including unselected roots. Extended precision is
    required; unsupported platforms and incomplete spectra raise. The
    residual and Gram gates refer to the original double-precision matrix.
    Ordinary isolated eigenvectors are unchanged by default. Explicit
    residual_iterations=1/2 permits bounded, extended-precision spectral
    corrections of failing ISOLATED roots in the same complete matrix.
    Original double-matrix residual/Gram/order gates still decide acceptance.
    No precision fallback or tolerance change.
    """
    if (
        isinstance(gap, (bool, np.bool_))
        or not isinstance(gap, Real)
        or not np.isfinite(gap)
        or gap <= 0
    ):
        raise ValueError("cluster gap must be a finite positive real number")
    if (
        isinstance(max_cluster_size, (bool, np.bool_))
        or not isinstance(max_cluster_size, (int, np.integer))
        or max_cluster_size < 2
    ):
        raise ValueError("max_cluster_size must be an integer >= 2")
    if (
        isinstance(residual_iterations, (bool, np.bool_))
        or not isinstance(residual_iterations, (int, np.integer))
        or residual_iterations not in (0, 1, 2)
    ):
        raise ValueError("residual_iterations must be an integer 0, 1 or 2")
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        raise RuntimeError("cluster refinement requires extended precision")
    arrays = tuple(np.asarray(a) for a in (matrix, energies, coefficients))
    if any(np.iscomplexobj(a) for a in arrays):
        raise ValueError("cluster refinement requires a real symmetric spectrum")
    h, w, c = (np.array(a, dtype=float, copy=True) for a in arrays)
    if (
        h.ndim != 2
        or h.shape[0] != h.shape[1]
        or w.shape != (h.shape[0],)
        or c.shape != h.shape
    ):
        raise ValueError("cluster refinement requires a complete square eigensystem")
    if any(not np.all(np.isfinite(a)) for a in (h, w, c)):
        raise ValueError("cluster refinement requires finite arrays")
    if not np.array_equal(h, h.T):
        raise ValueError(
            "cluster refinement requires an exactly symmetric input matrix"
        )
    if np.any(np.diff(w) < 0):
        raise ValueError("cluster energies must be sorted")
    if np.max(abs(c.T @ c - np.eye(len(w))), initial=0.0) > 1e-10:
        raise ValueError("cluster input eigenvectors failed Gram gate")
    boundaries = np.r_[0, np.flatnonzero(np.diff(w) > gap) + 1, len(w)]
    clusters = [
        (int(a), int(b))
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
        if b - a > 1
    ]
    if any(b - a > max_cluster_size for a, b in clusters):
        raise ValueError("cluster exceeds declared refinement resource cap")
    records = []
    for a, b in clusters:
        v = c[:, a:b].astype(np.longdouble)
        center = np.mean(w[a:b].astype(np.longdouble))
        shifted = h.astype(np.longdouble)
        shifted.flat[:: len(w) + 1] -= center
        metric = v.T @ v
        reduced = v.T @ (shifted @ v)
        # Transpose averaging removes contraction roundoff only. h itself
        # was required exactly symmetric and is never modified.
        reduced = (reduced + reduced.T) / 2
        dw, rotation = linalg.eigh(
            np.asarray(reduced, float), np.asarray(metric, float)
        )
        refined = v @ rotation.astype(np.longdouble)
        residual = shifted @ refined - refined * dw.astype(np.longdouble)
        old = w[a:b].copy()
        w[a:b] = np.asarray(center + dw.astype(np.longdouble), float)
        c[:, a:b] = np.asarray(refined, float)
        records.append(
            {
                "begin": a,
                "end": b,
                "center_hartree": float(center),
                "shifted_energies_hartree": dw.tolist(),
                "max_energy_change_hartree": float(np.max(abs(w[a:b] - old))),
                "max_shifted_absolute_residual": float(
                    np.max(np.linalg.norm(residual, axis=0))
                ),
            }
        )
    if np.any(np.diff(w) < 0):
        raise ValueError("cluster refinement changed ordering across a boundary")
    gram = float(np.max(abs(c.T @ c - np.eye(len(w))), initial=0.0))
    if gram > 1e-10:
        raise ValueError("cluster output eigenvectors failed Gram gate")
    hc = h @ c
    residual = np.linalg.norm(hc - c * w, axis=0)
    scale = np.linalg.norm(hc, axis=0) + abs(w) * np.linalg.norm(c, axis=0)
    relative = residual / np.maximum(scale, np.finfo(float).tiny)
    repairs = []
    for iteration in range(residual_iterations):
        failing = np.flatnonzero(relative > 1e-10)
        if not len(failing):
            break
        extended_h = h.astype(np.longdouble)
        for column in failing:
            differences = w.astype(np.longdouble) - np.longdouble(w[column])
            outside = np.arange(len(w)) != column
            if np.min(abs(differences[outside]), initial=np.inf) <= gap:
                raise ValueError(
                    "isolated residual refinement cannot split a numerical cluster"
                )
            v = c[:, column].astype(np.longdouble)
            r = extended_h @ v - np.longdouble(w[column]) * v
            factors = np.zeros(len(w), dtype=np.longdouble)
            factors[outside] = (
                -(c[:, outside].astype(np.longdouble).T @ r) / differences[outside]
            )
            correction = c.astype(np.longdouble) @ factors
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > 1e-6:
                raise ValueError(
                    "isolated residual correction exceeds local eigenvector bound"
                )
            refined = v + correction
            refined /= np.sqrt(refined @ refined)
            energy = float(refined @ (extended_h @ refined))
            if abs(energy - w[column]) > 1e-8 * max(1.0, abs(w[column])):
                raise ValueError(
                    "isolated residual correction exceeds local energy bound"
                )
            repairs.append(
                {
                    "iteration": iteration,
                    "column": int(column),
                    "before_relative_residual": float(relative[column]),
                    "correction_norm": correction_norm,
                    "energy_change_hartree": float(energy - w[column]),
                }
            )
            c[:, column] = refined
            w[column] = energy
        if np.any(np.diff(w) < 0):
            raise ValueError("isolated residual refinement changed energy ordering")
        gram = float(np.max(abs(c.T @ c - np.eye(len(w))), initial=0.0))
        if gram > 1e-10:
            raise ValueError("isolated residual refinement failed Gram gate")
        hc = h @ c
        residual = np.linalg.norm(hc - c * w, axis=0)
        scale = np.linalg.norm(hc, axis=0) + abs(w) * np.linalg.norm(c, axis=0)
        relative = residual / np.maximum(scale, np.finfo(float).tiny)
    if np.max(relative, initial=0.0) > 1e-10:
        raise ValueError("cluster eigenvectors failed original-matrix residual gate")
    info = {
        "gap_hartree": float(gap),
        "max_cluster_size": int(max_cluster_size),
        "extended_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "clusters": records,
        "gram_error": gram,
        "max_original_relative_residual": float(np.max(relative, initial=0.0)),
        "max_original_absolute_residual": float(np.max(residual, initial=0.0)),
    }
    if residual_iterations:
        info["isolated_residual_iterations"] = int(residual_iterations)
        info["isolated_residual_corrections"] = repairs
    return w, c, info
