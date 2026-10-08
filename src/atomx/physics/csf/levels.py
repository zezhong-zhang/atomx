"""Shared CI level selection; independent of orbital optimization or dynamics.

KNOWN GAP: roots are energy ordered inside J/parity, not tracked by character.
A selector is resolved again for each spectrum; it is not a cross-basis identity.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class CILevel:
    """Select the zero-based energy root within a doubled-J/parity block.

    Parity is +1/-1. An individual exact-degenerate root has no unique physical
    identity: use an equally weighted level population or an explicit density
    matrix in a recorded CI basis for a prepared coherent state.
    """

    J2: int
    parity: int
    root: int = 0

    def __post_init__(self):
        if (
            any(
                isinstance(x, (bool, np.bool_))
                or not isinstance(x, (int, np.integer))
                for x in (self.J2, self.parity, self.root)
            )
            or self.J2 < 0
            or self.parity not in (-1, 1)
            or self.root < 0
        ):
            raise ValueError("2J/root must be nonnegative integers; parity must be +/-1")
        for name in ("J2", "parity", "root"):
            object.__setattr__(self, name, int(getattr(self, name)))

    def resolve(self, csfs, result):
        """Return this spectrum's column; reject a nonunique degenerate root."""
        weights = CILevelPopulation(((self, 1.0),)).weights(csfs, result)
        return int(np.flatnonzero(weights)[0])


def ci_level_blocks(csfs, result, *, require_full=False):
    """Validate a real orthonormal CI spectrum and order roots within J/parity."""
    energies = np.asarray(result.state_energies)
    mixing = np.asarray(result.mixing_coefficients)
    count = energies.size
    if (
        energies.ndim != 1
        or count == 0
        or (require_full and count != len(csfs))
        or mixing.shape != (len(csfs), count)
        or np.iscomplexobj(energies)
        or np.iscomplexobj(mixing)
        or not np.isfinite(energies).all()
        or not np.isfinite(mixing).all()
    ):
        scope = "full " if require_full else ""
        raise ValueError(f"level selection requires the {scope}real orthonormal CI spectrum")
    sectors = {}
    for row, csf in enumerate(csfs):
        sectors.setdefault((csf.total_J2, csf.parity), []).append(row)
    blocks = {}
    block_grams = []
    structural_nonzeros = 0
    for sector, rows in sectors.items():
        populations = np.sum(mixing[rows] ** 2, axis=0)
        if np.any((populations > 1e-9) & (populations < 1 - 1e-9)):
            raise ValueError("level selection requires symmetry-pure CI eigenvectors")
        columns = np.flatnonzero(populations > 1 - 1e-9)
        blocks[sector] = columns[np.argsort(energies[columns], kind="stable")]
        part = mixing[np.ix_(rows, columns)]
        block_grams.append(part)
        structural_nonzeros += np.count_nonzero(part)
    assigned = np.concatenate(list(blocks.values())) if blocks else np.array([],dtype=int)
    # Only exact structural zeros justify omitting cross-block Gram products.
    # Small leakage is never truncated: it takes the original full check.
    exact_blocks = (len(assigned)==count and len(np.unique(assigned))==count
                    and structural_nonzeros==np.count_nonzero(mixing))
    if exact_blocks:
        orthonormal = all(np.allclose(part.T @ part,np.eye(part.shape[1]),
                                     atol=1e-9,rtol=0) for part in block_grams)
    else:
        orthonormal = np.allclose(mixing.T @ mixing,np.eye(count),atol=1e-9,rtol=0)
    if not orthonormal:
        scope = "full " if require_full else ""
        raise ValueError(f"level selection requires the {scope}real orthonormal CI spectrum")
    return energies, blocks


def normalize_ci_weights(weights, energies, blocks):
    """Normalize relative weights and reject gauge-dependent degenerate mixtures."""
    weights = np.asarray(weights, dtype=float).copy()
    weights /= weights.max()
    weights /= weights.sum()
    for columns in blocks.values():
        for column in columns:
            same = columns[abs(energies[columns] - energies[column]) < 1e-10]
            if not np.allclose(weights[same], weights[column], atol=1e-14, rtol=0):
                raise ValueError("unequal CI weights at a same-symmetry degeneracy")
    return weights


@dataclass(frozen=True)
class CILevelPopulation:
    """Relative physical level populations, unpolarized over each level's M.

    ``levels=((CILevel(...), weight), ...)``. These are total level populations,
    not per-M weights and not an orbital optimization objective. No automatic
    (2J+1) factor is applied. Coherences require an explicit magnetic density.
    """

    levels: tuple

    def __post_init__(self):
        levels = tuple(tuple(item) for item in self.levels)
        if not levels:
            raise ValueError("a CI level population requires selected levels")
        seen = set()
        for item in levels:
            if len(item) != 2 or not isinstance(item[0], CILevel):
                raise TypeError("levels must contain (CILevel, weight)")
            level, weight = item
            if (
                isinstance(weight, (bool, np.bool_))
                or not isinstance(weight, Real)
                or not np.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError("CI population weights must be finite and positive")
            if level in seen:
                raise ValueError("duplicate CI level selector")
            seen.add(level)
        object.__setattr__(self, "levels", tuple((s, float(w)) for s, w in levels))

    def weights(self, csfs, result):
        """Resolve to normalized physical populations in this CI column order."""
        energies, blocks = ci_level_blocks(csfs, result)
        weights = np.zeros(len(energies))
        scale = max(weight for _, weight in self.levels)
        for level, weight in self.levels:
            columns = blocks.get((level.J2, level.parity), ())
            if level.root >= len(columns):
                raise ValueError(f"CI level selector {level} is outside the space")
            weights[columns[level.root]] = weight / scale
        return normalize_ci_weights(weights, energies, blocks)


def initial_ci_level_weights(selection, csfs, result):
    """Resolve an explicit CI selector/population or a legacy snapshot column."""
    if isinstance(selection, CILevel):
        selection = CILevelPopulation(((selection, 1.0),))
    if isinstance(selection, CILevelPopulation):
        return selection.weights(csfs, result)
    if isinstance(selection, (int, np.integer)) and not isinstance(
        selection, (bool, np.bool_)
    ):
        if selection not in range(len(result.state_energies)):
            raise ValueError("initial CI column is outside the space")
        weights = np.zeros(len(result.state_energies))
        weights[int(selection)] = 1.0
        return weights
    raise TypeError("initial_state requires a CILevel, CILevelPopulation or integer column")
