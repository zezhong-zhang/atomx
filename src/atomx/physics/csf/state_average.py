"""Fixed CI level ensembles for a shared-orbital Hamiltonian objective."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from atomx.physics.csf.levels import ci_level_blocks, normalize_ci_weights


@dataclass(frozen=True)
class CIStateAverage:
    """Extended optimized levels, with roots ordered within each J/parity block.

    ``states=None`` averages every resolved CI level. Otherwise provide
    ``((2J, parity, zero_based_root, weight), ...)``. Parity is +1/-1.
    Weights are relative, positive and fixed during orbital optimization.
    ``statistical=True`` additionally multiplies each weight by ``2J+1``.
    This is a CI energy ensemble, not a fractional-occupation DF model.

    Roots follow energy order within their symmetry block, not configuration
    character. Unequal weights at an exact same-symmetry degeneracy are
    rejected: that objective has no basis-independent orbital gradient.
    """

    states: tuple | None = None
    statistical: bool = False

    def __post_init__(self):
        if not isinstance(self.statistical, bool):
            raise TypeError("statistical must be a bool")
        if self.states is None:
            return
        states = tuple(tuple(item) for item in self.states)
        if not states:
            raise ValueError("a CI state average requires selected levels")
        seen = set()
        for item in states:
            if len(item) != 4:
                raise ValueError("states must contain (2J, parity, root, weight)")
            j2, parity, root, weight = item
            if (
                any(
                    isinstance(x, (bool, np.bool_))
                    or not isinstance(x, (int, np.integer))
                    for x in (j2, parity, root)
                )
                or j2 < 0
                or parity not in (-1, 1)
                or root < 0
            ):
                raise ValueError(
                    "2J/root must be nonnegative integers; parity must be +/-1"
                )
            if not isinstance(weight, Real) or not np.isfinite(weight) or weight <= 0:
                raise ValueError("CI ensemble weights must be finite and positive")
            if item[:3] in seen:
                raise ValueError("duplicate CI level selector")
            seen.add(item[:3])
        object.__setattr__(
            self,
            "states",
            tuple((int(j), int(p), int(r), float(w)) for j, p, r, w in states),
        )

    def weights(self, csfs, result):
        """Resolve symmetry/local-root selectors to normalized CI columns."""
        energies, blocks = ci_level_blocks(csfs, result, require_full=True)
        weights = np.zeros(len(energies))
        if self.states is None:
            for (j2, _), columns in blocks.items():
                weights[columns] = j2 + 1 if self.statistical else 1
        else:
            scale = max(item[3] for item in self.states)
            for j2, parity, root, weight in self.states:
                columns = blocks.get((j2, parity), ())
                if root >= len(columns):
                    raise ValueError(
                        f"CI level selector {(j2, parity, root)} is outside the space"
                    )
                weights[columns[root]] = (
                    weight / scale * (j2 + 1 if self.statistical else 1)
                )
        return normalize_ci_weights(weights, energies, blocks)


@dataclass(frozen=True)
class CIChargeStateAverage:
    """Fixed CI Hamiltonian average across electron-number sectors.

    ``sectors=((N, charge_weight, CIStateAverage(...)), ...)`` specifies
    independent normalized level ensembles and relative charge weights.
    The parent sector must be included; daughters span the complete Fock
    space over the retained parent orbitals. All share one orthonormal
    orbital set, optimized for sum_N W_N sum_i w_Ni E_Ni. These are fixed
    optimization weights, not experimental populations or fractional DF.

    B-spline Dirac-Coulomb MCDHF only. Distinct charge-specific orbital
    sets and dynamic orbital relaxation are not supplied. This declaration
    uses sorted roots; the orbital functional can explicitly bind it to
    CIBasisTracking. The magnetic-state cap applies before enumeration.
    """

    sectors: tuple
    magnetic_state_cap: int = 512

    def __post_init__(self):
        cap = self.magnetic_state_cap
        if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)) or cap < 1:
            raise ValueError("positive integer magnetic_state_cap required")
        sectors = tuple(tuple(item) for item in self.sectors)
        if not sectors:
            raise ValueError("a charge state average requires selected sectors")
        seen = set()
        output = []
        for item in sectors:
            if len(item) != 3:
                raise ValueError("sectors must contain (electrons, weight, CIStateAverage)")
            number, weight, levels = item
            if (isinstance(number, (bool, np.bool_))
                    or not isinstance(number, (int, np.integer)) or number < 1):
                raise ValueError("positive integer electron number required")
            if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real) or not np.isfinite(weight) or weight <= 0:
                raise ValueError("charge weights must be finite and positive")
            if not isinstance(levels, CIStateAverage):
                raise TypeError("each charge sector requires CIStateAverage")
            if number in seen:
                raise ValueError("duplicate electron-number sector")
            seen.add(number)
            output.append((int(number), float(weight), levels))
        object.__setattr__(self, "sectors", tuple(sorted(output, reverse=True)))
        object.__setattr__(self, "magnetic_state_cap", int(cap))

    def normalized_sectors(self, parent_electrons):
        """Return charge weights without modifying any level ensemble."""
        if parent_electrons not in {n for n, _, _ in self.sectors}:
            raise ValueError("charge objective must include the parent electron sector")
        if any(n > parent_electrons for n, _, _ in self.sectors):
            raise ValueError("charge objective cannot exceed the parent electron number")
        scale = max(w for _, w, _ in self.sectors)
        weights = np.array([w / scale for _, w, _ in self.sectors])
        weights /= weights.sum()
        if np.any(weights == 0):
            raise ValueError("charge weight normalization underflow")
        return tuple((n, float(w), levels) for (n, _, levels), w in zip(self.sectors, weights, strict=True))
