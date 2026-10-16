"""Signed same-kappa matrix attachment Sigma2 in a frozen closed-core field.

The two diagram classes and unnormalized antisymmetric jj pair amplitudes
are shared with ``mbpt_transitions``. A residue is a sum of signed outer
products, including magnetic averaging and identical-pair multiplicity.
Finite external-space Dyson eigenvectors include orbital mixing; the core,
intermediate states and pole energies remain frozen. This is not all-order
SD, a self-consistent GF2 calculation or a dressed transition operator.
"""
from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atomx.methods.mbpt_transitions import (
    DiagonalSelfEnergy,
    _capture_attachment_self_energy,
)

__all__ = ["MatrixSelfEnergy", "MatrixDysonRoot", "build_mbpt2_matrix_self_energy"]


def _radial_table_evaluators(targets, partner, ket_states, data, cache):
    """One physical bra at a time; reuse the shared full Slater GEMM tables.

    Ownership is limited to this capture and this bra. Keep every declared
    ket coefficient and every requested multipole; selection/antisymmetry
    remains in the common jj kernel. Old tables are released at the next bra.
    """
    from atomx.numerics.bspline.convol import (
        build_slater_rk_bilinear,
        compute_slater_rk_table,
    )

    index = {(id(s[0]), id(s[1])): i for i, s in enumerate(ket_states)}
    cp = np.array([s[0] for s in ket_states]).T
    cq = np.array([s[1] for s in ket_states]).T

    def evaluator(target):
        tables = {}

        def radial(k, P2, Q2, P4, Q4):
            if k not in tables:
                bilinear = build_slater_rk_bilinear(*target[:2], *partner[:2], data, k, cache=cache)
                tables[k] = compute_slater_rk_table(bilinear, cp, cq, cp, cq)
            return float(tables[k][index[id(P2), id(Q2)], index[id(P4), id(Q4)]])

        return radial

    return [evaluator(target) for target in targets]


def _immutable(values):
    values = np.asarray(values, dtype=float)
    return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)


@dataclass(frozen=True)
class MatrixDysonRoot:
    energy_Ha: float
    shift_Ha: float
    coefficients: np.ndarray
    residue: float
    df_weight: float
    residual_Ha: float
    reference_index: int
    bracket_Ha: tuple[float, float]
    reference_identity: str

    def __post_init__(self):
        object.__setattr__(self, "coefficients", _immutable(self.coefficients))


@dataclass(frozen=True)
class MatrixSelfEnergy:
    """Immutable Sigma(E)=sum R_t/(E-p_t), with real PSD residue matrices.

    External basis is orthonormal, same kappa and diagonal in the declared
    DF reference Hamiltonian. The caller supplies its full physical snapshot
    identity. No residue magnitude threshold or pole merging is applied.
    Only exactly zero matrices are discarded. PSD/symmetry roundoff gates
    use 64 machine eps times each matrix norm; they do not alter residues.
    """
    reference_energies_Ha: np.ndarray
    poles_Ha: np.ndarray
    residues_Ha2: np.ndarray
    reference_identity: str
    denominator_floor_Ha: float = 1e-8
    ci_binding: object | None = None

    def __post_init__(self):
        eps, poles, residues = map(np.asarray, (
            self.reference_energies_Ha, self.poles_Ha, self.residues_Ha2))
        n = eps.size
        if self.ci_binding is not None:
            from atomx.physics.csf.effective_one_body import CIReferenceBinding
            if not isinstance(self.ci_binding, CIReferenceBinding) or len(self.ci_binding.external) != n:
                raise ValueError("invalid CI producer binding")
        if (eps.ndim != 1 or n == 0 or poles.ndim != 1
                or residues.shape != (poles.size, n, n)
                or any(not np.isrealobj(v) or not np.all(np.isfinite(v))
                       for v in (eps, poles, residues))
                or not isinstance(self.reference_identity, str) or not self.reference_identity.strip()
                or not np.isfinite(self.denominator_floor_Ha) or self.denominator_floor_Ha <= 0):
            raise ValueError("invalid real matrix self-energy snapshot or missing reference identity")
        for start in range(0, len(poles), 8192):
            block = residues[start:start+8192]
            scale = np.max(np.abs(block), axis=(1, 2))
            tolerance = 64*np.finfo(float).eps*scale
            if (np.any(np.max(np.abs(block-block.swapaxes(1, 2)), axis=(1, 2)) > tolerance)
                    or np.any(np.linalg.eigvalsh(block)[:, 0] < -tolerance)):
                raise ValueError("matrix residues must be symmetric positive semidefinite")
        keep = np.any(residues != 0, axis=(1, 2))
        for name, values in (("reference_energies_Ha", eps),
                             ("poles_Ha", poles[keep]), ("residues_Ha2", residues[keep])):
            object.__setattr__(self, name, _immutable(values))

    def _denominators(self, energy_Ha):
        if np.ndim(energy_Ha) != 0 or not np.isrealobj(energy_Ha) or not np.isfinite(energy_Ha):
            raise ValueError("self-energy energy must be a finite real scalar")
        den = energy_Ha-self.poles_Ha
        if np.any(np.abs(den) < self.denominator_floor_Ha):
            raise ValueError("resonant second-order self-energy denominator")
        return den

    def evaluate(self, energy_Ha: float) -> np.ndarray:
        return np.einsum("t,tij->ij", 1/self._denominators(energy_Ha), self.residues_Ha2)

    def derivative(self, energy_Ha: float) -> np.ndarray:
        return np.einsum("t,tij->ij", -1/self._denominators(energy_Ha)**2, self.residues_Ha2)

    def diagonal(self, index: int) -> DiagonalSelfEnergy:
        """Exact projection of this snapshot; not a separately rebuilt space."""
        return DiagonalSelfEnergy(self.reference_energies_Ha[index], self.poles_Ha,
                                  self.residues_Ha2[:, index, index],
                                  self.reference_identity+f":diagonal={index}",
                                  self.denominator_floor_Ha)

    def subspace(self, indices: Sequence[int]) -> MatrixSelfEnergy:
        indices = tuple(indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(self.reference_energies_Ha)
                       for i in indices)):
            raise ValueError("require distinct valid external indices")
        return MatrixSelfEnergy(self.reference_energies_Ha[list(indices)], self.poles_Ha,
                                self.residues_Ha2[:, indices, :][:, :, indices],
                                self.reference_identity+f":P={indices}", self.denominator_floor_Ha,
                                self._subspace_binding(indices))

    def _subspace_binding(self, indices):
        if self.ci_binding is None:
            return None
        from dataclasses import replace
        return replace(self.ci_binding, external=tuple(self.ci_binding.external[i] for i in indices))

    def as_ci_operator(self, energy_Ha, orbital_keys, orbitals, workspace, Z, ns, *,
                       correlated_core, nuclear_potential=None):
        """Export Σ₁(E*) only into its captured signed physical reference.

        Legacy/unbound snapshots can still evaluate and solve Dyson roots,
        but cannot silently acquire a different core or orbital phase at CI.
        This frozen-energy operator is not a self-consistent CI+all-order solve.
        """
        from atomx.physics.csf.effective_one_body import CIEffectiveOneBody
        if self.ci_binding is None:
            raise ValueError("unbound self-energy snapshot: producer CI binding required")
        return CIEffectiveOneBody(
            orbital_keys, self.evaluate(energy_Ha), orbitals, workspace, Z, ns,
            correlated_core=correlated_core, reference_identity=self.reference_identity,
            theory="closed-core second-order attachment Sigma1 (2p1h + 2h1p)",
            energy_Ha=energy_Ha, nuclear_potential=nuclear_potential,
            producer_binding=self.ci_binding)

    def solve_dyson(self, bracket_Ha: tuple[float, float], *, reference_index: int = 0) -> MatrixDysonRoot:
        """Solve an ordered DF-connected eigenbranch in one pole-free interval.

        Residue PSD implies Sigma'(E)<=0, so E-lambda_j(H_DF+Sigma(E))
        is strictly increasing. Order j is inherited from nondegenerate DF
        energies, making the root unique and continuously connected without
        choosing a root for its experimental agreement. Degenerate DF or
        final eigenvalues are rejected; a degenerate subspace needs its own
        state matching contract.
        """
        from scipy.optimize import brentq

        n = len(self.reference_energies_Ha)
        if not isinstance(reference_index, (int, np.integer)) or not 0 <= reference_index < n:
            raise ValueError("invalid DF reference index")
        eps = self.reference_energies_Ha
        ordered = np.argsort(eps)
        if n > 1 and np.any(np.diff(eps[ordered]) <= 1e-12):
            raise ValueError("degenerate DF states require a subspace matching contract")
        order = int(np.flatnonzero(ordered == reference_index)[0])
        lo, hi = bracket_Ha
        if (any(np.ndim(x) != 0 or not np.isrealobj(x) or not np.isfinite(x) for x in (lo, hi))
                or lo >= hi or not lo <= eps[reference_index] <= hi):
            raise ValueError("Dyson bracket must contain the DF reference energy")
        if np.any((self.poles_Ha >= lo-self.denominator_floor_Ha)
                  & (self.poles_Ha <= hi+self.denominator_floor_Ha)):
            raise ValueError("Dyson bracket crosses a nonzero-residue self-energy pole")

        def eigen(energy):
            return np.linalg.eigh(np.diag(eps)+self.evaluate(energy))

        energy = brentq(lambda e: e-eigen(e)[0][order], lo, hi,
                        xtol=1e-13, rtol=4*np.finfo(float).eps)
        values, vectors = eigen(energy)
        if n > 1 and np.min(np.abs(np.delete(values, order)-values[order])) <= 1e-11:
            raise ValueError("degenerate Dyson states require a subspace matching contract")
        u = vectors[:, order]
        pivot = reference_index if u[reference_index] != 0 else int(np.argmax(abs(u)))
        if u[pivot] < 0:
            u = -u
        error = float(np.linalg.norm((np.diag(eps)+self.evaluate(energy))@u-energy*u))
        residue = 1/(1-float(u@self.derivative(energy)@u))
        if error > 1e-10 or not 0 < residue <= 1+1e-13:
            raise RuntimeError("matrix Dyson root failed residual or residue gate")
        return MatrixDysonRoot(float(energy), float(energy-eps[reference_index]), u,
                               float(residue), float(u[reference_index]**2), error,
                               int(reference_index), (float(lo), float(hi)), self.reference_identity)


def build_mbpt2_matrix_self_energy(
    scf_result, valence_orbitals: Sequence[tuple[np.ndarray, np.ndarray, int, float]], *,
    reference_identity: str, virtual_kappas: Sequence[int],
    energy_window_au: float = 50., n_virt_max: int | None = None,
    denom_floor: float = 1e-8, term_recorder=None, radial_backend: str = "table",
) -> MatrixSelfEnergy:
    """Capture both full signed jj diagrams in a same-kappa external block.

    All external spinors must be normalized, mutually orthogonal and core
    orthogonal in the native metric. The reference H0 is their supplied
    diagonal DF energies; orbital diagonalization itself is a caller duty.
    The complete reference VN virtual field includes bound virtuals; no
    external target is removed from the intermediate sum. Callback arguments
    are ``(pole_Ha, residue_matrix_Ha2, diagram)`` for every physical pair.
    ``radial_backend="table"`` reuses the shared full Slater GEMM kernel;
    ``"scalar"`` retains per-pair contractions as a numerical control. Both
    use exactly the same orbitals, multipoles, diagram sums and angular kernel.
    """
    from atomx.numerics.radial_workspace import eval_orbital_pair, workspace_arrays

    if not isinstance(reference_identity, str) or not reference_identity.strip():
        raise ValueError("missing reference identity")
    if radial_backend not in ("table", "scalar"):
        raise ValueError("radial_backend must be 'table' or 'scalar'")
    targets = list(valence_orbitals)
    if not targets or len({t[2] for t in targets}) != 1:
        raise ValueError("require a nonempty same-kappa external block")
    data = scf_result._bsp_data
    # Shape/finiteness/core/norm checks are performed by the shared capture
    # before any angular sum. Cross overlaps are the added matrix contract.
    _, quadrature = workspace_arrays(data)
    waves = [eval_orbital_pair(t[0], t[1], data) for t in targets]
    for i, (p, q) in enumerate(waves):
        for pp, qq in waves[:i]:
            if abs(float(np.sum(quadrature*(p*pp+q*qq)))) > 1e-7:
                raise ValueError("external matrix orbitals must be mutually orthogonal")
    poles, residues = array("d"), array("d")

    def capture(pole, matrix, diagram):
        if np.any(matrix != 0):
            poles.append(pole)
            residues.extend(matrix.ravel())
        if term_recorder is not None:
            term_recorder(pole, matrix, diagram)

    _capture_attachment_self_energy(
        scf_result, "external_matrix", valence_orbital=targets[0],
        virtual_kappas=virtual_kappas, energy_window_au=energy_window_au,
        n_virt_max=n_virt_max, denom_floor=denom_floor,
        _matrix_orbitals=targets, _matrix_recorder=capture,
        _matrix_radial_factory=_radial_table_evaluators if radial_backend == "table" else None,
    )
    n = len(targets)
    from atomx.physics.csf.effective_one_body import CIReferenceBinding
    core_orbitals = {key: value[:2] for key, value in scf_result.orbitals_by_subshell().items()
                     if value[2] > .01}
    binding = CIReferenceBinding.capture(targets, core_orbitals, data, scf_result.Z,
                                        data['ns'], scf_result._nuclear_potential)
    return MatrixSelfEnergy(np.array([t[3] for t in targets]), np.asarray(poles),
                            np.asarray(residues).reshape(-1, n, n), reference_identity, denom_floor, binding)
