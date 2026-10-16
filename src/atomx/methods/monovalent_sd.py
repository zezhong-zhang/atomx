"""Finite native monovalent linked SD with correlation-energy feedback.

This bounded fermionic anchor holds the converged closed-core T1/T2 fixed.
Its attachment equation retains valence singles and core-valence doubles,
including doubles with a particle in the reference valence orbital. It is
separate from fixed-energy CI+all-order, and from exponential FS-CCSD.
No full determinant Hamiltonian, ordered quartic amplitude array, amplitude
cutoff, fitted shift or Hermitian average is used. It is not the scalable
J-reduced production implementation.
"""

import hashlib
import itertools
from dataclasses import dataclass
from math import comb
from pathlib import Path
from types import MappingProxyType

import numpy as np

from atomx.angular.csf_hamiltonian import _ops_sign
from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
from atomx.angular.sd_rho import rho_coefficient
from atomx.methods.core_sd import CoreSD, CoreSDResult, _owned
from atomx.numerics.fermion import apply_terms

_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"positive integer {name} required")
    return int(value)


def _positive_real(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or np.ndim(value) != 0
        or not np.isrealobj(value)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"positive finite {name} required")
    return float(value)


def _symmetry(labels, particles, holes):
    return (
        sum(labels[p][2] for p in particles) == sum(labels[h][2] for h in holes)
        and (
            sum(l_of_kappa(labels[p][1]) for p in particles)
            - sum(l_of_kappa(labels[h][1]) for h in holes)
        )
        % 2
        == 0
    )


def plan_finite_monovalent_sd(
    labels,
    noccupied,
    valence_index,
    *,
    amplitude_cap=256,
    determinant_cap=2000,
    memory_bytes=64 * 1024**2,
    work_cap=50_000_000,
):
    """Count the whole declared finite space before integral/cluster actions.

    Fixed-N binomial counts bound every intermediate dictionary, even though
    no full fixed-N matrix is allocated. Payload estimates include generous
    Python term/dictionary allowances and dense eigensolver scratch; exclude
    the already-owned native core, integral caches and interpreter RSS.
    """
    labels = tuple(tuple(label) for label in labels)
    n = len(labels)
    o = _positive_integer(noccupied, "occupied count")
    if (
        n > 64
        or o >= n
        or len(set(labels)) != n
        or any(
            len(x) != 3
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                for v in x
            )
            or x[0] < 1
            or x[1] == 0
            for x in labels
        )
        or isinstance(valence_index, (bool, np.bool_))
        or not isinstance(valence_index, (int, np.integer))
        or not o <= valence_index < n
    ):
        raise ValueError(
            "complete distinct finite spinors and virtual valence index required"
        )
    caps = {
        name: _positive_integer(value, name)
        for name, value in (
            ("amplitude cap", amplitude_cap),
            ("determinant cap", determinant_cap),
            ("memory cap", memory_bytes),
            ("work cap", work_cap),
        )
    }
    multiplets = {}
    for i, (nr, k, m) in enumerate(labels):
        multiplets.setdefault((nr, k), []).append((i, m))
    for (_, k), members in multiplets.items():
        if set(m for _, m in members) != set(
            range(-twoj_of_kappa(k), twoj_of_kappa(k) + 1, 2)
        ) or any(i < o for i, _ in members) != all(i < o for i, _ in members):
            raise ValueError(
                "complete magnetic multiplets and closed occupied core required"
            )
    core_count, attached_count = comb(n, o), comb(n, o + 1)
    if max(core_count, attached_count) > caps["determinant cap"]:
        raise MemoryError(
            "complete finite core/attached determinant intermediate bound exceeds cap"
        )
    virtual = range(o, n)
    singles = sum(
        m != valence_index and _symmetry(labels, (m,), (valence_index,))
        for m in virtual
    )
    doubles = sum(
        _symmetry(labels, pair, (valence_index, b))
        for pair in itertools.combinations(virtual, 2)
        for b in range(o)
    )
    core_coordinates = o * (n - o) + comb(o, 2) * comb(n - o, 2)
    size = 1 + singles + doubles
    if size - 1 > caps["amplitude cap"]:
        raise MemoryError(
            "complete finite valence singles/doubles exceed amplitude cap"
        )
    terms = n * n + comb(n, 2) ** 2 + 1
    # All commutators, basis columns and the full physical-state residual.
    work = (
        8
        * (size + 1)
        * (core_count + attached_count)
        * (terms + core_coordinates + size)
    )
    payload = (
        384 * (terms + core_coordinates + size)
        + 512 * 12 * (core_count + attached_count)
        + 8 * (16 * size * size + 32 * size)
    )
    if payload > caps["memory cap"]:
        raise MemoryError(
            f"complete finite monovalent payload bound {payload} exceeds memory cap"
        )
    if work > caps["work cap"]:
        raise MemoryError(
            f"complete finite monovalent work bound {work} exceeds work cap"
        )
    return dict(
        spinors=n,
        core_electrons=o,
        valence_index=int(valence_index),
        singles=singles,
        doubles=doubles,
        amplitudes=size - 1,
        core_determinant_bound=core_count,
        attached_determinant_bound=attached_count,
        operator_term_bound=terms,
        working_payload_bound_bytes=payload,
        elementary_action_bound=work,
        symmetry="exact total 2M and parity; no J truncation",
        scope="finite anchor; complete intermediates counted, no production HPC claim",
    )


def _core_terms(core, amplitudes):
    """Expand owned redundant rho_k directly to independent excitation terms.

    The 1/2 direct core convention is antisymmetrized in BOTH particle and
    hole order. Radial blocks retain original Fortran column order. No
    magnetic quartic amplitude tensor is formed.
    """
    plan, labels, o = core.plan, core.integrals.labels, core.integrals.noccupied
    occupied, virtual = range(o), range(o, len(labels))
    positions = {key: i for i, key in enumerate(plan["occupied_keys"])}
    radial = {key: i for i, group in enumerate(plan["groups"]) for key in group}
    singles, tables, cursor = {}, {}, 0
    for a in plan["occupied_keys"]:
        for m in plan["groups"][radial[a]]:
            if m in plan["virtual_keys"]:
                singles[m, a] = amplitudes[cursor]
                cursor += 1
    for a, b, im, jn, rank in plan["channels"]:
        mkeys = tuple(key for key in plan["groups"][im] if key in plan["virtual_keys"])
        nkeys = tuple(key for key in plan["groups"][jn] if key in plan["virtual_keys"])
        size = len(mkeys) * len(nkeys)
        values = amplitudes[cursor : cursor + size].reshape(
            (len(mkeys), len(nkeys)), order="F"
        )
        tables[a, b, im, jn, rank] = {
            (m, n): values[i, j]
            for i, m in enumerate(mkeys)
            for j, n in enumerate(nkeys)
        }
        cursor += size
    if cursor != len(amplitudes):
        raise ValueError("complete core rho coordinate coverage failed")

    def rho(m, n, a, b):
        ai, bi = positions[labels[a][:2]], positions[labels[b][:2]]
        im, jn = radial[labels[m][:2]], radial[labels[n][:2]]
        pair = (
            (labels[m][:2], labels[n][:2])
            if ai >= bi
            else (labels[n][:2], labels[m][:2])
        )
        total = 0.0
        for rank in range(plan["dimensions"]["KK"] + 1):
            key = (ai, bi, im, jn, rank) if ai >= bi else (bi, ai, jn, im, rank)
            table = tables.get(key)
            if table is not None:
                total += (
                    rho_coefficient(rank, labels[m], labels[n], labels[a], labels[b])
                    * table[pair]
                )
        return total

    terms = []
    for m, a in itertools.product(virtual, occupied):
        if labels[m][1:] == labels[a][1:]:
            value = float(singles[labels[m][:2], labels[a][:2]])
            if value != 0:
                terms.append((value, ((m, True), (a, False))))
    for m, n in itertools.combinations(virtual, 2):
        for a, b in itertools.combinations(occupied, 2):
            if _symmetry(labels, (m, n), (a, b)):
                value = 0.5 * (
                    rho(m, n, a, b)
                    - rho(n, m, a, b)
                    - rho(m, n, b, a)
                    + rho(n, m, b, a)
                )
                if value != 0:
                    terms.append(
                        (float(value), ((m, True), (n, True), (b, False), (a, False)))
                    )
    return tuple(terms)


@dataclass(frozen=True)
class FiniteMonovalentSDResult:
    valence_index: int
    orbital_energy_Ha: float
    correlation_energy_Ha: float
    attachment_energy_Ha: float
    total_energy_Ha: float
    amplitudes: np.ndarray
    determinants: tuple
    coefficients: np.ndarray
    wavefunction_norm: float
    projected_residual_Ha: float
    energy_residual_Ha: float
    full_eigenstate_residual_Ha: float
    expectation_energy_Ha: float
    continuation_path: tuple
    provenance: object
    _owner: object

    def __post_init__(self):
        for name in ("amplitudes", "coefficients"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class FiniteMonovalentSD:
    """Source-bound native DC anchor for the standalone monovalent equations.

    The core part is H+[H,T_core] minus its attached core spectator. The
    valence part retains the core-normal-ordered connected SD diagrams;
    an ordinary attached-reference [H,S_v] alone removes a required S_v1
    diagram and is insufficient. Its excited projection is
    ``b + K s = deltaE_v s``; its reference
    projection is ``eps_v + c + e s``. The fixed core is never evaluated
    with a valence-shifted denominator. The physical SD state is the
    normalized ``(1+T_core+S_v)|Phi_v>``; it is not asserted to be an exact
    eigenstate or an exponential CC state.
    """

    def __init__(self, core_problem, core_result, valence_index, **resource_caps):
        if not isinstance(core_problem, CoreSD) or not isinstance(
            core_result, CoreSDResult
        ):
            raise TypeError("owned native CoreSD problem and result required")
        self.core, self.integrals = core_problem, core_problem.integrals
        self.plan = MappingProxyType(
            plan_finite_monovalent_sd(
                self.integrals.labels,
                self.integrals.noccupied,
                valence_index,
                **resource_caps,
            )
        )
        if (
            any(
                core_result.provenance.get(k) != v
                for k, v in self.core.provenance.items()
            )
            or core_result.amplitudes.shape != self.core.source.shape
            or not np.all(np.isfinite(core_result.amplitudes))
        ):
            raise ValueError("complete core result source/physical identity differs")
        if not self.core._has_current_solution_certificate(core_result, 1e-10):
            residual = self.core.action(core_result.amplitudes) - self.core.source
            if not np.all(np.isfinite(residual)) or np.max(abs(residual)) > 1e-10:
                raise ValueError("actual complete core SD residual failed")
        self._core_binding = self.core._solution_identity(core_result)
        self._core_result = core_result
        self.valence_index = int(valence_index)
        self.core_mask = (1 << self.integrals.noccupied) - 1
        self.reference_mask = self.core_mask | (1 << self.valence_index)
        self.core_terms = _core_terms(self.core, core_result.amplitudes)
        self._hterms = self._hamiltonian_terms()
        reference = {self.core_mask: 1.0}
        core_action = apply_terms(self._hterms, reference)
        core_correction = self._commutator(self.core_terms, reference)
        core_residual = max(
            (
                abs(core_action.get(mask, 0) + core_correction.get(mask, 0))
                for mask in core_action.keys() | core_correction.keys()
                if (self.core_mask & ~mask).bit_count() in (1, 2)
            ),
            default=0.0,
        )
        core_reference = core_action.get(self.core_mask, 0)
        core_correlation = core_correction.get(self.core_mask, 0)
        if (
            max(
                core_residual,
                abs(core_reference - core_result.reference_energy_Ha),
                abs(core_correlation - core_result.correlation_energy_Ha),
                abs(core_reference + core_correlation - core_result.total_energy_Ha),
            )
            > 1e-10
        ):
            raise ValueError(
                "expanded core SD residual/energy does not match actual native result"
            )
        self.expanded_core_residual_Ha = float(core_residual)
        self.excitations = self._excitations()
        if len(self.excitations) != self.plan["amplitudes"]:
            raise RuntimeError("complete finite valence coordinate count changed")
        self._signs, self._targets = [], []
        eps = np.diag(self.integrals.fock_Ha).real
        gaps = []
        for particles, holes, operations in self.excitations:
            self._signs.append(_ops_sign(self.reference_mask, operations))
            self._targets.append(
                (self.reference_mask & ~sum(1 << i for i in holes))
                | sum(1 << i for i in particles)
            )
            gaps.append(sum(eps[p] for p in particles) - sum(eps[h] for h in holes))
        if len(set(self._targets)) != len(self._targets) or not all(self._signs):
            raise RuntimeError("independent valence excitation coordinates required")
        self.diagonal_gaps_Ha = _owned(gaps)
        self.orbital_energy_Ha = float(eps[self.valence_index])
        self._valence_induced_core_singles = self._induced_core_singles()
        source = self.linked_action(np.zeros(len(gaps)))
        self.source_Ha = _owned(source)
        matrix = np.empty((len(gaps) + 1, len(gaps)))
        for i in range(len(gaps)):
            unit = np.zeros(len(gaps))
            unit[i] = 1
            # Source-separated exact homogeneous commutator, no affine subtraction.
            matrix[:, i] = self._project(
                self._valence_action(((1.0, self.excitations[i][2]),))
            )
        self.linear_map_Ha = _owned(matrix)
        self._solutions = []

    def _assert_current(self):
        if (
            self.core.integrals is not self.integrals
            or self.core._solution_identity(self._core_result) != self._core_binding
        ):
            raise ValueError("native core/physical/source binding changed")

    def _hamiltonian_terms(self):
        integrals, n = self.integrals, len(self.integrals.labels)
        h = integrals.one_body_Ha
        if np.any(h.imag != 0):
            raise ValueError("finite standalone SD currently requires real native DC")
        terms = [(float(integrals.frozen_energy_Ha), ())]
        for p, q in itertools.product(range(n), repeat=2):
            if h[p, q] != 0:
                terms.append((float(h[p, q].real), ((p, True), (q, False))))
        for p, q in itertools.combinations(range(n), 2):
            for r, s in itertools.combinations(range(n), 2):
                value = integrals.eri(p, q, r, s)
                if value.imag != 0:
                    raise ValueError(
                        "finite standalone SD currently requires real native DC"
                    )
                if value != 0:
                    terms.append(
                        (
                            float(value.real),
                            ((p, True), (q, True), (s, False), (r, False)),
                        )
                    )
        return tuple(terms)

    def _excitations(self):
        o, n, v = (
            self.integrals.noccupied,
            len(self.integrals.labels),
            self.valence_index,
        )
        labels, result = self.integrals.labels, []
        for m in range(o, n):
            if m != v and _symmetry(labels, (m,), (v,)):
                result.append(((m,), (v,), ((m, True), (v, False))))
        for m, nn in itertools.combinations(range(o, n), 2):
            for b in range(o):
                if _symmetry(labels, (m, nn), (v, b)):
                    result.append(
                        (
                            (m, nn),
                            (v, b),
                            ((m, True), (nn, True), (b, False), (v, False)),
                        )
                    )
        return tuple(result)

    def _commutator(self, terms, reference):
        ht = apply_terms(self._hterms, apply_terms(terms, reference))
        th = apply_terms(terms, apply_terms(self._hterms, reference))
        return {
            mask: ht.get(mask, 0) - th.get(mask, 0) for mask in ht.keys() | th.keys()
        }

    def _project(self, vector):
        return np.array(
            [vector.get(self.reference_mask, 0)]
            + [
                sign * vector.get(mask, 0)
                for mask, sign in zip(self._targets, self._signs, strict=True)
            ]
        )

    def _induced_core_singles(self):
        """Bare core-single source induced by occupation of the valence v.

        Full attached-reference commutation subtracts S_v1 times this
        valence interaction as well as disconnected core Fock singles.
        Core-normal-ordered atomic SD subtracts only the latter. Restore
        the former exactly; this is the Eq.9 singles diagram, not a
        nonlinear cluster product. Fock-only H has zero such source.
        """
        attached = apply_terms(self._hterms, {self.reference_mask: 1.0})
        spectator = apply_terms(
            ((1.0, ((self.valence_index, True),)),),
            apply_terms(self._hterms, {self.core_mask: 1.0}),
        )
        phase = _ops_sign(self.core_mask, ((self.valence_index, True),))
        return {
            mask: attached.get(mask, 0) - phase * spectator.get(mask, 0)
            for mask in attached.keys() | spectator.keys()
            if mask & (1 << self.valence_index)
            and (self.reference_mask & ~mask).bit_count() == 1
        }

    def _valence_action(self, terms):
        # The canonical Coulomb diagrams use core normal ordering. Keep
        # their fixed-reference commutator and required S1 restoration;
        # extending that attached-H commutator to other M would introduce
        # excluded Coulomb contractions. Only the bare-core Fock source
        # needs the other-M components of the scalar radial S1 operator.
        singles = tuple(row for row in terms if len(row[1]) == 2)
        result = self._commutator(terms, {self.reference_mask: 1.0})
        for mask, value in apply_terms(
            singles, self._valence_induced_core_singles
        ).items():
            result[mask] = result.get(mask, 0) + value
        labels = self.integrals.labels
        by_label = {label: i for i, label in enumerate(labels)}
        other_m = []
        for value, operations in singles:
            m, v = (row[0] for row in operations)
            if labels[m][1] != labels[v][1]:
                continue
            for other, label in enumerate(labels):
                if label[:2] == labels[v][:2] and other != v:
                    destination = by_label[(*labels[m][:2], label[2])]
                    other_m.append((value, ((destination, True), (other, False))))
        if other_m:
            fock = self.integrals.fock_Ha.real
            source = apply_terms(
                tuple(
                    (float(fock[p, b]), ((p, True), (b, False)))
                    for p in range(self.integrals.noccupied, len(labels))
                    for b in range(self.integrals.noccupied)
                    if fock[p, b] != 0
                ),
                {self.reference_mask: 1.0},
            )
            for mask, value in apply_terms(tuple(other_m), source).items():
                result[mask] = result.get(mask, 0) - value
        return result

    def _valence_terms(self, amplitudes):
        values = np.asarray(amplitudes)
        if (
            values.shape != (len(self.excitations),)
            or not np.isrealobj(values)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("finite complete real valence amplitudes required")
        return tuple(
            (float(value), row[2])
            for value, row in zip(values, self.excitations, strict=True)
            if value != 0
        )

    def linked_action(self, amplitudes):
        """Reference energy and every excited linked numerator/gap, in Ha."""
        # Also usable during construction, before the homogeneous map exists.
        self._assert_current()
        valence = self._valence_terms(amplitudes)
        attached = {self.reference_mask: 1.0}
        core = {self.core_mask: 1.0}
        result = apply_terms(self._hterms, attached)
        correction = self._commutator(self.core_terms, attached)
        for mask, value in self._valence_action(valence).items():
            correction[mask] = correction.get(mask, 0) + value
        for mask, value in correction.items():
            result[mask] = result.get(mask, 0) + value
        spectator = apply_terms(self._hterms, core)
        for mask, value in self._commutator(self.core_terms, core).items():
            spectator[mask] = spectator.get(mask, 0) + value
        spectator = apply_terms(((1.0, ((self.valence_index, True),)),), spectator)
        # a_v^dagger|core> has sign (-1)^Ncore; closed shells have even Ncore.
        phase = _ops_sign(self.core_mask, ((self.valence_index, True),))
        for mask, value in spectator.items():
            result[mask] = result.get(mask, 0) - phase * value
        return self._project(result)

    def residual(self, amplitudes, correlation_energy_Ha):
        self._assert_current()
        self._valence_terms(amplitudes)
        if (
            isinstance(correlation_energy_Ha, (bool, np.bool_))
            or np.ndim(correlation_energy_Ha) != 0
            or not np.isrealobj(correlation_energy_Ha)
            or not np.isfinite(correlation_energy_Ha)
        ):
            raise ValueError("finite real correlation energy required")
        values = self.source_Ha + self.linear_map_Ha @ amplitudes
        values[0] -= self.orbital_energy_Ha + correlation_energy_Ha
        values[1:] -= correlation_energy_Ha * np.asarray(amplitudes)
        return values

    def solve(
        self,
        *,
        continuation=(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
        residual_tolerance_Ha=1e-10,
        root_separation_Ha=1e-8,
        minimum_overlap=0.7,
        minimum_reference_weight=1e-6,
    ):
        """Track the reference-connected root of the non-Hermitian SD pencil.

        Homotopy scales source, energy coupling and the off-diagonal/genuine
        interaction part relative to diagonal Fock gaps. This is a branch
        prescription, not perturbative truncation. Reject ambiguous/complex,
        isolated-reference-loss roots and small fixed-energy denominators;
        no branch substitution, level shift or symmetrization is performed.
        """
        self._assert_current()
        tolerance = _positive_real(residual_tolerance_Ha, "residual tolerance")
        separation = _positive_real(root_separation_Ha, "root separation")
        overlap_gate = _positive_real(minimum_overlap, "minimum overlap")
        reference_gate = _positive_real(
            minimum_reference_weight, "minimum reference weight"
        )
        if overlap_gate >= 1 or reference_gate >= 1:
            raise ValueError("overlap and reference weight must be less than one")
        stages = tuple(continuation)
        if (
            not stages
            or any(
                isinstance(x, (bool, np.bool_))
                or np.ndim(x) != 0
                or not np.isrealobj(x)
                or not np.isfinite(x)
                or not 0 < x <= 1
                for x in stages
            )
            or stages[-1] != 1
            or any(a >= b for a, b in zip(stages, stages[1:], strict=False))
        ):
            raise ValueError(
                "strict increasing finite continuation ending at one required"
            )
        diagonal = np.diag(np.r_[0.0, self.diagonal_gaps_Ha])
        if (
            len(self.diagonal_gaps_Ha)
            and np.min(abs(self.diagonal_gaps_Ha)) <= separation
        ):
            raise ValueError(
                "reference is degenerate with an SD excitation; enlarge model space"
            )
        pencil = np.column_stack((self.source_Ha, self.linear_map_Ha)).copy()
        pencil[0, 0] -= self.orbital_energy_Ha
        previous = np.zeros(len(diagonal))
        previous[0] = 1
        path = []
        for stage in stages:
            matrix = diagonal + stage * (pencil - diagonal)
            energies, vectors = np.linalg.eig(matrix)
            lengths = np.linalg.norm(vectors, axis=0)
            scores = abs(previous.conj() @ vectors) / lengths
            selected = int(np.argmax(scores))
            sorted_scores = np.sort(scores)
            if (
                scores[selected] < overlap_gate
                or len(scores) > 1
                and sorted_scores[-1] - sorted_scores[-2] < 1e-8
            ):
                raise RuntimeError(
                    "ambiguous SD continuation root; refine declared branch path"
                )
            energy, vector = energies[selected], vectors[:, selected]
            if abs(energy.imag) > tolerance or np.max(abs(vector.imag)) > tolerance:
                raise RuntimeError(
                    "complex SD continuation root in real bound-state anchor"
                )
            distance = min(
                (
                    abs(energy - other)
                    for i, other in enumerate(energies)
                    if i != selected
                ),
                default=np.inf,
            )
            weight = abs(vector[0]) ** 2 / lengths[selected] ** 2
            if distance <= separation or weight < reference_gate:
                raise RuntimeError(
                    "SD root separation/reference projection failed; enlarge model space"
                )
            vector = vector.real / vector[0].real
            previous = vector / np.linalg.norm(vector)
            delta = float(energy.real)
            if (
                len(self.diagonal_gaps_Ha)
                and np.min(abs(self.diagonal_gaps_Ha - delta)) <= separation
            ):
                raise RuntimeError("small SD energy denominator; no shift applied")
            defect = float(np.max(abs(matrix @ vector - delta * vector)))
            if defect > tolerance or not np.all(np.isfinite(vector)):
                raise RuntimeError("raw SD continuation pencil residual failed")
            path.append(
                (
                    float(stage),
                    delta,
                    float(scores[selected]),
                    float(weight),
                    float(distance),
                    defect,
                )
            )
        amplitudes = vector[1:]
        raw = self.residual(amplitudes, delta)
        direct = self.linked_action(amplitudes)
        direct[0] -= self.orbital_energy_Ha + delta
        direct[1:] -= delta * amplitudes
        if max(np.max(abs(raw)), np.max(abs(direct))) > tolerance:
            raise RuntimeError(
                "full signed monovalent SD energy/projected residual failed"
            )
        state = {self.reference_mask: 1.0}
        for mask, value in apply_terms(
            self.core_terms + self._valence_terms(amplitudes), state
        ).items():
            state[mask] = state.get(mask, 0) + value
        norm = sum(value * value for value in state.values())
        state = {
            mask: value / np.sqrt(norm) for mask, value in state.items() if value != 0
        }
        hstate = apply_terms(self._hterms, state)
        total = self._core_result.total_energy_Ha + self.orbital_energy_Ha + delta
        full = max(
            abs(hstate.get(mask, 0) - total * state.get(mask, 0))
            for mask in hstate.keys() | state.keys()
        )
        expectation = sum(value * hstate.get(mask, 0) for mask, value in state.items())
        masks = tuple(sorted(state))
        result = FiniteMonovalentSDResult(
            self.valence_index,
            self.orbital_energy_Ha,
            delta,
            self.orbital_energy_Ha + delta,
            total,
            amplitudes,
            masks,
            np.array([state[mask] for mask in masks]),
            float(norm),
            float(np.max(abs(raw[1:]))) if len(raw) > 1 else 0.0,
            float(abs(raw[0])),
            float(full),
            float(expectation),
            tuple(path),
            dict(
                method="finite standalone monovalent linked SD",
                integral_binding=self.integrals.identity["binding"],
                source_sha256=_source_sha256,
                expanded_core_residual_Ha=self.expanded_core_residual_Ha,
                core_amplitudes_sha256=hashlib.sha256(
                    self._core_result.amplitudes
                ).hexdigest(),
                valence_terms_sha256=hashlib.sha256(
                    repr(self.excitations).encode()
                ).hexdigest(),
                representation="independent signed magnetic excitation coefficients",
                equation="core-normal-ordered standalone SD Eq.8/9/13; Q residual=deltaE_v S_v; fixed core",
                physical_state="normalized (1+T_core+S_v)|Phi_v>; Eq.6 SD, no exponential products",
                resources=dict(self.plan),
                numerical_residual_is_accuracy_bound=False,
                omitted="nonlinear cluster products, triples, Breit/QED/recoil, production J reduction, experimental/space convergence",
            ),
            self,
        )
        self._solutions.append(result)
        return result


def one_body_matrix_element(bra, ket, matrix):
    """Normalized signed SD-state bilinear, not a CC left-response property.

    Matrix is the caller's explicitly chosen spinor one-body operator; both
    states must retain this same native integral owner and spinor phases.
    No bare-operator replacement of the correlated states is performed.
    """
    if not isinstance(bra, FiniteMonovalentSDResult) or not isinstance(
        ket, FiniteMonovalentSDResult
    ):
        raise TypeError("owned finite monovalent SD states required")
    for state in (bra, ket):
        owner = state._owner
        owner._assert_current()
        if not any(state is result for result in owner._solutions):
            raise ValueError(
                "state does not have its actual owner solution certificate"
            )
    if bra._owner.integrals is not ket._owner.integrals:
        raise ValueError(
            "same native integral/spinor phase owner required at both endpoints"
        )
    n = len(bra._owner.integrals.labels)
    operator = np.asarray(matrix)
    if (
        operator.shape != (n, n)
        or not np.issubdtype(operator.dtype, np.number)
        or not np.all(np.isfinite(operator))
    ):
        raise ValueError("complete finite spinor one-body operator required")
    terms = tuple(
        (operator[p, q], ((p, True), (q, False)))
        for p, q in itertools.product(range(n), repeat=2)
        if operator[p, q] != 0
    )
    acted = apply_terms(
        terms, dict(zip(ket.determinants, ket.coefficients, strict=True))
    )
    return sum(
        coefficient * acted.get(mask, 0)
        for mask, coefficient in zip(bra.determinants, bra.coefficients, strict=True)
    )
