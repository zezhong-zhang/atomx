"""Complete signed J-pair bridge for spherical SD Hamiltonians.

This bounded correctness interface accepts the public antisymmetric spinor
integrals, including full static or finite-frequency transverse exchange.
It supplies normalized antisymmetric pair matrices, not Coulomb RINT values
and not an extension of the existing DC-only pCI diagrams. The complete
magnetic reconstruction is checked before a result is returned.
"""

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from math import comb, sqrt
from types import MappingProxyType

import numpy as np

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
from atomx.methods.linearized_sd import _coupled_pairs
from atomx.methods.linearized_sd import solve as solve_linearized_sd
from atomx.physics.csf.spinor_integrals import (
    OnDemandSpinorIntegrals,
    SpinorIntegrals,
)


def _freeze(value):
    if hasattr(value, "items"):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _owned(value):
    a = np.asarray(value, dtype=np.complex128)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _layout(labels, noccupied, *, materialize_channels=True):
    labels = tuple(tuple(label) for label in labels)
    groups = {}
    for i, label in enumerate(labels):
        if (
            len(label) != 3
            or any(
                isinstance(x, bool) or not isinstance(x, (int, np.integer))
                for x in label
            )
            or label[0] < 1
            or label[1] == 0
        ):
            raise ValueError("integer (n,kappa,2m) spinor labels required")
        groups.setdefault(label[:2], {})[label[2]] = i
    if len(set(labels)) != len(labels):
        raise ValueError("distinct spinor labels required")
    for key, multiplet in groups.items():
        j = twoj_of_kappa(key[1])
        if set(multiplet) != set(range(-j, j + 1, 2)):
            raise ValueError("complete magnetic multiplets required")
        occupied = [i < noccupied for i in multiplet.values()]
        if any(occupied) and not all(occupied):
            raise ValueError("closed spherical occupied multiplets required")
    if not materialize_channels:
        return (
            labels,
            groups,
            {
                key: count
                for key, count in _coupled_pairs(tuple(groups)).items()
                if count
            },
        )
    channels = {}
    for a, b in combinations_with_replacement(groups, 2):
        ja, jb = twoj_of_kappa(a[1]), twoj_of_kappa(b[1])
        parity = (-1) ** (l_of_kappa(a[1]) + l_of_kappa(b[1]))
        for twoj in range(abs(ja - jb), ja + jb + 1, 2):
            if a != b or twoj % 4 == 0:
                channels.setdefault((twoj, parity), []).append((a, b))
    return labels, groups, {key: tuple(value) for key, value in channels.items()}


def plan_j_coupled_sd_interaction(integrals):
    """Bound the complete bridge before ERI evaluation; no magnetic tensor.

    Bounds cover retained numeric arrays and conservative Python/stencil
    allowances, excluding the already-owned integral factory and interpreter.
    The all-coordinate verification is deliberately finite; this bridge is
    not a claim of a scalable native all-order SD solver.
    """
    if not isinstance(integrals, (SpinorIntegrals, OnDemandSpinorIntegrals)):
        raise TypeError("public owned spinor integrals required")
    labels, groups, counts = _layout(
        integrals.labels, integrals.noccupied, materialize_channels=False
    )
    n, pairs = len(labels), comb(len(labels), 2)
    entries = sum(count**2 for count in counts.values())
    max_j_count = max(twoj_of_kappa(key[1]) for key in groups) + 1
    max_pair_terms = max(len(multiplet) for multiplet in groups.values()) ** 2
    work = entries * max_pair_terms**2 + pairs**2 * max_j_count
    payload = 16 * (entries + 2 * n * n) + 1024 * (
        n + pairs * max_j_count + sum(counts.values())
    )
    return dict(
        spinors=n,
        magnetic_pair_dimension=pairs,
        j_pair_coordinates=sum(counts.values()),
        j_block_entries=entries,
        working_payload_bound_bytes=payload,
        eri_action_bound=work,
        channels={str(key): count for key, count in counts.items()},
        scope="complete finite J-pair bridge; excludes owned integral cache and interpreter RSS",
    )


def _coefficient(labels, p, q, channel, twoj):
    """Coefficient of a†p a†q in the normalized CG pair, p < q."""
    a, b = channel
    left, right = labels[p], labels[q]
    if a == b:
        if left[:2] != a or right[:2] != b:
            return 0.0
        factor = sqrt(2.0)
    elif left[:2] == a and right[:2] == b:
        factor = 1.0
    elif left[:2] == b and right[:2] == a:
        left, right, factor = right, left, -1.0
    else:
        return 0.0
    return factor * clebsch_gordan(
        twoj_of_kappa(a[1]),
        left[2],
        twoj_of_kappa(b[1]),
        right[2],
        twoj,
        left[2] + right[2],
    )


def _stencil(labels, channel, twoj, twom):
    return tuple(
        (p, q, coefficient)
        for p, q in combinations(range(len(labels)), 2)
        if labels[p][2] + labels[q][2] == twom
        and (coefficient := _coefficient(labels, p, q, channel, twoj)) != 0
    )


@dataclass(frozen=True)
class JCoupledSDInteraction:
    """Immutable normalized J-pair Hamiltonian with its exact core ledger.

    Pair vectors use Condon--Shortley CG coefficients and the *input* spinor
    creation order, with 1/sqrt(2) for equal radial orbitals. No reduced C^k
    factors divide transverse integrals. ``blocks_Ha`` contain two-body
    matrix elements alone. ``hamiltonian_block`` uses h_eff, never Fock, so
    the occupied contraction is included exactly once.
    """

    labels: tuple
    noccupied: int
    channels: object
    blocks_Ha: object
    one_body_Ha: np.ndarray
    fock_Ha: np.ndarray
    frozen_energy_Ha: float
    reference_energy_Ha: float
    identity: object
    gates: object
    resources: object

    def __post_init__(self):
        for name in ("channels", "identity", "gates", "resources"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(
            self,
            "blocks_Ha",
            MappingProxyType(
                {key: _owned(value) for key, value in self.blocks_Ha.items()}
            ),
        )
        for name in ("one_body_Ha", "fock_Ha"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        index = {}
        for key, rows in self.channels.items():
            for i, row in enumerate(rows):
                index[key, row] = i
        object.__setattr__(self, "_index", MappingProxyType(index))
        order = {
            key: i for i, key in enumerate(dict.fromkeys(x[:2] for x in self.labels))
        }
        object.__setattr__(self, "_radial_order", MappingProxyType(order))

    def _pair(self, p, q):
        a, b = self.labels[p][:2], self.labels[q][:2]
        return (a, b) if self._radial_order[a] <= self._radial_order[b] else (b, a)

    def eri(self, p, q, r, s):
        """Reconstruct a signed magnetic ERI without a quartic tensor."""
        n = len(self.labels)
        if min(p, q, r, s) < 0 or max(p, q, r, s) >= n:
            raise IndexError("spinor index out of range")
        if p == q or r == s:
            return 0j
        sign = 1
        if p > q:
            p, q, sign = q, p, -sign
        if r > s:
            r, s, sign = s, r, -sign
        if (
            self.labels[p][2] + self.labels[q][2]
            != self.labels[r][2] + self.labels[s][2]
        ):
            return 0j
        ab, cd = self._pair(p, q), self._pair(r, s)
        parity = (-1) ** sum(l_of_kappa(key[1]) for key in ab)
        if parity != (-1) ** sum(l_of_kappa(key[1]) for key in cd):
            return 0j
        value = 0j
        ja, jb = (twoj_of_kappa(key[1]) for key in ab)
        for twoj in range(abs(ja - jb), ja + jb + 1, 2):
            block = (twoj, parity)
            left, right = self._index.get((block, ab)), self._index.get((block, cd))
            if left is not None and right is not None:
                value += (
                    _coefficient(self.labels, p, q, ab, twoj)
                    * self.blocks_Ha[block][left, right]
                    * _coefficient(self.labels, r, s, cd, twoj)
                )
        return sign * value

    def pair_vectors(self, twoj, parity, *, twom=None):
        """Complete normalized signed creation-pair stencils at any M."""
        twom = twoj if twom is None else twom
        if twom not in range(-twoj, twoj + 1, 2):
            raise ValueError("valid pair magnetic projection required")
        return tuple(
            _stencil(self.labels, row, twoj, twom)
            for row in self.channels[twoj, parity]
        )

    def hamiltonian_block(self, twoj, parity):
        """Full two-electron H, including effective h and frozen constant."""
        vectors = self.pair_vectors(twoj, parity)
        h = self.blocks_Ha[twoj, parity].copy()
        for i, left in enumerate(vectors):
            for j, right in enumerate(vectors):
                for p, q, a in left:
                    for r, s, b in right:
                        value = (
                            self.one_body_Ha[p, r] * (q == s)
                            - self.one_body_Ha[p, s] * (q == r)
                            - self.one_body_Ha[q, r] * (p == s)
                            + self.one_body_Ha[q, s] * (p == r)
                        )
                        h[i, j] += a * value * b
        h += np.eye(len(h)) * self.frozen_energy_Ha
        return _owned(h)


def build_j_coupled_sd_interaction(
    integrals,
    *,
    memory_bytes=128 * 1024**2,
    work_cap=20_000_000,
    tolerance_Ha=1e-11,
):
    """Compile and certify all signed J/parity pair blocks on one snapshot.

    Oversized whole spaces reject before ERI calls. All magnetic coordinates,
    Fock contractions, reference energy and one-body scalar conditions are
    checked. No amplitude/radial cutoff, tensor averaging or energy fitting.
    """
    for name, value in (("memory_bytes", memory_bytes), ("work_cap", work_cap)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"positive integer {name} required")
    if (
        isinstance(tolerance_Ha, bool)
        or not np.isscalar(tolerance_Ha)
        or not np.isrealobj(tolerance_Ha)
        or not np.isfinite(tolerance_Ha)
        or tolerance_Ha <= 0
    ):
        raise ValueError("positive finite tolerance required")
    resources = plan_j_coupled_sd_interaction(integrals)
    if resources["working_payload_bound_bytes"] > memory_bytes:
        raise MemoryError("complete J-pair bridge exceeds memory cap")
    if resources["eri_action_bound"] > work_cap:
        raise MemoryError("complete J-pair verification exceeds work cap")
    labels, groups, channels = _layout(integrals.labels, integrals.noccupied)
    blocks, norm_error = {}, 0.0
    for key, rows in channels.items():
        vectors = tuple(_stencil(labels, row, key[0], key[0]) for row in rows)
        for vector in vectors:
            norm_error = max(norm_error, abs(sum(c * c for _, _, c in vector) - 1))
        blocks[key] = np.array(
            [
                [
                    sum(
                        a * integrals.eri(p, q, r, s) * b
                        for p, q, a in left
                        for r, s, b in right
                    )
                    for right in vectors
                ]
                for left in vectors
            ],
            dtype=complex,
        )
    if norm_error > 1e-12:
        raise ValueError("normalized pair angular metric failed")
    hermitian = max(float(np.max(abs(a - a.conj().T))) for a in blocks.values())
    if hermitian > tolerance_Ha:
        raise ValueError("signed J-pair Hermiticity failed")
    identity = dict(
        representation="normalized antisymmetric J-pair blocks",
        spinor_integral_identity=dict(integrals.identity),
        pair_phase="Condon--Shortley CG; input spinor creation order; equal-radial 1/sqrt(2)",
        normal_ordering="h_eff=F-sum_occupied g; frozen constant retained exactly once",
        operator_coverage="same complete signed spinor Hamiltonian; existing pCI SD diagrams remain DC-only",
    )
    digest = hashlib.sha256(repr((labels, integrals.noccupied)).encode())
    for value in (
        integrals.one_body_Ha,
        integrals.fock_Ha,
        np.asarray([integrals.frozen_energy_Ha, integrals.reference_energy_Ha]),
    ):
        digest.update(value.tobytes())
    for key, value in sorted(blocks.items()):
        digest.update(repr((key, channels[key])).encode())
        digest.update(value.tobytes())
    digest.update(json.dumps(identity, sort_keys=True, default=repr).encode())
    identity["input_sha256"] = digest.hexdigest()
    result = JCoupledSDInteraction(
        labels,
        integrals.noccupied,
        channels,
        blocks,
        integrals.one_body_Ha,
        integrals.fock_Ha,
        integrals.frozen_energy_Ha,
        integrals.reference_energy_Ha,
        identity,
        {},
        resources,
    )
    eri_error = max(
        abs(result.eri(p, q, r, s) - integrals.eri(p, q, r, s))
        for p, q in combinations(range(len(labels)), 2)
        for r, s in combinations(range(len(labels)), 2)
    )
    scalar_error, fock_error = 0.0, 0.0
    for p, a in enumerate(labels):
        for q, b in enumerate(labels):
            f = result.one_body_Ha[p, q] + sum(
                result.eri(p, i, q, i) for i in range(result.noccupied)
            )
            fock_error = max(fock_error, abs(f - result.fock_Ha[p, q]))
            if a[1:] != b[1:]:
                scalar_error = max(
                    scalar_error,
                    abs(result.one_body_Ha[p, q]),
                    abs(result.fock_Ha[p, q]),
                )
            else:
                p0, q0 = (
                    groups[a[:2]][min(groups[a[:2]])],
                    groups[b[:2]][min(groups[b[:2]])],
                )
                scalar_error = max(
                    scalar_error,
                    abs(result.one_body_Ha[p, q] - result.one_body_Ha[p0, q0]),
                    abs(result.fock_Ha[p, q] - result.fock_Ha[p0, q0]),
                )
    o = result.noccupied
    energy = (
        result.frozen_energy_Ha
        + np.trace(result.one_body_Ha[:o, :o])
        + 0.5 * sum(result.eri(i, j, i, j) for i in range(o) for j in range(o))
    )
    energy_error = abs(energy - result.reference_energy_Ha)
    if max(eri_error, scalar_error, fock_error, energy_error) > tolerance_Ha:
        raise ValueError("complete signed J-pair/normal-ordering reconstruction failed")
    object.__setattr__(
        result,
        "gates",
        _freeze(
            dict(
                complete_magnetic_eri_max_Ha=float(eri_error),
                scalar_one_body_max_Ha=float(scalar_error),
                fock_contraction_max_Ha=float(fock_error),
                reference_energy_error_Ha=float(energy_error),
                pair_norm_error=norm_error,
                hermitian_max_Ha=hermitian,
                tolerance_Ha=float(tolerance_Ha),
            )
        ),
    )
    return result


def solve_two_electron_core_sd(
    interaction,
    *,
    reference_identity,
    memory_bytes=128 * 1024**2,
    residual_tolerance_Ha=1e-10,
):
    """Complete finite closed two-electron J=0 linked SD on this Hamiltonian.

    Every scalar single/double in the declared virtual pool is included. This
    provides a dynamic-transverse *finite anchor*, not large native pCI SD,
    multivalent/monovalent equations, nonlinear CCSD or FS-CCSDT.
    """
    if not isinstance(interaction, JCoupledSDInteraction) or not interaction.gates:
        raise TypeError("certified owned J-pair interaction required")
    if (
        interaction.noccupied != 2
        or interaction.labels[0][:2] != interaction.labels[1][:2]
    ):
        raise ValueError("complete single j=1/2 two-electron core required")
    vectors = interaction.pair_vectors(0, 1)
    rows = interaction.channels[0, 1]
    core = interaction.labels[0][:2]
    ref_index = rows.index((core, core))
    order = (ref_index,) + tuple(i for i in range(len(rows)) if i != ref_index)
    h = interaction.hamiltonian_block(0, 1)[np.ix_(order, order)]
    if np.max(abs(h.imag)) > 1e-14 or np.max(abs(h - h.T)) > 1e-11:
        raise ValueError("finite linked SD requires real symmetric Hamiltonian")
    projection = float(np.max(abs(h.real - h.real.T))) / 2
    h = (h.real + h.real.T) / 2
    signed = tuple(
        tuple(((1 << p) | (1 << q), c) for p, q, c in vectors[i]) for i in order
    )
    result = solve_linearized_sd(
        h,
        signed,
        3,
        reference_identity=reference_identity,
        memory_bytes=memory_bytes,
        residual_tolerance_Ha=residual_tolerance_Ha,
    )
    object.__setattr__(
        result,
        "provenance",
        _freeze(
            dict(
                result.provenance,
                interaction_identity=interaction.identity,
                hermitian_roundoff_projection_max_Ha=projection,
                scope="complete finite two-electron scalar SD anchor; native pCI transverse diagrams unfinished",
            )
        ),
    )
    return result
