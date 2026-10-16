"""Source-bound fixed-energy reduced CI+all-order two-valence screening.

Complete original Coulomb diagrams and the irreducible scalar Fov term retain
all declared core/virtual channels. Directed values are not assumed Hermitian.
This module does not construct a CI metric or solve physical excitation roots.
"""

import ctypes
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from atomx.angular.recoupling import l_of_kappa
from atomx.angular.sd_rho import kappa_from_partial, rho_coefficient
from atomx.angular.wigner import wigner_3j
from atomx.methods.valence_sd import (
    ValenceSD,
    ValenceSDResult,
    _valence_channels,
    plan_valence_sd,
)
from atomx.methods.valence_sd import (
    _source_identity as _valence_source_identity,
)

__all__ = ["Sigma2SD", "Sigma2SDResult", "plan_sigma2_sd"]
_pin = "828ea5d34c70e63bb21ceca5f6e6652da7dc3007ec9ba1021d0d6835c9726105"
_source_identity = {
    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    for name in ("sigma2_sd.py", "_pci_sigma2.f90")
}


def _pair_channels(kv, kw, occupied, lmax):
    vw, cn = [], []
    for im in range(2 * lmax + 1):
        km = kappa_from_partial(im + 1)
        for jn in range(2 * lmax + 1):
            kn = kappa_from_partial(jn + 1)
            for k in range(
                max(abs(abs(km) - abs(kv)), abs(abs(kn) - abs(kw))),
                min(abs(km) + abs(kv) - 1, abs(kn) + abs(kw) - 1) + 1,
            ):
                if (l_of_kappa(km) + l_of_kappa(kv)) % 2 == (
                    l_of_kappa(kn) + l_of_kappa(kw)
                ) % 2:
                    vw.append((im, jn, k))
        for c, (_, kc) in enumerate(occupied):
            for k in range(
                max(abs(abs(kc) - abs(kv)), abs(abs(km) - abs(kw))),
                min(abs(kc) + abs(kv) - 1, abs(km) + abs(kw) - 1) + 1,
            ):
                if (
                    not (l_of_kappa(kc) + l_of_kappa(kv) + k) % 2
                    and not (l_of_kappa(km) + l_of_kappa(kw) + k) % 2
                ):
                    cn.append((c, im, k))
    return tuple(vw), len(cn)


def plan_sigma2_sd(
    core_plan, valence_keys, *, valence_term2_blas=False, valence_term2_threads=1
):
    """Count complete reduced screening before native image allocation.

    Radial columns share kappa angular templates. External valence columns are
    explicit; every core/virtual intermediate from the core plan remains. The
    complete upper bound includes retained source images, screened tensor copies
    and wrapper scratch, but excludes interpreter/JIT/runtime RSS.
    """
    val = plan_valence_sd(
        core_plan,
        valence_keys,
        valence_term2_blas=valence_term2_blas,
        valence_term2_threads=valence_term2_threads,
    )
    keys = val["valence_keys"]
    d = core_plan["dimensions"]
    lmax = (d["NK"] - 1) // 2
    occupied = core_plan["occupied_keys"]
    kappas = {key[1] for key in keys}
    templates = {
        (kv, kw): _pair_channels(kv, kw, occupied, lmax)
        for kv in kappas
        for kw in kappas
    }
    sizes = dict(
        NZ=len(keys),
        NXV=val["nxv"],
        NVH=val["nvh"],
        NVH2=sum(len(val["channels"][key]) for key in keys),
        NVWH=1 + sum(len(templates[a[1], b[1]][0]) for a in keys for b in keys),
        NXCN=1 + sum(templates[a[1], b[1]][1] for a in keys for b in keys),
        NXV2=1
        + sum(
            max(
                len(_valence_channels(occupied, key[1], lmax, kind))
                for kind in ("x1", "x2")
            )
            for key in keys
        ),
    )
    if max(sizes.values()) > np.iinfo(np.int32).max:
        raise MemoryError("complete screening exceeds native int32 channel dimensions")
    # Each padded array in original/radial/response/wrapper/owned-output work is
    # covered by a conservative 32-array bound. The separate lookup bound uses
    # exact dimension products; none of these terms is a magnetic quartic ERI.
    radial = 8 * (
        32 * d["NXX"] ** 2 * sum(sizes[name] for name in ("NVWH", "NVH2", "NXV2"))
        + 32 * d["NXX"] * sizes["NXCN"]
        + 24 * d["NHF"] * (d["NX"] ** 2 + d["NX"] * sizes["NZ"])
        + 32 * d["NX"] * sizes["NZ"]
    )
    lookup = (
        4
        * 8
        * (d["KK"] + 1)
        * (
            sizes["NZ"] ** 2 * d["NK"] * (d["NK"] + d["NN"])
            + sizes["NZ"] * d["NN"] * d["NK"] ** 2
            + sizes["NVWH"]
            + sizes["NVH2"]
            + sizes["NXCN"]
        )
    )
    packed_indices = 12 * sum(
        val["raw_amplitudes"][next(key for key in keys if key[1] == kappa)]
        for kappa in kappas
    )
    return MappingProxyType(
        dict(
            valence_keys=keys,
            sizes=MappingProxyType(sizes),
            angular_templates=MappingProxyType(
                {pair: rows for pair, (rows, _) in templates.items()}
            ),
            packed_column_indices_bound_bytes=packed_indices,
            screening_payload_bound_bytes=radial + lookup + packed_indices,
            complete_payload_bound_bytes=val["complete_payload_bound_bytes"]
            + radial
            + lookup
            + packed_indices,
            scope="complete retained core/valence/screening static arrays, scratch and owned screening outputs; add simultaneous column solve Krylov storage explicitly; excludes interpreter/JIT/runtime RSS",
        )
    )


def _owned(array):
    a = np.asarray(array, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _s1(rank, left, right):
    a, b = abs(left), abs(right)
    return (
        2
        * (-1) ** a
        * np.sqrt(a * b)
        * wigner_3j(2 * a - 1, 2 * b - 1, 2 * rank, -1, 1, 0)
    )


@dataclass(frozen=True, eq=False)
class Sigma2SDResult:
    """Immutable directed reduced values in original pCI S1 normalization.

    `direct_element` converts a complete signed magnetic element using the
    stored basis and normalization, without imposing particle-exchange or
    adjoint symmetry. Joint response varies all stored column energies together.
    """

    valence_keys: tuple
    virtual_keys: tuple
    groups: tuple
    channels: tuple
    energy_parameters_Ha: np.ndarray
    radial_Ha: np.ndarray
    joint_derivative: object
    provenance: object

    def __post_init__(self):
        object.__setattr__(
            self, "energy_parameters_Ha", _owned(self.energy_parameters_Ha)
        )
        object.__setattr__(self, "radial_Ha", _owned(self.radial_Ha))
        if self.joint_derivative is not None:
            object.__setattr__(self, "joint_derivative", _owned(self.joint_derivative))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(
            self,
            "_positions",
            MappingProxyType(
                {
                    key: (p, i)
                    for p, g in enumerate(self.groups)
                    for i, key in enumerate(g)
                }
            ),
        )
        object.__setattr__(
            self,
            "_columns",
            MappingProxyType({key: i for i, key in enumerate(self.valence_keys)}),
        )
        object.__setattr__(
            self,
            "_channels",
            MappingProxyType({row: i for i, row in enumerate(self.channels)}),
        )

    def direct_element(self, left1, left2, right1, right2, *, derivative=False):
        """Signed unantisymmetrized rho element for four bound magnetic labels."""
        if not isinstance(derivative, bool):
            raise ValueError("derivative selection must be bool")
        labels = tuple(tuple(x) for x in (left1, left2, right1, right2))
        if any(
            len(x) != 3
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                for v in x
            )
            or x[:2] not in self.virtual_keys
            or x[2] not in range(-2 * abs(x[1]) + 1, 2 * abs(x[1]), 2)
            for x in labels
        ):
            raise ValueError("complete bound virtual magnetic labels required")
        a, b, v, w = labels
        if v[:2] not in self._columns or w[:2] not in self._columns:
            raise ValueError(
                "right labels must belong to the declared solved valence columns"
            )
        radial = self.joint_derivative if derivative else self.radial_Ha
        if radial is None:
            raise ValueError("joint response was not computed")
        im, m = self._positions[a[:2]]
        jn, n = self._positions[b[:2]]
        iv, iw = self._columns[v[:2]], self._columns[w[:2]]
        value = 0.0
        for k in range(
            max(abs(abs(a[1]) - abs(v[1])), abs(abs(b[1]) - abs(w[1]))),
            min(abs(a[1]) + abs(v[1]) - 1, abs(b[1]) + abs(w[1]) - 1) + 1,
        ):
            c = self._channels.get((iv, iw, im, jn, k))
            if c is not None:
                value += (
                    rho_coefficient(k, a, b, v, w)
                    * (-1) ** k
                    * _s1(k, a[1], v[1])
                    * _s1(k, b[1], w[1])
                    * radial[m, n, c]
                )
        return float(value)


class Sigma2SD:
    """Complete fixed-core screening on source-bound solved valence columns.

    Every call validates the actual supplied root/response with the same valence
    operator, restores core COMMON and serializes access using the core lock.
    No input column, intermediate, rank or small amplitude is omitted.
    """

    def __init__(self, valence_problem):
        if not isinstance(valence_problem, ValenceSD):
            raise TypeError("source-bound native ValenceSD problem required")
        self.valence = valence_problem
        self.core = valence_problem.core
        self.plan = plan_sigma2_sd(
            self.core.plan,
            self.valence.plan["valence_keys"],
            valence_term2_blas=self.valence.plan["valence_term2_blas"],
            valence_term2_threads=self.valence.plan["valence_term2_threads"],
        )
        if self.plan["complete_payload_bound_bytes"] > self.core.memory_bytes:
            raise MemoryError(
                "complete core/valence/screening image exceeds memory cap"
            )
        m = self.core._manifest.get("sigma2_sd")
        if (
            not isinstance(m, dict)
            or m.get("source_sha256") != _pin
            or m.get("template_sha256") != _source_identity["_pci_sigma2.f90"]
            or m.get("sizes") != dict(self.plan["sizes"])
        ):
            raise ValueError(
                "screening backend/source/dimensions do not match complete plan"
            )
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")
        self._init = self.core._backend.atomx_vw_init
        self._init.argtypes = [ctypes.c_int, iptr, ptr]
        self._init.restype = None
        self._load = self.core._backend.atomx_vw_load
        self._load.argtypes = [ctypes.c_int, ctypes.c_int, iptr, ptr, ptr]
        self._load.restype = None

    def evaluate(self, columns, *, joint_response=True):
        """Full directed screening and optional total common-energy derivative.

        Supplied column derivatives must include their implicit SD response.
        The core remains fixed, and every column-energy slope is exactly one.
        This derivative is not standalone pCI's single stored derivative field.
        """
        if not isinstance(joint_response, bool):
            raise ValueError("joint response selection must be bool")
        columns = tuple(columns)
        keys = self.plan["valence_keys"]
        if (
            len(columns) != len(keys)
            or any(not isinstance(c, ValenceSDResult) for c in columns)
            or len({c.column_key for c in columns}) != len(keys)
            or {c.column_key for c in columns} != set(keys)
        ):
            raise ValueError(
                "every declared distinct source-bound valence result required"
            )
        by_key = {c.column_key: c for c in columns}
        columns = tuple(by_key[key] for key in keys)
        if self.plan["complete_payload_bound_bytes"] > self.core.memory_bytes:
            raise MemoryError("complete screening payload exceeds current cap")
        for c in columns:
            if (
                any(
                    c.provenance.get(k) != v
                    for k, v in self.core.provenance.items()
                    if k != "method"
                )
                or c.provenance.get("method")
                != "fixed-energy CI+all-order monovalent linearized SD, full scalar Fock"
                or c.provenance.get("valence_source_sha256") != _valence_source_identity
                or c.provenance.get("core_identity_sha256")
                != self.valence._core_identity
                or c.provenance.get("core_amplitudes_sha256")
                != self.valence._core_amplitudes_sha256
                or c.provenance.get("equation_coverage")
                != self.core._manifest["valence_sd"]
            ):
                raise ValueError("valence result physical/source/core identity differs")
            if c.channels != self.valence.plan["channels"][c.column_key]:
                raise ValueError("complete valence channel identity differs")
            if joint_response and c.numerator_derivative is None:
                raise ValueError("complete implicit numerator response required")
        with self.core._lock:
            # Validate all roots before initializing screening, because a valence
            # operation restores/mutates the owning legacy core COMMON image.
            packed_by_kappa = {}
            for c in columns:
                context = self.valence._initialize(c.column_key, c.energy_parameter_Ha)
                mapped = self.valence._evaluate(
                    c.energy_parameter_Ha,
                    c.numerators,
                    c.numerator_derivative if joint_response else None,
                    context,
                )
                packed_by_kappa.setdefault(c.column_key[1], context[3])
                if np.max(abs(mapped[1] - c.numerators)) > 1e-10:
                    raise ValueError(
                        "actual full valence root residual fails screening gate"
                    )
                if (
                    joint_response
                    and np.max(abs(mapped[3] - c.numerator_derivative)) > 1e-10
                ):
                    raise ValueError(
                        "actual full implicit response residual fails screening gate"
                    )
            dims = self.core.plan["dimensions"]
            sizes = self.plan["sizes"]
            energies = np.asfortranarray([c.energy_parameter_Ha for c in columns])
            self._init(
                len(keys),
                np.asfortranarray(np.asarray(keys).T, dtype=np.int32),
                energies,
            )
            for i, c in enumerate(columns):
                # The lock prevents intervening COMMON mutation. Validation
                # restored the full core; reuse its checked packed ordering
                # without rebuilding native valence X1/X2 a second time.
                index = packed_by_kappa[c.column_key[1]]
                full = np.zeros((dims["NXX"], dims["NXX"], len(c.channels)), order="F")
                prime = np.zeros_like(full, order="F")
                full[index] = c.numerators
                if joint_response:
                    prime[index] = c.numerator_derivative
                channels = np.asfortranarray(
                    np.asarray(c.channels).T + 1, dtype=np.int32
                )
                channels[3] -= 1
                self._load(i + 1, len(c.channels), channels, full, prime)
            shape = (dims["NXX"], dims["NXX"], sizes["NVWH"] + 1)
            count = np.zeros(1, dtype=np.int32)
            rows = np.zeros((5, sizes["NVWH"]), dtype=np.int32, order="F")
            values = np.zeros(shape, order="F")
            prime = np.zeros_like(values, order="F")
            self.core._call("atomx_vw_evaluate_joint", [count, rows, values, prime])
            before_count = int(count[0])
            before_rows = rows.copy()
            fock = np.zeros_like(values, order="F")
            fock_prime = np.zeros_like(values, order="F")
            self.core._call("atomx_vw_fock", [count, rows, fock, fock_prime])
            if int(count[0]) != before_count or not np.array_equal(rows, before_rows):
                raise ValueError("Coulomb/Fov complete channel identity differs")
            actual = tuple(
                tuple(int(v) for v in row)
                for row in (rows[:, :before_count].T - np.asarray([1, 1, 1, 1, 0]))
            )
            expected = tuple(
                (iv, iw, *row)
                for iv, v in enumerate(keys)
                for iw, w in enumerate(keys)
                for row in self.plan["angular_templates"][v[1], w[1]]
            )
            if actual != expected:
                raise ValueError(
                    "native screening channel order differs from complete angular plan"
                )
            values += fock
            if joint_response:
                prime += fock_prime
            if (
                not np.all(np.isfinite(values))
                or joint_response
                and not np.all(np.isfinite(prime))
            ):
                raise ValueError("nonfinite native screening value or joint response")
            provenance = dict(
                self.core.provenance,
                method="fixed-energy CI+all-order reduced Sigma2",
                core_identity_sha256=self.valence._core_identity,
                core_amplitudes_sha256=self.valence._core_amplitudes_sha256,
                valence_source_sha256=dict(
                    columns[0].provenance["valence_source_sha256"]
                ),
                screening_source_sha256=dict(_source_identity),
                equation_coverage=dict(self.core._manifest["sigma2_sd"]),
                column_numerator_sha256=[
                    hashlib.sha256(c.numerators.tobytes()).hexdigest() for c in columns
                ],
                column_response_sha256=[
                    hashlib.sha256(c.numerator_derivative.tobytes()).hexdigest()
                    for c in columns
                ]
                if joint_response
                else None,
                response="all column energy slopes1, screening explicit slopes2, fixed core, full implicit valence response"
                if joint_response
                else "not requested",
                complete_payload_bound_bytes=self.plan["complete_payload_bound_bytes"],
                energy_reference="same native core Fock orbital zero; explicit column parameters",
                normalization="original parity-free S1; direct element restores (-1)^k S1(m,v) S1(n,w)",
                symmetry="directed values; no approximate index symmetry, Hermitian averaging or extrapolation",
                space_converged=False,
                forward_accuracy_certified=False,
                experiment_validated=False,
                ci_metric_constructed=False,
            )
            return Sigma2SDResult(
                keys,
                tuple(self.core.plan["virtual_keys"]),
                tuple(self.core.plan["groups"]),
                actual,
                energies,
                values[:, :, 1 : before_count + 1],
                prime[:, :, 1 : before_count + 1] if joint_response else None,
                provenance,
            )
