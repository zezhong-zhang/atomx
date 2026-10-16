"""Native direct reduced closed-core linearized SD with pinned pCI diagrams.

Original reduced Coulomb diagrams consume an exact factorization of the
shared native cell integrator. Full scalar noncanonical Fock contributions
are retained. This real DC closed-core method is not valence SD, screened
CI+all-order, nonlinear CCSD, FS-CCSDT or a spectroscopic accuracy guarantee.
Case-specific external GPL backend builds carry source/ABI identities.
"""

import ctypes
import hashlib
import json
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
from atomx.angular.sd_rho import _import_source_sha256 as _angular_source_sha256
from atomx.angular.sd_rho import kappa_from_partial, partial_index, rho_coefficient
from atomx.methods.linearized_sd import plan_scalar_sd
from atomx.numerics.bspline.coulomb_features import (
    CellCoulombFeatures,
    PolynomialCellCoulombFeatures,
)
from atomx.numerics.bspline.coulomb_features import (
    _import_source_sha256 as _feature_source_sha256,
)
from atomx.physics.csf.spinor_integrals import build_spinor_integrals

_live = weakref.WeakValueDictionary()
_registry_lock = RLock()
_pin = "4d685d82bfdf27876098a87808dadc5fcf3038d1"
_source_identity = MappingProxyType(
    {
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sd_rho.py": _angular_source_sha256,
        "coulomb_features.py": _feature_source_sha256,
    }
)


def _owned(a):
    a = np.asarray(a, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _channels(occupied, lmax, kind):
    rows = []
    for i, (_, ki) in enumerate(occupied):
        for j, (_, kj) in enumerate(occupied[: i + 1]):
            for im in range(1, 2 * lmax + 2):
                km = kappa_from_partial(im)
                for inn in range(1, 2 * lmax + 2):
                    kn = kappa_from_partial(inn)
                    a, b, c, d = (km, ki, kn, kj) if kind != "x2" else (km, kn, ki, kj)
                    low = max(abs(abs(a) - abs(b)), abs(abs(c) - abs(d)))
                    high = min(abs(a) + abs(b) - 1, abs(c) + abs(d) - 1)
                    for rank in range(low, high + 1):
                        parity = (
                            l_of_kappa(a) + l_of_kappa(b) + rank,
                            l_of_kappa(c) + l_of_kappa(d) + rank,
                        )
                        allowed = (
                            parity[0] % 2 == parity[1] % 2
                            if kind == "rho"
                            else not parity[0] % 2 and not parity[1] % 2
                        )
                        if allowed:
                            rows.append((i, j, im - 1, inn - 1, rank))
    return tuple(rows)


def plan_core_sd(
    occupied_keys,
    virtual_keys,
    feature_identity,
    *,
    radial_pair_cache_size=16384,
    term21_threads=0,
    term21_mode="compact",
    core_singles_blas=False,
    core_single_to_doubles_blas=False,
):
    """Complete redundant rho storage and conservative payload preflight.

    Unlike the independent SU(2) amplitude count, pCI rho_k contains direct
    particle-hole exchange redundancy. Only core/padding slots are omitted
    from the Krylov vector; all requested virtuals and multipoles remain.
    Estimates exclude interpreter/JIT/runtime RSS and are not timing claims.
    """
    if (
        isinstance(radial_pair_cache_size, bool)
        or not isinstance(radial_pair_cache_size, (int, np.integer))
        or radial_pair_cache_size < 1
    ):
        raise ValueError("positive integer radial pair cache size required")
    if (
        isinstance(term21_threads, bool)
        or not isinstance(term21_threads, (int, np.integer))
        or term21_threads < 0
    ):
        raise ValueError("nonnegative integer term21 worker count required")
    if term21_mode not in ("compact", "streamed"):
        raise ValueError("term21 mode must be compact or streamed")
    if not isinstance(core_singles_blas, bool) or (
        core_singles_blas and not term21_threads
    ):
        raise ValueError(
            "BLAS core singles requires a boolean and positive worker count"
        )
    if not isinstance(core_single_to_doubles_blas, bool) or (
        core_single_to_doubles_blas and not term21_threads
    ):
        raise ValueError(
            "BLAS single-to-doubles requires a boolean and positive workers"
        )
    scalar = plan_scalar_sd(occupied_keys, virtual_keys)
    occupied, virtual = scalar.occupied_keys, scalar.virtual_keys
    keys = occupied + virtual
    lmax = max(l_of_kappa(k) for _, k in keys)
    groups = tuple(
        tuple(a for a in keys if a[1] == kappa_from_partial(index))
        for index in range(1, 2 * lmax + 2)
    )
    counts = tuple(sum(a in virtual for a in g) for g in groups)
    nxx, nk, nn = max(map(len, groups)), 2 * lmax + 1, len(occupied)
    nhf = feature_identity["auxiliary_nodes"]
    nv, ng = feature_identity["quadrature_nodes"]
    kk = max(nk, 2 * max(abs(k) for _, k in keys) - 1)
    rows = _channels(occupied, lmax, "rho")
    nch, nxk = (
        len(rows),
        max(len(_channels(occupied, lmax, "x1")), len(_channels(occupied, lmax, "x2"))),
    )
    dims = dict(
        NHF=nhf,
        NX=max(nxx, nk),
        NS=nn,
        KX1=nk,
        NXX=nxx,
        NK=nk,
        NN=nn,
        KK=kk,
        NCH=nch,
        NXK=nxk,
        NV=nv,
        NG=ng,
        NE=feature_identity["same_cell_edges"],
    )
    raw = sum(counts[partial_index(int(k)) - 1] for _, k in occupied)
    raw += sum(counts[im] * counts[inn] for _, _, im, inn, _ in rows)
    nx = dims["NX"]
    commons = 8 * (
        2 * nhf * nx * nk
        + 2 * nhf * nn
        + 3 * nhf
        + nx * nk
        + 3 * nxx * nxx * nch
        + 2 * nxx * nxx * nxk
        + 4 * nxx * nn
        + (2 * nv * ng + dims["NE"]) * (kk + 1)
        + 2 * nx * nx * nk
        + 2 * nk * nk * (kk + 1)
    )
    lookup = 4 * (3 * nn * nn * nk * nk * (kk + 1) + 5 * nch + nx + 5 * nn) + 8 * (
        2 * nv * ng + 2 * dims["NE"]
    )
    # Bound all retained original scratch routines, including term21 and
    # term2 (compiled as an alternative anchor), rather than just one vector.
    scratch = 8 * (
        2 * (nhf * nx * nx + 8 * nhf * nx + 2 * nhf * nxx * (kk + 1) * kk)
        + 64 * nhf
        + 32 * nx * nx
        + 16 * nxx * nn
    )
    initialization = 8 * (2 * nhf * nx * nk + (2 * nv * ng + dims["NE"]) * (kk + 1))
    nmag = sum(2 * abs(k) for _, k in keys)
    pair_cache = (
        int(radial_pair_cache_size)
        * 16
        * max(feature_identity["coefficient_sizes"])
        * max(feature_identity["component_orders"])
    )
    # The optional exact BLAS initializer needs three NHF-by-NXX blocks.
    # Account for it even when the original initializer is selected.
    blas_initialization = 24 * nhf * nxx
    term21_threads = min(int(term21_threads), nn * (nn + 1) // 2)
    compact_worker = 8 * nhf * (nxx**2 + nxx * (kk + 1) * nk + 4 * nxx + 3) + 16 * nv
    compact_scratch = term21_threads * compact_worker + (
        4 * nn * (nn + 1) if term21_threads else 0
    )
    if term21_threads and term21_mode == "streamed":
        compact_worker = 8 * (2 * nhf * nxx + nxx**2 + 2 * nhf) + 16 * nv
        compact_scratch = (
            8 * nhf * (nxx**2 + nx * nk)
            + term21_threads * compact_worker
            + 4 * nn * (nn + 1)
        )
    singles_worker = 8 * (nhf * nxx + nxx**2 + 2 * nhf)
    singles_scratch = (
        8 * nhf * (nx * nk + nxx) + term21_threads * singles_worker
        if core_singles_blas
        else 0
    )
    single_to_doubles_scratch = (
        8 * nhf * (nxx * (nk + 1 + term21_threads) + nn + 2 * term21_threads)
        + 16 * nv * term21_threads
        if core_single_to_doubles_blas
        else 0
    )
    bound = (
        commons
        + lookup
        + scratch
        + initialization
        + 64 * nmag * nmag
        + 40 * raw
        + feature_identity["payload_estimate_bytes"]
        + pair_cache
        + blas_initialization
        + compact_scratch
        + singles_scratch
        + single_to_doubles_scratch
    )
    return dict(
        dimensions=dims,
        occupied_keys=occupied,
        virtual_keys=virtual,
        groups=groups,
        channels=rows,
        raw_amplitudes=raw,
        independent_scalar_amplitudes=scalar.amplitudes,
        complete_payload_bound_bytes=bound,
        krylov_vector_bytes=8 * raw,
        radial_pair_cache_size=int(radial_pair_cache_size),
        radial_pair_cache_array_bound_bytes=pair_cache,
        blas_initialization_scratch_bytes=blas_initialization,
        term21_threads=term21_threads,
        term21_mode=term21_mode if term21_threads else "original",
        term21_worker_scratch_bytes=compact_worker if term21_threads else 0,
        term21_parallel_scratch_bytes=compact_scratch,
        core_singles_blas=core_singles_blas,
        core_singles_worker_scratch_bytes=singles_worker if core_singles_blas else 0,
        core_singles_parallel_scratch_bytes=singles_scratch,
        core_single_to_doubles_blas=core_single_to_doubles_blas,
        core_single_to_doubles_scratch_bytes=single_to_doubles_scratch,
        scope="bound for complete static arrays/scratch and initialization payload; add requested Krylov storage; excludes JIT/interpreter/runtime RSS",
    )


@dataclass(frozen=True)
class CoreSDResult:
    reference_energy_Ha: float
    correlation_energy_Ha: float
    total_energy_Ha: float
    amplitudes: np.ndarray
    residual_max_Ha: float
    residual_l2_Ha: float
    matrix_vector_products: int
    wall_seconds: float
    provenance: object

    def __post_init__(self):
        object.__setattr__(self, "amplitudes", _owned(self.amplitudes))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class CoreSD:
    """Owned native core SD problem; the external COMMON instance is isolated.

    Preparation checks the native metric/integral binding, full backend
    dimensions and file hashes. Frozen core, Breit/QED, complex spinors and
    open/valence sectors are unsupported and must not be silently supplied.
    Each live problem requires its own backend build path, because pCI uses
    mutable Fortran COMMON arrays. Calls within one problem are serialized.

    Physical direct-rho inputs obey simultaneous electron-pair exchange.
    In particular, identical occupied radial holes and identical virtual
    angular groups require a symmetric virtual radial block. Raw redundant
    coordinates outside that domain are numerical auxiliary inputs; shape
    and finiteness checks alone do not certify a physical cluster operator.
    """

    def __init__(
        self,
        orbitals,
        workspace,
        Z,
        ns,
        *,
        occupied_keys,
        virtual_keys,
        backend_directory,
        nuclear_potential=None,
        memory_bytes=2 * 1024**3,
        feature_memory_bytes=128 * 1024**2,
        radial_pair_cache_size=16384,
        term21_threads=0,
        term21_mode="compact",
        core_singles_blas=False,
        core_single_to_doubles_blas=False,
        radial_representation="native",
    ):
        if (
            isinstance(memory_bytes, bool)
            or not isinstance(memory_bytes, int)
            or memory_bytes < 1
        ):
            raise ValueError("positive integer complete memory cap required")
        if radial_representation not in ("native", "polynomial"):
            raise ValueError("radial representation must be native or polynomial")
        features_type = (
            CellCoulombFeatures
            if radial_representation == "native"
            else PolynomialCellCoulombFeatures
        )
        self.features = features_type(workspace, memory_bytes=feature_memory_bytes)
        plan = plan_core_sd(
            occupied_keys,
            virtual_keys,
            self.features.identity,
            radial_pair_cache_size=radial_pair_cache_size,
            term21_threads=term21_threads,
            term21_mode=term21_mode,
            core_singles_blas=core_singles_blas,
            core_single_to_doubles_blas=core_single_to_doubles_blas,
        )
        plan["dimensions"] = MappingProxyType(dict(plan["dimensions"]))
        self.plan = MappingProxyType(plan)
        if self.plan["complete_payload_bound_bytes"] > memory_bytes:
            raise MemoryError("complete reduced SD payload exceeds declared memory cap")
        self.memory_bytes = memory_bytes
        keys = self.plan["occupied_keys"] + self.plan["virtual_keys"]
        if any(
            not np.isrealobj(x) or not np.all(np.isfinite(x))
            for a in keys
            for x in orbitals[a]
        ):
            raise ValueError("native real finite DC spinors required")
        directory = Path(backend_directory).resolve(strict=True)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["pin"] != _pin or manifest["dimensions"] != self.plan["dimensions"]:
            raise ValueError(
                "backend pin/complete dimensions do not match native problem"
            )
        term21 = manifest.get("term21", dict(mode="original pCI term21", threads=0))
        expected_mode = {
            "original": "original pCI term21",
            "compact": "exact component-tagged core-pair term21",
            "streamed": "exact shared virtual-pair streamed term21",
        }[self.plan["term21_mode"]]
        if (
            term21.get("threads") != self.plan["term21_threads"]
            or term21.get("mode") != expected_mode
        ):
            raise ValueError("backend term21 workers/mode do not match resource plan")
        if (
            (
                self.plan["term21_mode"] == "streamed"
                or core_singles_blas
                or core_single_to_doubles_blas
            )
            and self.plan["term21_threads"] > 1
            and (
                not isinstance(manifest.get("blas"), dict)
                or manifest["blas"].get("runtime", {}).get("parallel") != 2
            )
        ):
            raise ValueError("parallel SD backend lacks a bound OpenMP BLAS runtime")
        singles = manifest.get(
            "core_singles", dict(mode="original pCI terma4", workers=0)
        )
        if singles.get("mode") != (
            "exact component-tagged BLAS terma4"
            if core_singles_blas
            else "original pCI terma4"
        ) or singles.get("workers") != (
            self.plan["term21_threads"] if core_singles_blas else 0
        ):
            raise ValueError(
                "backend core singles mode/workers do not match resource plan"
            )
        if core_singles_blas and not isinstance(manifest.get("blas"), dict):
            raise ValueError("BLAS core singles lacks a bound BLAS runtime")
        single_to_doubles = manifest.get(
            "core_single_to_doubles", dict(mode="original pCI term31/32", workers=0)
        )
        if single_to_doubles.get("mode") != (
            "exact native weighted BLAS term31/32"
            if core_single_to_doubles_blas
            else "original pCI term31/32"
        ) or single_to_doubles.get("workers") != (
            self.plan["term21_threads"] if core_single_to_doubles_blas else 0
        ):
            raise ValueError(
                "backend single-to-doubles mode/workers do not match resource plan"
            )
        if core_single_to_doubles_blas and not isinstance(manifest.get("blas"), dict):
            raise ValueError("BLAS single-to-doubles lacks a bound BLAS runtime")
        for filename, expected in manifest["files"].items():
            if (
                Path(filename).name != filename
                or hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError("backend artifact hash failed")
        # Bind the imported code, including for later problems in a long-lived
        # process whose on-disk files may have changed during earlier runs.
        direct_sources = dict(_source_identity)
        self._lock = RLock()
        self._blas_runtime = None
        self._path = str(directory / "native_core_sd.so")
        with _registry_lock:
            if self._path in _live:
                raise ValueError(
                    "live Fortran COMMON instance already owned; build another case path"
                )
            self._backend = ctypes.CDLL(self._path)
            blas = manifest.get("blas")
            if blas is not None and "runtime" in blas:
                configuration = self._backend.openblas_get_config
                configuration.restype = ctypes.c_char_p
                parallel = self._backend.openblas_get_parallel
                parallel.restype = ctypes.c_int
                threads = self._backend.openblas_get_num_threads
                threads.restype = ctypes.c_int
                actual = dict(
                    config=configuration().decode(),
                    parallel=parallel(),
                    threads=threads(),
                )
                if (
                    actual["config"] != blas["runtime"]["config"]
                    or actual["parallel"] != blas["runtime"]["parallel"]
                ):
                    raise ValueError(
                        "loaded BLAS model differs from the bound build runtime"
                    )
                if (
                    (
                        self.plan["term21_mode"] == "streamed"
                        or core_singles_blas
                        or core_single_to_doubles_blas
                    )
                    and self.plan["term21_threads"] > 1
                    and (actual["parallel"] != 2 or actual["threads"] != 1)
                ):
                    raise ValueError(
                        "parallel SD requires OpenMP BLAS pinned to one inner thread"
                    )
                self._blas_runtime = actual
            _live[self._path] = self
        self._manifest = manifest
        try:
            self.integrals = build_spinor_integrals(
                orbitals,
                workspace,
                Z,
                ns,
                occupied_keys=self.plan["occupied_keys"],
                virtual_keys=self.plan["virtual_keys"],
                nuclear_potential=nuclear_potential,
                representation="on_demand",
                max_integral_bytes=memory_bytes,
                radial_pair_cache_size=int(radial_pair_cache_size),
            )
            self._prepare(orbitals)
        except BaseException:
            with _registry_lock:
                if _live.get(self._path) is self:
                    del _live[self._path]
            raise
        count = self.plan["raw_amplitudes"]
        self.source, self.denominators = np.zeros(count), np.zeros(count)
        self._vector_call("atomx_sd_vector_info", [self.source, self.denominators])
        if not np.all(np.isfinite(self.source)):
            raise ValueError("nonfinite native core excitation source")
        if not np.all(np.isfinite(self.denominators)) or np.any(self.denominators == 0):
            raise ValueError("singular/nonfinite native core excitation denominator")
        self.source, self.denominators = _owned(self.source), _owned(self.denominators)
        self.provenance = dict(
            self.integrals.identity,
            backend_pin=_pin,
            backend_sha256=manifest["files"]["native_core_sd.so"],
            initialization=manifest.get("initialization", "original pCI rhocore"),
            blas=manifest.get("blas"),
            core_single_to_doubles=single_to_doubles,
            blas_runtime=self._blas_runtime,
            term21=term21,
            core_singles=singles,
            feature_identity=dict(self.features.identity),
            method="closed-core linearized SD, native DC",
            equation="Q_SD(H+[H,T1+T2])|Phi>=0; full noncanonical scalar Fock",
            amplitude_representation="original redundant direct rho_k, active virtual slots only",
            resources={
                k: dict(v) if k == "dimensions" else v
                for k, v in self.plan.items()
                if k not in ("channels", "groups")
            },
        )
        self.provenance["direct_sd_source_sha256"] = direct_sources
        self._matvecs = 0
        self._last_solution_certificate = None

    def _solution_identity(self, result):
        """Exact current owner data; contiguous buffer hashes allocate no copies."""
        action = self.action
        action_identity = (
            getattr(action, "__func__", action),
            id(getattr(action, "__self__", None)),
        )
        return (
            hashlib.sha256(
                json.dumps(
                    self.provenance, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            hashlib.sha256(self.source).hexdigest(),
            hashlib.sha256(self.denominators).hexdigest(),
            hashlib.sha256(result.amplitudes).hexdigest(),
            (
                result.reference_energy_Ha,
                result.correlation_energy_Ha,
                result.total_energy_Ha,
                result.residual_max_Ha,
                result.residual_l2_Ha,
            ),
            hashlib.sha256(
                json.dumps(
                    dict(result.provenance), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            action_identity,
        )

    def _has_current_solution_certificate(self, result, tolerance_Ha):
        """Reuse only this owner's most recent immutable, measured raw result."""
        certificate = self._last_solution_certificate
        return (
            certificate is not None
            and certificate[0]() is result
            and np.isfinite(result.residual_max_Ha)
            and result.residual_max_Ha <= tolerance_Ha
            and certificate[1] == self._solution_identity(result)
        )

    def _call(self, name, arrays):
        function = getattr(self._backend, name)
        function.argtypes = [
            np.ctypeslib.ndpointer(dtype=a.dtype, flags="F_CONTIGUOUS") for a in arrays
        ]
        function.restype = None
        function(*arrays)

    def _vector_call(self, name, arrays):
        function = getattr(self._backend, name)
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        function.argtypes = [ctypes.c_int64] + [pointer] * len(arrays)
        function.restype = None
        function(self.plan["raw_amplitudes"], *arrays)

    def _prepare(self, orbitals):
        d = self.plan["dimensions"]
        groups = self.plan["groups"]
        keys = self.plan["occupied_keys"] + self.plan["virtual_keys"]
        slots = {a: g.index(a) for g in groups for a in g}
        labels = self.integrals.labels
        p, q = (np.zeros((d["NHF"], d["NX"], d["KX1"]), order="F") for _ in range(2))
        f = np.zeros((d["NX"], d["NX"], d["KX1"]), order="F")
        eps = np.zeros((d["NX"], d["KX1"]), order="F")
        # Separate component node tags; unequal kinetic-balance orders can
        # have unequal node counts, so half of NHF is not a valid split.
        first_q = int(self.features.outer_indices[1, 0, 0])
        for a in keys:
            value = self.features.sample(*orbitals[a])
            s = slots[a]
            ind = partial_index(int(a[1])) - 1
            p[:first_q, s, ind] = value[:first_q]
            q[first_q:, s, ind] = value[first_q:]
        for ind, g in enumerate(groups):
            for a in g:
                ma = labels.index((*a, -twoj_of_kappa(a[1])))
                for b in g:
                    mb = labels.index((*b, -twoj_of_kappa(b[1])))
                    z = self.integrals.fock_Ha[ma, mb]
                    if z.imag != 0:
                        raise ValueError("real scalar native Fock required")
                    f[slots[a], slots[b], ind] = z.real
                eps[slots[a], ind] = f[slots[a], slots[a], ind]
        ranks = [self.features.rank_arrays(k) for k in range(d["KK"] + 1)]
        rw, fw = (
            np.asfortranarray(np.stack([r[i] for r in ranks], axis=2)) for i in (0, 1)
        )
        ew = np.asfortranarray(np.stack([r[2] for r in ranks], axis=1))

        def ints(a):
            return np.asfortranarray(a, dtype=np.int32)

        lengths = np.zeros(d["NX"], dtype=np.int32)
        lengths[: len(groups)] = [len(g) for g in groups]
        occupied = self.plan["occupied_keys"]
        self._call(
            "atomx_sd_init",
            [
                p,
                q,
                eps,
                ints([slots[a] + 1 for a in occupied]),
                ints([a[1] for a in occupied]),
                lengths,
                np.asfortranarray(self.features.radii),
                ints(self.features.outer_indices + 1),
                ints(self.features.edge_left + 1),
                ints(self.features.edge_right + 1),
                rw,
                fw,
                ew,
            ],
        )
        alpha, beta = (
            np.zeros((d["NK"], d["NK"], d["KK"] + 1), order="F") for _ in range(2)
        )
        for im in range(d["NK"]):
            ki = kappa_from_partial(im + 1)
            mi = -twoj_of_kappa(ki)
            for inn in range(d["NK"]):
                kj = kappa_from_partial(inn + 1)
                for rank in range(d["KK"] + 1):
                    for mj in range(-twoj_of_kappa(kj), twoj_of_kappa(kj) + 1, 2):
                        a, b = (1, ki, mi), (1, kj, mj)
                        alpha[im, inn, rank] += rho_coefficient(rank, a, b, a, b)
                        beta[im, inn, rank] += rho_coefficient(rank, b, a, a, b)
        self._call("atomx_sd_set_fock", [f, alpha, beta])

    def action(self, amplitudes):
        """Raw redundant-rho equation action K*t, before source subtraction.

        A physical projected-commutator interpretation requires the direct-rho
        pair-exchange domain documented by CoreSD. This entry point does not
        project or validate that domain for an arbitrary numerical vector.
        """
        t = np.asarray(amplitudes)
        if (
            t.shape != self.source.shape
            or not np.isrealobj(t)
            or not np.all(np.isfinite(t))
        ):
            raise ValueError("finite complete real reduced amplitude vector required")
        t = np.asfortranarray(t, dtype=float)
        result = np.empty_like(t)
        with self._lock:
            self._vector_call("atomx_sd_vector_action", [t, result])
            self._matvecs += 1
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite native core SD equation action")
        return result

    def set_diagram_progress(self, enabled=True):
        """Print native per-diagram times to stderr without changing equations.

        Disabled by default. Old backend builds require rebuilding to expose
        the progress control; silently ignoring the request is unsupported.
        """
        if not isinstance(enabled, bool):
            raise ValueError("boolean diagram progress flag required")
        with self._lock:
            try:
                function = self._backend.atomx_sd_set_progress
            except AttributeError as error:
                raise ValueError(
                    "backend build lacks diagram progress control"
                ) from error
            function.argtypes = [ctypes.c_int]
            function.restype = None
            function(int(enabled))

    def energy(self, amplitudes):
        if not np.isrealobj(amplitudes):
            raise ValueError("real reduced amplitude vector required")
        t = np.asfortranarray(amplitudes, dtype=float)
        if t.shape != self.source.shape or not np.all(np.isfinite(t)):
            raise ValueError("finite complete reduced amplitude vector required")
        result = np.zeros(1)
        with self._lock:
            self._vector_call("atomx_sd_vector_energy", [t, result])
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite native core SD energy")
        return float(result[0])

    def solve(
        self,
        *,
        residual_tolerance_Ha=1e-10,
        restart=30,
        maxiter=100,
        initial_amplitudes=None,
    ):
        """Matrix-free restarted GMRES; measured raw residual is authoritative.

        Storage includes the full declared radial pool. Diagonal
        preconditioning changes iteration only; it removes no coordinates.
        The raw rho residual is distinct from correlation-space convergence.
        An optional complete signed initial vector is only an iterative seed:
        the current operator, source, energy and final raw gate are re-evaluated.
        Its hash is retained; no prior result identity is silently transferred.
        Physical seed interpretation additionally requires the direct-rho
        pair-exchange domain. Arbitrary numerical seeds receive shape and
        finiteness checks, not a separate physical-domain certificate.
        """
        if not np.isfinite(residual_tolerance_Ha) or residual_tolerance_Ha <= 0:
            raise ValueError("positive finite physical residual tolerance required")
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x < 1
            for x in (restart, maxiter)
        ):
            raise ValueError(
                "positive integer Krylov restart and iteration cap required"
            )
        n = len(self.source)
        extra = 8 * n * (min(restart, n) + 10)
        if self.plan["complete_payload_bound_bytes"] + extra > self.memory_bytes:
            raise MemoryError(
                "complete native SD and requested Krylov storage exceed memory cap"
            )
        if initial_amplitudes is None:
            initial = self.source / self.denominators
            initial_kind = "complete source divided by diagonal denominators"
        else:
            initial = np.asarray(initial_amplitudes)
            if (
                initial.shape != self.source.shape
                or not np.isrealobj(initial)
                or not np.all(np.isfinite(initial))
            ):
                raise ValueError(
                    "finite complete real initial amplitude vector required"
                )
            initial = np.array(initial, dtype=np.float64, copy=True)
            initial_kind = "explicit complete signed iterative seed"
        initial_sha256 = hashlib.sha256(initial.tobytes()).hexdigest()
        started = time.perf_counter()
        before = self._matvecs
        operator = LinearOperator((n, n), matvec=self.action, dtype=float)
        preconditioner = LinearOperator(
            (n, n), matvec=lambda v: v / self.denominators, dtype=float
        )
        t, info = gmres(
            operator,
            self.source,
            M=preconditioner,
            x0=initial,
            atol=residual_tolerance_Ha * 0.1,
            rtol=0.0,
            restart=restart,
            maxiter=maxiter,
        )
        residual = self.action(t) - self.source
        maximum = float(np.max(abs(residual)))
        l2 = float(np.linalg.norm(residual))
        if info != 0 or maximum > residual_tolerance_Ha:
            raise RuntimeError(
                f"native core SD residual gate failed: info={info}, max={maximum} Ha"
            )
        correlation = self.energy(t)
        reference = float(self.integrals.reference_energy_Ha)
        provenance = dict(
            self.provenance,
            solver="matrix-free restarted GMRES with full diagonal preconditioner",
            krylov_restart=restart,
            maxiter=maxiter,
            requested_residual_tolerance_Ha=residual_tolerance_Ha,
            initial_amplitudes_kind=initial_kind,
            initial_amplitudes_sha256=initial_sha256,
            estimated_krylov_bytes=extra,
            space_converged=False,
            experiment_validated=False,
        )
        result = CoreSDResult(
            reference,
            correlation,
            reference + correlation,
            t,
            maximum,
            l2,
            self._matvecs - before,
            time.perf_counter() - started,
            provenance,
        )
        # This private certificate is never serialized or transferred. A new
        # result object, source/operator/provenance change or weaker raw gate
        # still requires a complete consumer-side action.
        self._last_solution_certificate = (
            weakref.ref(result),
            self._solution_identity(result),
        )
        return result
