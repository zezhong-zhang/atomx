"""Native fixed-energy CI+all-order monovalent SD and Sigma1 response.

This composes one owned CoreSD COMMON image and a source-bound pinned valence
backend. It preserves complete radial channels, signed numerators and scalar
noncanonical Fock blocks. A fixed-energy column is an effective-operator input,
not an eigenenergy or a spectral accuracy guarantee. Sigma2, Breit, QED and
valence-singles/energy feedback are outside this interface's equation coverage.
"""

import ctypes
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from atomx.angular.recoupling import l_of_kappa
from atomx.angular.sd_rho import kappa_from_partial
from atomx.methods.core_sd import CoreSD, CoreSDResult, _owned

_valence_pin = "6bf791841f1d9f6c0e0c2fd176d4517007ac622d2aea69fd1278107a414ab07b"
_source_identity = MappingProxyType(
    {
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "_pci_valence.f90": hashlib.sha256(
            Path(__file__).with_name("_pci_valence.f90").read_bytes()
        ).hexdigest(),
        "_pci_valence_term2_streamed.f90": hashlib.sha256(
            Path(__file__).with_name("_pci_valence_term2_streamed.f90").read_bytes()
        ).hexdigest(),
    }
)


def _valence_channels(occupied, kappa, lmax, kind):
    rows = []
    for i, (_, ki) in enumerate(occupied):
        for im in range(1, 2 * lmax + 2):
            km = kappa_from_partial(im)
            for jn in range(1, 2 * lmax + 2):
                kn = kappa_from_partial(jn)
                a, b, c, d = (
                    (km, kappa, kn, ki) if kind != "x2" else (km, kn, kappa, ki)
                )
                low = max(abs(abs(a) - abs(b)), abs(abs(c) - abs(d)))
                high = min(abs(a) + abs(b) - 1, abs(c) + abs(d) - 1)
                for rank in range(low, high + 1):
                    p, q = (
                        l_of_kappa(a) + l_of_kappa(b) + rank,
                        l_of_kappa(c) + l_of_kappa(d) + rank,
                    )
                    allowed = (
                        p % 2 == q % 2 if kind == "rho" else not p % 2 and not q % 2
                    )
                    if allowed:
                        rows.append((i, im - 1, jn - 1, rank))
    return tuple(rows)


def plan_valence_sd(
    core_plan, valence_keys, *, valence_term2_blas=False, valence_term2_threads=1
):
    """Complete column channels and conservative retained-core memory bound.

    The backend allocates the largest column's arrays; one column at a time
    is solved. No magnetic quartic tensor or all-column amplitude cube is
    materialized. Requested Krylov/response storage is checked at solve time.
    """
    if not isinstance(valence_term2_blas, bool):
        raise ValueError("explicit boolean valence term2 BLAS policy required")
    if (
        isinstance(valence_term2_threads, bool)
        or not isinstance(valence_term2_threads, (int, np.integer))
        or valence_term2_threads < 1
        or not valence_term2_blas
        and valence_term2_threads != 1
    ):
        raise ValueError(
            "positive valence term2 worker count; original path requires one"
        )
    valence_term2_threads = int(valence_term2_threads)
    keys = tuple(tuple(k) for k in valence_keys)
    virtual = core_plan["virtual_keys"]
    if not keys or len(set(keys)) != len(keys) or any(k not in virtual for k in keys):
        raise ValueError(
            "distinct valence keys in the complete core virtual pool required"
        )
    d, groups = core_plan["dimensions"], core_plan["groups"]
    lmax = (d["NK"] - 1) // 2
    occupied = core_plan["occupied_keys"]
    # Angular channel structure is identical for radial columns of one kappa.
    kinds = {
        (kappa, kind): _valence_channels(occupied, kappa, lmax, kind)
        for kappa in {key[1] for key in keys}
        for kind in ("rho", "x1", "x2")
    }
    rows = {key: kinds[key[1], "rho"] for key in keys}
    nxv = max(len(kinds[key[1], kind]) for key in keys for kind in ("x1", "x2"))
    nvh = max(map(len, rows.values()))
    counts = tuple(sum(k in virtual for k in g) for g in groups)
    amplitudes = {
        key: sum(counts[im] * counts[jn] for _, im, jn, _ in rows[key]) for key in keys
    }
    extra = 8 * (
        d["NXX"] ** 2 * (24 * nvh + 2 * nxv)
        + d["NHF"] * d["NX"] ** 2 * 24
        + 20 * d["NX"]
    )
    term2_scratch = (
        8
        * (
            d["NHF"] * (d["NX"] * d["NK"] + d["NXX"] ** 2)
            + valence_term2_threads
            * (d["NHF"] * (4 * d["NXX"] + 2) + 4 * d["NXX"] ** 2)
        )
        if valence_term2_blas
        else 0
    )
    extra += term2_scratch
    return MappingProxyType(
        dict(
            valence_keys=keys,
            channels=MappingProxyType(rows),
            nxv=nxv,
            nvh=nvh,
            raw_amplitudes=MappingProxyType(amplitudes),
            valence_payload_bound_bytes=extra,
            valence_term2_blas=valence_term2_blas,
            valence_term2_threads=valence_term2_threads,
            valence_term2_scratch_bytes=term2_scratch,
            complete_payload_bound_bytes=core_plan["complete_payload_bound_bytes"]
            + extra,
            scope="complete retained core plus original valence static/scratch and vector payload; add per-column Krylov/response storage; excludes interpreter/JIT/runtime RSS",
        )
    )


def _identity(provenance):
    return hashlib.sha256(
        json.dumps(dict(provenance), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ValenceSDResult:
    column_key: tuple
    energy_parameter_Ha: float
    orbital_energy_Ha: float
    row_keys: tuple
    channels: tuple
    numerators: np.ndarray
    denominators_Ha: np.ndarray
    sigma1_Ha: np.ndarray
    residual_max_Ha: float
    numerator_derivative: object
    sigma1_derivative: object
    derivative_residual_max_Ha: object
    matrix_vector_products: int
    wall_seconds: float
    provenance: object

    def __post_init__(self):
        for field in (
            "numerators",
            "denominators_Ha",
            "sigma1_Ha",
            "numerator_derivative",
            "sigma1_derivative",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _owned(value))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class ValenceSD:
    """Owned fixed-core valence columns with complete scalar noncanonical Fock.

    The core result must bind the identical problem and satisfy its actual raw
    residual. All calls share the core instance lock. Initializing a column
    restores its core numerators, since a core action mutates the same COMMON.
    The core/valence amplitudes are read-only snapshots; no energy fitting is
    performed. Raw residual checks do not supply a forward-error bound.
    """

    def __init__(self, core_problem, core_result, *, valence_keys):
        if not isinstance(core_problem, CoreSD) or not isinstance(
            core_result, CoreSDResult
        ):
            raise TypeError(
                "owned CoreSD problem and source-bound CoreSDResult required"
            )
        self.core = core_problem
        manifest = self.core._manifest.get("valence_sd")
        term2 = (
            manifest.get("term2", {"mode": "original"})
            if isinstance(manifest, dict)
            else {}
        )
        if term2.get("mode") not in ("original", "streamed_blas"):
            raise ValueError(
                "declared original/streamed valence term2 backend required"
            )
        streamed = term2["mode"] == "streamed_blas"
        self.plan = plan_valence_sd(
            self.core.plan,
            valence_keys,
            valence_term2_blas=streamed,
            valence_term2_threads=term2.get("threads", 1),
        )
        if streamed and (
            term2.get("template_sha256")
            != _source_identity["_pci_valence_term2_streamed.f90"]
            or term2.get("scratch_bytes") != self.plan["valence_term2_scratch_bytes"]
            or term2.get("replaced_calls") != 3
            or term2.get("value_only_calls") != 1
            or term2.get("response_policy")
            != "explicit requested derivative; one shared value kernel"
        ):
            raise ValueError(
                "complete streamed valence term2 source/scratch identity differs"
            )
        if streamed and (
            self.core._blas_runtime is None
            or self.plan["valence_term2_threads"] > 1
            and (
                self.core._blas_runtime["parallel"] != 2
                or self.core._blas_runtime["threads"] != 1
            )
        ):
            raise ValueError("parallel valence SD requires bound inner-one OpenMP BLAS")
        if self.plan["complete_payload_bound_bytes"] > self.core.memory_bytes:
            raise MemoryError("complete retained core/valence image exceeds memory cap")
        if any(
            core_result.provenance.get(k) != v for k, v in self.core.provenance.items()
        ):
            raise ValueError(
                "core result Hamiltonian/basis/backend/source identity differs"
            )
        if (
            not np.all(np.isfinite(core_result.amplitudes))
            or core_result.amplitudes.shape != self.core.source.shape
        ):
            raise ValueError("finite complete source-bound core amplitudes required")
        if not self.core._has_current_solution_certificate(core_result, 1e-10):
            residual = self.core.action(core_result.amplitudes) - self.core.source
            if np.max(abs(residual)) > 1e-10 or not np.all(np.isfinite(residual)):
                raise ValueError(
                    "bound core result fails actual full raw residual gate"
                )
        manifest = self.core._manifest.get("valence_sd")
        if (
            not isinstance(manifest, dict)
            or manifest.get("source_sha256") != _valence_pin
            or manifest.get("template_sha256") != _source_identity["_pci_valence.f90"]
            or manifest.get("nxv") != self.plan["nxv"]
            or manifest.get("nvh") != self.plan["nvh"]
        ):
            raise ValueError(
                "complete valence backend/source/dimensions do not match plan"
            )
        self._core_amplitudes = _owned(core_result.amplitudes)
        self._core_identity = _identity(self.core.provenance)
        self._core_amplitudes_sha256 = hashlib.sha256(
            self._core_amplitudes.tobytes()
        ).hexdigest()
        self._ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        self._init = self.core._backend.atomx_val_init
        self._init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, self._ptr]
        self._init.restype = None
        self._mapping = self.core._backend.atomx_val_map_full_fock
        self._linear = self.core._backend.atomx_val_linear_fock
        for function in (self._mapping, self._linear):
            function.argtypes = [ctypes.c_double] + [self._ptr] * 8
            function.restype = None
        self._linear_value = self.core._backend.atomx_val_linear_value_fock
        self._linear_value.argtypes = [ctypes.c_double] + [self._ptr] * 4
        self._linear_value.restype = None
        self._matvecs = 0

    def _initialize(self, key, energy):
        key = tuple(key)
        if (
            key not in self.plan["valence_keys"]
            or isinstance(energy, (bool, np.bool_))
            or not np.isscalar(energy)
            or not np.isrealobj(energy)
            or not np.isfinite(energy)
        ):
            raise ValueError(
                "declared valence column and finite energy parameter required"
            )
        if _identity(self.core.provenance) != self._core_identity:
            raise ValueError(
                "core source/physical identity changed after valence binding"
            )
        if self.plan["complete_payload_bound_bytes"] > self.core.memory_bytes:
            raise MemoryError("complete core/valence payload exceeds current cap")
        groups, dims = self.core.plan["groups"], self.core.plan["dimensions"]
        partial = next(i for i, group in enumerate(groups) if key in group)
        slot = groups[partial].index(key)
        self._init(
            slot + 1,
            key[1],
            len(self._core_amplitudes),
            np.asfortranarray(self._core_amplitudes),
        )
        rows = self.plan["channels"][key]
        actual = np.zeros((4, self.plan["nvh"]), dtype=np.int32, order="F")
        source = np.zeros((dims["NXX"], dims["NXX"], self.plan["nvh"]), order="F")
        orbital = np.zeros(1)
        self.core._call("atomx_val_info", [actual, source, orbital])
        expected = np.asarray(rows).T + 1
        expected[3] -= 1
        count = np.zeros(1, dtype=np.int32)
        self.core._call("atomx_val_count", [count])
        if count[0] != len(rows) or not np.array_equal(
            actual[:, : len(rows)], expected
        ):
            raise RuntimeError("complete native valence channel ordering differs")
        virtual = self.core.plan["virtual_keys"]
        # Complete packed coordinates, 12 bytes per amplitude. Avoid millions
        # of Python tuples and retain original channel / Fortran radial order.
        count = self.plan["raw_amplitudes"][key]
        packed = np.empty((3, count), dtype=np.int32)
        active = tuple(
            np.asarray([i for i, k in enumerate(group) if k in virtual], dtype=np.int32)
            for group in groups
        )
        offset = 0
        for c, (_, im, jn, _) in enumerate(rows):
            m, n = active[im], active[jn]
            size = len(m) * len(n)
            packed[0, offset : offset + size] = np.tile(m, len(n))
            packed[1, offset : offset + size] = np.repeat(n, len(m))
            packed[2, offset : offset + size] = c
            offset += size
        if offset != count:
            raise RuntimeError("complete packed radial coordinates disagree with plan")
        index = tuple(packed)
        labels = self.core.integrals.labels
        eps = self.core.integrals.fock_Ha.diagonal().real
        by_key = {label[:2]: float(eps[i]) for i, label in enumerate(labels)}
        occupied = self.core.plan["occupied_keys"]
        row_table = np.asarray(rows, dtype=np.int32)
        core_eps = np.asarray([by_key[k] for k in occupied])
        orbital_eps = np.zeros((dims["NX"], dims["NK"]))
        for im, group in enumerate(groups):
            orbital_eps[: len(group), im] = [by_key[k] for k in group]
        bc, imc, jnc = row_table[index[2], :3].T
        denominator = (
            energy
            + core_eps[bc]
            - orbital_eps[index[0], imc]
            - orbital_eps[index[1], jnc]
        )
        if not np.all(np.isfinite(denominator)) or np.any(denominator == 0):
            raise ValueError("singular/nonfinite complete valence denominator")
        work = tuple(np.zeros(source.shape, order="F") for _ in range(4))
        singles = tuple(np.zeros(dims["NX"]) for _ in range(3))
        return (
            partial,
            rows,
            source.shape,
            index,
            denominator,
            float(orbital[0]),
            (work, singles),
        )

    def _load_inputs(self, vector, prime, context):
        _, _, _, index, _, _, (work, singles) = context
        count = len(index[0])
        if vector is None or any(
            a is not None
            and (
                np.asarray(a).shape != (count,)
                or not np.isrealobj(a)
                or not np.all(np.isfinite(a))
            )
            for a in (vector, prime)
        ):
            raise ValueError("finite complete real numerator/response vector required")
        d, dd, rd, pd = work
        d[index] = vector
        dd[index] = 0 if prime is None else prime
        s, rs, ps = singles
        return index, work, (s, rs, ps)

    def _evaluate(self, energy, vector, prime, context, *, linear=False):
        index, (d, dd, rd, pd), (s, rs, ps) = self._load_inputs(vector, prime, context)
        function = self._linear if linear else self._mapping
        function(energy, s, d, s, dd, rs, rd, ps, pd)
        if not all(np.all(np.isfinite(a)) for a in (rs, rd, ps, pd)):
            raise RuntimeError("nonfinite complete native valence action")
        # Own the small singles outputs before the next call overwrites work.
        return rs.copy(), rd[index], ps.copy(), pd[index]

    def _evaluate_value(self, energy, vector, context):
        """Exact homogeneous values only; no derivative is returned or implied."""
        index, (d, _, rd, _), (s, rs, _) = self._load_inputs(vector, None, context)
        self._linear_value(energy, s, d, rs, rd)
        if not all(np.all(np.isfinite(a)) for a in (rs, rd)):
            raise RuntimeError("nonfinite complete native valence value action")
        return rs.copy(), rd[index]

    def evaluate(self, key, energy_Ha, numerators, *, numerator_derivative=None):
        """Full signed fixed-energy map and total supplied energy derivative.

        Returns complete native partial-wave singles and packed doubles for
        the declared column. This does not add bare F_mv to correlation Sigma1.
        """
        with self.core._lock:
            context = self._initialize(key, energy_Ha)
            return tuple(
                _owned(a)
                for a in self._evaluate(
                    energy_Ha, numerators, numerator_derivative, context
                )
            )

    def solve(
        self,
        key,
        energy_Ha,
        *,
        solve_response=True,
        residual_tolerance_Ha=1e-10,
        restart=30,
        maxiter=100,
        initial_numerators=None,
        initial_numerator_derivative=None,
        iteration_callback=None,
    ):
        """Complete matrix-free column and optional implicit numerator response.

        Source-free contractions avoid cancellation when constructing I-L(E).
        Three actual-operator refinement solves reduce Krylov residual error;
        raw residual and response gates retain full original coordinates.
        The core is fixed for this derivative. Correlation-space/forward-error
        convergence, an effective-Hamiltonian metric and eigenvalue feedback
        require additional independent checks.
        Optional complete real numerator arrays are numerical initial guesses;
        every result is re-certified by the current full physical operator.
        Checkpoint consumers must separately bind their state/input identity.
        An optional callback receives a read-only complete numerical guess at
        each GMRES restart, including current source/coordinate identity.
        These snapshots are not certified results; callback errors propagate.
        """
        if not isinstance(solve_response, bool):
            raise ValueError("explicit boolean response policy required")
        if iteration_callback is not None and not callable(iteration_callback):
            raise ValueError("callable iteration callback required")
        if (
            isinstance(residual_tolerance_Ha, bool)
            or not np.isfinite(residual_tolerance_Ha)
            or residual_tolerance_Ha <= 0
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 1
                for x in (restart, maxiter)
            )
        ):
            raise ValueError(
                "positive finite residual and integer Krylov controls required"
            )
        key = tuple(key)
        if key not in self.plan["valence_keys"]:
            raise ValueError("declared column required")
        count = self.plan["raw_amplitudes"][key]
        if initial_numerator_derivative is not None and not solve_response:
            raise ValueError("response initial guess requires requested response")
        seeds = []
        for candidate in (initial_numerators, initial_numerator_derivative):
            if candidate is None:
                seeds.append(None)
                continue
            array = np.asarray(candidate)
            if (
                array.shape != (count,)
                or np.iscomplexobj(array)
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    "finite complete real numerator initial guess required"
                )
            seeds.append(array)
        extra = (
            8
            * count
            * (
                min(restart, count)
                + 18
                + sum(seed is not None for seed in seeds)
                + (2 if iteration_callback is not None else 0)
            )
        )
        if self.plan["complete_payload_bound_bytes"] + extra > self.core.memory_bytes:
            raise MemoryError(
                "complete core/valence and Krylov response exceed memory cap"
            )
        seeds = [None if seed is None else _owned(seed) for seed in seeds]
        started, before = time.perf_counter(), self._matvecs
        with self.core._lock:
            context = self._initialize(key, energy_Ha)
            snapshots = 0
            checkpoint_identity = None
            if iteration_callback is not None:
                checkpoint_identity = dict(
                    key=key,
                    energy_Ha=float(energy_Ha),
                    core_identity_sha256=self._core_identity,
                    core_amplitudes_sha256=self._core_amplitudes_sha256,
                    valence_source_sha256=tuple(_source_identity.items()),
                    packed_coordinates_sha256=tuple(
                        hashlib.sha256(a.tobytes()).hexdigest() for a in context[3]
                    ),
                    denominators_sha256=hashlib.sha256(
                        context[4].tobytes()
                    ).hexdigest(),
                    certified=False,
                    role="complete numerical initial guess; current full gates still required",
                )
            zero = np.zeros(count)
            constant = self._evaluate(energy_Ha, zero, None, context)[1]

            def action(vector):
                self._matvecs += 1
                return vector - self._evaluate_value(energy_Ha, vector, context)[1]

            operator = LinearOperator((count, count), matvec=action, dtype=float)

            def inverse(rhs, initial=None, *, stage):
                # Inner accuracy is stricter than the requested raw physical
                # gate; it is not a claimed condition-independent error bound.
                inner = min(residual_tolerance_Ha * 0.01, 1e-14)

                def callback(base=None, refinement=0):
                    if iteration_callback is None:
                        return None

                    def emit(vector):
                        nonlocal snapshots
                        complete = _owned(vector if base is None else base + vector)
                        if complete.shape != (count,) or not np.all(
                            np.isfinite(complete)
                        ):
                            raise RuntimeError(
                                "nonfinite/incomplete numerical iteration snapshot"
                            )
                        snapshots += 1
                        iteration_callback(
                            MappingProxyType(
                                dict(
                                    checkpoint_identity,
                                    stage=stage,
                                    refinement=refinement,
                                    ordinal=snapshots,
                                    numerators=complete,
                                )
                            )
                        )

                    return emit

                root, info = gmres(
                    operator,
                    rhs,
                    x0=initial,
                    atol=inner,
                    rtol=0,
                    restart=restart,
                    maxiter=maxiter,
                    callback=callback(),
                    callback_type="x",
                )
                for refinement in range(3):
                    defect = rhs - action(root)
                    if np.max(abs(defect)) <= inner:
                        break
                    delta, step_info = gmres(
                        operator,
                        defect,
                        atol=inner * 0.1,
                        rtol=0,
                        restart=restart,
                        maxiter=maxiter,
                        callback=callback(root, refinement + 1),
                        callback_type="x",
                    )
                    root += delta
                    info = step_info
                defect = action(root) - rhs
                if (
                    not np.all(np.isfinite(root))
                    or np.max(abs(defect)) > residual_tolerance_Ha
                ):
                    raise RuntimeError(
                        f"full valence raw residual gate failed: info={info}, max={np.max(abs(defect))} Ha"
                    )
                return root

            root = inverse(constant, seeds[0], stage="root")
            mapped = self._evaluate(energy_Ha, root, None, context)
            residual = root - mapped[1]
            maximum = float(np.max(abs(residual)))
            if not np.all(np.isfinite(residual)) or maximum > residual_tolerance_Ha:
                raise RuntimeError("complete affine valence residual gate failed")
            prime, sigma_prime, derivative_residual = None, None, None
            if solve_response:
                rhs = mapped[3]
                prime = inverse(rhs, seeds[1], stage="response")
                derivative_residual = float(np.max(abs(action(prime) - rhs)))
                # The full root map already certifies b'(E)+L'(E) root.
                # Add the exact homogeneous singles action on root'; do not
                # recompute all fixed-core source diagrams for this response.
                sigma_prime = (
                    mapped[2] + self._evaluate_value(energy_Ha, prime, context)[0]
                )
            partial, rows, _, _, denominator, orbital_energy, _ = context
            group = self.core.plan["groups"][partial]
            row_indices = tuple(
                i for i, k in enumerate(group) if k in self.core.plan["virtual_keys"]
            )
            provenance = dict(
                self.core.provenance,
                valence_source_sha256=dict(_source_identity),
                core_identity_sha256=self._core_identity,
                core_amplitudes_sha256=self._core_amplitudes_sha256,
                method="fixed-energy CI+all-order monovalent linearized SD, full scalar Fock",
                equation_coverage=self.core._manifest["valence_sd"],
                solver="matrix-free GMRES and actual homogeneous-operator refinement",
                homogeneous_action="complete values only; full root/implicit-rhs response retained separately",
                energy_reference="same native core Fock orbital zero; explicit E-column parameter",
                response="core numerators fixed, column energy varied, implicit numerator response"
                if solve_response
                else "not requested",
                estimated_krylov_bytes=extra,
                initial_guess_sha256=tuple(
                    None if seed is None else hashlib.sha256(seed.tobytes()).hexdigest()
                    for seed in seeds
                ),
                iteration_snapshots=snapshots,
                requested_residual_tolerance_Ha=residual_tolerance_Ha,
                space_converged=False,
                forward_accuracy_certified=False,
                experiment_validated=False,
            )
            return ValenceSDResult(
                key,
                float(energy_Ha),
                orbital_energy,
                tuple(group[i] for i in row_indices),
                rows,
                root,
                denominator,
                mapped[0][list(row_indices)],
                maximum,
                prime,
                None if sigma_prime is None else sigma_prime[list(row_indices)],
                derivative_residual,
                self._matvecs - before,
                time.perf_counter() - started,
                provenance,
            )
