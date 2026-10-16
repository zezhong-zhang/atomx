"""Native J-reduced standalone monovalent SD with fixed-core feedback.

All direct rho channels and radial virtual coordinates are retained. The
original diagrams consume physical amplitudes at bare ev; deltaEv enters
only the coupled valence equation. No determinant Hamiltonian, dense SD
matrix or magnetic amplitude tensor is built. This real DC interface does
not yet supply a production normalized SD state or property consumer.
"""

import ctypes
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from atomx.methods.core_sd import CoreSD, CoreSDResult, _owned
from atomx.methods.valence_sd import ValenceSD, plan_valence_sd

_template = Path(__file__).with_name("_pci_monovalent.f90")
_template_sha = hashlib.sha256(_template.read_bytes()).hexdigest()
_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _integer(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"integer {name} >= {minimum} required")
    return int(value)


def _real(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or np.ndim(value) != 0
        or not np.isrealobj(value)
        or not np.isfinite(value)
    ):
        raise ValueError(f"finite real {name} required")
    return float(value)


def plan_reduced_monovalent_sd(core_plan, valence_keys, **valence_options):
    """Complete reduced payload before any native initialization or action.

    Counts include independent singles excluding reference, every original
    redundant double, native static scratch, packed coordinates and scalar
    solve scratch. Arnoldi vectors and their small Hessenberg matrix are
    counted separately by solve(). Existing core/cache/runtime bounds retain
    their documented scope; no virtual or rho channel is dropped.
    """
    base = plan_valence_sd(core_plan, valence_keys, **valence_options)
    virtual, groups = core_plan["virtual_keys"], core_plan["groups"]
    singles = {
        key: tuple(
            row
            for group in groups
            for row in group
            if row in virtual and row[1] == key[1] and row != key
        )
        for key in base["valence_keys"]
    }
    sizes = {
        key: len(singles[key]) + base["raw_amplitudes"][key]
        for key in base["valence_keys"]
    }
    d = core_plan["dimensions"]
    static = 8 * (2 * d["NXX"] ** 2 * base["nvh"] + 4 * d["NX"])
    packed = 12 * max(base["raw_amplitudes"].values())
    scratch = 8 * 16 * max(sizes.values())
    return MappingProxyType(
        dict(
            base,
            single_keys=MappingProxyType(singles),
            amplitudes=MappingProxyType(sizes),
            standalone_static_bytes=static,
            packed_coordinate_bytes=packed,
            scalar_solve_scratch_bytes=scratch,
            complete_payload_bound_bytes=base["complete_payload_bound_bytes"]
            + static
            + packed
            + scratch,
            scope="complete native core/valence, standalone static arrays, packed coordinates and 16 scalar vectors; add Arnoldi/Hessenberg scratch; excludes interpreter/JIT/runtime RSS",
        )
    )


@dataclass(frozen=True)
class ReducedMonovalentSDResult:
    column_key: tuple
    orbital_energy_Ha: float
    correlation_energy_Ha: float
    attachment_energy_Ha: float
    total_energy_Ha: float
    single_keys: tuple
    channels: tuple
    amplitudes: np.ndarray
    denominators_Ha: np.ndarray
    residual_max_Ha: float
    energy_residual_Ha: float
    matrix_vector_products: int
    continuation_path: tuple
    provenance: object

    def __post_init__(self):
        for name in ("amplitudes", "denominators_Ha"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class ReducedMonovalentSD:
    """Separate opt-in standalone SD on a source-bound native CoreSD owner.

    COMMON restoration and all solves run under that owner's lock. Scalar
    feedback uses matrix-free GMRES inverses and an analytic energy derivative,
    with explicit coupling continuation. This is a numerical branch tracker,
    not an intruder/continuum-pole or forward-error certificate.
    """

    def __init__(self, core_problem, core_result, *, valence_keys):
        if not isinstance(core_problem, CoreSD) or not isinstance(
            core_result, CoreSDResult
        ):
            raise TypeError("actual native CoreSD problem and result required")
        native = core_problem._manifest.get("monovalent_sd")
        if (
            not isinstance(native, dict)
            or native.get("template_sha256") != _template_sha
            or native.get("core_denominators")
            != "unshifted closed-core; all diagrams at bare ev"
        ):
            raise ValueError("source-bound standalone native SD extension required")
        self.core, self.core_result = core_problem, core_result
        # Budget the extension before binding ValenceSD can re-certify an
        # uncached parent through a native action or allocate owned copies.
        fixed = self.core._manifest.get("valence_sd", {})
        term2 = fixed.get("term2", {"mode": "original"})
        self.plan = plan_reduced_monovalent_sd(
            self.core.plan,
            valence_keys,
            valence_term2_blas=term2.get("mode") == "streamed_blas",
            valence_term2_threads=term2.get("threads", 1),
        )
        if native.get("extra_static_bytes") != self.plan["standalone_static_bytes"]:
            raise ValueError("standalone native static resource identity differs")
        self._current_binding = self.core._solution_identity(core_result)
        self._check()
        self.valence = ValenceSD(
            core_problem, core_result, valence_keys=self.plan["valence_keys"]
        )
        self._map = self.core._backend.atomx_monovalent_map
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        self._map.argtypes = [ctypes.c_int] + [pointer] * 4
        self._map.restype = None
        self._matvecs = 0
        self._check()
        if not self.core._has_current_solution_certificate(core_result, 1e-10):
            reference = self.core.integrals.reference_energy_Ha
            correlation = self.core.energy(core_result.amplitudes)
            if (
                max(
                    abs(reference - core_result.reference_energy_Ha),
                    abs(correlation - core_result.correlation_energy_Ha),
                    abs(reference + correlation - core_result.total_energy_Ha),
                )
                > 1e-10
            ):
                raise ValueError("actual standalone core energies differ")

    def _check(self, extra_bytes=0):
        if self.core._solution_identity(self.core_result) != self._current_binding:
            raise ValueError("actual standalone core/source/result identity changed")
        if (
            self.plan["complete_payload_bound_bytes"] + extra_bytes
            > self.core.memory_bytes
        ):
            raise MemoryError("complete reduced standalone SD payload exceeds cap")

    def _prepare(self, key):
        key = tuple(key)
        if key not in self.plan["valence_keys"]:
            raise ValueError("declared standalone valence key required")
        labels = self.core.integrals.labels
        eps = self.core.integrals.fock_Ha.diagonal().real
        by_key = {row[:2]: float(eps[i]) for i, row in enumerate(labels)}
        context = self.valence._initialize(key, by_key[key])
        partial, rows, _, _, doubles, orbital, _ = context
        group = self.core.plan["groups"][partial]
        keys = self.plan["single_keys"][key]
        slots = np.asarray([group.index(row) for row in keys], dtype=np.int32)
        diagonal = np.empty(len(slots) + len(doubles))
        diagonal[: len(slots)] = [orbital - by_key[row] for row in keys]
        diagonal[len(slots) :] = doubles
        diagonal = _owned(diagonal)
        if np.any(abs(diagonal) < 1e-8):
            raise ValueError("small standalone unshifted denominator")
        return context, slots, diagonal, group.index(key), orbital

    def _evaluate(self, vector, prepared, *, linear=False):
        context, slots, diagonal, reference, _ = prepared
        values = np.asarray(vector)
        ns, count = len(slots), len(diagonal)
        if (
            values.shape != (count,)
            or not np.isrealobj(values)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("finite complete physical standalone amplitudes required")
        _, _, _, index, _, _, (work, singles) = context
        d, _, rd, _ = work
        s, rs, _ = singles
        s.fill(0)
        s[slots] = values[:ns]
        # Division inside the original diagrams then recovers physical rho.
        d[index] = diagonal[ns:] * values[ns:]
        self._map(int(linear), s, d, rs, rd)
        self._matvecs += 1
        if not np.all(np.isfinite(rs)) or not np.all(np.isfinite(rd)):
            raise RuntimeError("nonfinite complete standalone native SD action")
        return float(rs[reference]), np.concatenate((rs[slots], rd[index]))

    def evaluate(self, key, amplitudes, *, linear=False):
        """Return delta-energy map and every physical rho RHS, in Ha.

        Energy feedback is not applied here. linear=True is source-separated,
        with the same complete diagrams and no affine subtraction.
        """
        if not isinstance(linear, bool):
            raise ValueError("explicit boolean standalone linear policy required")
        with self.core._lock:
            self._check()
            prepared = self._prepare(key)
            reference, rhs = self._evaluate(amplitudes, prepared, linear=linear)
            self._check()
            return reference, _owned(rhs)

    def solve(
        self,
        key,
        *,
        residual_tolerance_Ha=1e-10,
        continuation=None,
        max_outer_iterations=16,
        restart=20,
        maxiter=200,
        krylov_vectors=40,
    ):
        """Solve (D0+deltaEv) rho=R(rho), deltaEv=R_reference(rho).

        No denominator shifts, Hermitian averaging, physical-coordinate
        truncation or automatic backend fallback. Retains full raw and energy
        gates; a failed inverse/Newton/denominator gate raises.
        """
        tolerance = _real(residual_tolerance_Ha, "raw tolerance")
        if not 0 < tolerance <= 1e-10:
            raise ValueError("positive standalone raw tolerance <=1e-10 Ha required")
        outer_limit = _integer(max_outer_iterations, "outer iterations")
        restart = _integer(restart, "restart")
        maxiter = _integer(maxiter, "inverse iterations")
        vectors = _integer(krylov_vectors, "Krylov vector budget")
        steps = np.asarray(
            np.linspace(0.125, 1, 8) if continuation is None else continuation
        )
        if (
            steps.ndim != 1
            or not 1 <= steps.size <= 64
            or not np.isrealobj(steps)
            or not np.all(np.isfinite(steps))
            or steps[0] <= 0
            or steps[-1] != 1
            or np.any(np.diff(steps) <= 0)
        ):
            raise ValueError("increasing standalone continuation ending at1 required")
        key = tuple(key)
        if key not in self.plan["valence_keys"]:
            raise ValueError("declared standalone valence key required")
        count = self.plan["amplitudes"][key]
        restart = min(restart, count)
        if restart + 2 > vectors:
            raise MemoryError("complete standalone Arnoldi vectors exceed budget")
        extra = 8 * (vectors * count + 4 * (restart + 1) ** 2)
        with self.core._lock:
            self._check(extra)
            prepared = self._prepare(key)
            _, _, diagonal, _, orbital = prepared
            start = self._matvecs
            constant, source = self._evaluate(np.zeros(count), prepared)
            delta, amplitudes, path = 0.0, np.zeros(count), []
            inverse_tolerance = min(tolerance / 20, 1e-12)
            for coupling in steps:
                for iteration in range(outer_limit):
                    shifted = diagonal + delta
                    if np.any(abs(shifted) < 1e-8):
                        raise ValueError("small feedback-shifted valence denominator")

                    def action(vector, shifted=shifted, coupling=coupling):
                        return (
                            shifted * vector
                            - coupling
                            * self._evaluate(vector, prepared, linear=True)[1]
                        )

                    operator = LinearOperator(
                        (count, count), matvec=action, dtype=float
                    )
                    preconditioner = LinearOperator(
                        (count, count),
                        matvec=lambda v, shifted=shifted: v / shifted,
                        dtype=float,
                    )

                    def inverse(
                        rhs,
                        initial=None,
                        operator=operator,
                        preconditioner=preconditioner,
                        action=action,
                    ):
                        value, info = gmres(
                            operator,
                            rhs,
                            x0=initial,
                            M=preconditioner,
                            atol=inverse_tolerance,
                            rtol=0,
                            restart=restart,
                            maxiter=maxiter,
                        )
                        error = action(value) - rhs
                        if (
                            info != 0
                            or not np.all(np.isfinite(error))
                            or np.max(abs(error)) > inverse_tolerance
                        ):
                            raise RuntimeError(
                                "complete standalone GMRES raw inverse gate failed"
                            )
                        return value

                    amplitudes = inverse(coupling * source, amplitudes)
                    energy_linear, rhs_linear = self._evaluate(
                        amplitudes, prepared, linear=True
                    )
                    energy_error = coupling * (constant + energy_linear) - delta
                    raw = coupling * (source + rhs_linear) - shifted * amplitudes
                    if max(abs(energy_error), np.max(abs(raw))) <= tolerance:
                        path.append(
                            (
                                float(coupling),
                                delta,
                                iteration + 1,
                                float(np.max(abs(raw))),
                                abs(energy_error),
                            )
                        )
                        break
                    derivative = inverse(-amplitudes)
                    slope = (
                        coupling * self._evaluate(derivative, prepared, linear=True)[0]
                        - 1
                    )
                    if not np.isfinite(slope) or abs(slope) < 1e-8:
                        raise RuntimeError(
                            "singular standalone analytic feedback derivative"
                        )
                    delta -= energy_error / slope
                    if not np.isfinite(delta):
                        raise RuntimeError("nonfinite standalone energy feedback")
                else:
                    raise RuntimeError("standalone energy feedback did not converge")
            reference, rhs = self._evaluate(amplitudes, prepared)
            raw = rhs - (diagonal + delta) * amplitudes
            maximum, energy_error = float(np.max(abs(raw))), abs(reference - delta)
            if max(maximum, energy_error) > tolerance:
                raise RuntimeError(
                    "actual affine standalone final raw/energy gate failed"
                )
            self._check(extra)
            provenance = dict(
                self.core.provenance,
                method="J-reduced standalone linearized monovalent SD, native DC",
                standalone_source_sha256=_source_sha,
                standalone_template_sha256=_template_sha,
                core_denominators="fixed closed-core",
                physical_coordinates="singles then all redundant direct rho_k core-valence doubles",
                core_amplitudes_sha256=hashlib.sha256(
                    self.core_result.amplitudes
                ).hexdigest(),
                resources=dict(
                    complete_payload_bound_bytes=self.plan[
                        "complete_payload_bound_bytes"
                    ]
                    + extra,
                    krylov_vectors=vectors,
                    restart=restart,
                ),
                scope="equation/root result; no normalized production state, property, physical precision or resonance certificate",
            )
            return ReducedMonovalentSDResult(
                key,
                orbital,
                delta,
                orbital + delta,
                self.core_result.total_energy_Ha + orbital + delta,
                self.plan["single_keys"][key],
                self.plan["channels"][key],
                amplitudes,
                diagonal + delta,
                maximum,
                energy_error,
                self._matvecs - start,
                tuple(path),
                provenance,
            )
