"""Properties from an explicitly supplied, complete finite correlated wave.

This consumer shares ``FiniteCITransitionOperator`` parent/ordering identity.
It requires the actual ket wave e^T (1+S) P and total model Hamiltonian;
neither a metric nor correlation amplitudes are inferred from energies. The
complete full-H residual is an admission gate. Truncated large-space CC
waves that fail it need a separately declared approximation contract.
"""

from dataclasses import dataclass

import numpy as np

from atomx.methods.ci_effective import FiniteCIModel, FiniteCITransitionOperator


def _owned(array):
    value = np.asarray(array, dtype=np.complex128)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _positive(value, name):
    if (
        isinstance(value, bool)
        or np.ndim(value) != 0
        or not np.isrealobj(value)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"positive finite {name} required")


@dataclass(frozen=True, eq=False, init=False)
class FiniteCorrelatedWave:
    """Owned finite wave, physical metric and isometry in a bound full basis.

    Model coordinates are fixed by P Psi = I. Complex phases and directed
    model matrices are retained. This class proves its finite equations,
    not omitted orbital/excitation convergence or experimental accuracy.
    """

    def __init__(
        self,
        parent,
        wave_operator,
        model_hamiltonian_Ha,
        *,
        source_identity,
        metric=None,
        dimension_cap=1000,
        memory_bytes=64 * 1024**2,
        residual_tolerance_Ha=1e-11,
        normalization_tolerance=1e-12,
        metric_condition_max=1e8,
    ):
        if not isinstance(parent, FiniteCIModel):
            raise TypeError("exact finite CI parent snapshot required")
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise ValueError("actual wave source identity required")
        n, p = len(parent.hamiltonian_Ha), len(parent.p_indices)
        if (
            isinstance(dimension_cap, bool)
            or not isinstance(dimension_cap, (int, np.integer))
            or dimension_cap < n
        ):
            raise ValueError("complete finite dimension exceeds cap")
        if (
            isinstance(memory_bytes, bool)
            or not isinstance(memory_bytes, (int, np.integer))
            or memory_bytes <= 0
        ):
            raise ValueError("positive integer memory cap required")
        if 16 * (3 * n * n + 8 * n * p + 15 * p * p) > memory_bytes:
            raise MemoryError("complete wave and metric work exceeds cap")
        for value, name in (
            (residual_tolerance_Ha, "residual tolerance"),
            (normalization_tolerance, "normalization tolerance"),
            (metric_condition_max, "metric condition"),
        ):
            _positive(value, name)
        if metric_condition_max < 1:
            raise ValueError("metric condition bound must be at least one")
        wave, model = np.asarray(wave_operator), np.asarray(model_hamiltonian_Ha)
        if (
            wave.shape != (n, p)
            or model.shape != (p, p)
            or wave.dtype.kind not in "iufc"
            or model.dtype.kind not in "iufc"
            or not np.all(np.isfinite(wave))
            or not np.all(np.isfinite(model))
        ):
            raise ValueError(
                "complete finite signed wave and total model matrix required"
            )
        wave, model = _owned(wave), _owned(model)
        g = wave.conj().T @ wave
        if metric is not None:
            supplied = np.asarray(metric)
            if (
                supplied.shape != (p, p)
                or not np.all(np.isfinite(supplied))
                or np.max(abs(supplied - g)) > normalization_tolerance
            ):
                raise ValueError(
                    "supplied physical metric does not equal full wave adjoint product"
                )
        vals, rotation = np.linalg.eigh(g)
        condition = float(vals[-1] / vals[0]) if vals[0] > 0 else float("inf")
        if not np.isfinite(condition) or condition > metric_condition_max:
            raise ValueError("physical wave metric is singular or ill-conditioned")
        inverse_sqrt = (rotation / np.sqrt(vals)) @ rotation.conj().T
        isometry = wave @ inverse_sqrt
        residual = parent.hamiltonian_Ha @ wave - wave @ model
        residual_max = float(np.max(abs(residual)))
        adjoint_max = float(np.max(abs(model.conj().T @ g - g @ model)))
        normalization = max(
            float(np.max(abs(wave[list(parent.p_indices)] - np.eye(p)))),
            float(np.max(abs(isometry.conj().T @ isometry - np.eye(p)))),
        )
        if (
            max(residual_max, adjoint_max) > residual_tolerance_Ha
            or normalization > normalization_tolerance
        ):
            raise ValueError(
                "full correlated wave equation or normalization gate failed"
            )
        for key, value in dict(
            parent=parent,
            wave_operator=wave,
            model_hamiltonian_Ha=model,
            source_identity=source_identity,
            metric=_owned(g),
            orthonormal_wave_operator=_owned(isometry),
            intertwining_residual_Ha=residual_max,
            metric_adjoint_residual_Ha=adjoint_max,
            normalization_error=normalization,
            metric_condition=condition,
        ).items():
            object.__setattr__(self, key, value)

    def effective_transition(
        self, operator, ket, *, representation, memory_bytes=64 * 1024**2
    ):
        """Complete signed property using both independently supplied metrics.

        Bloch coordinates use G_b^-1 Psi_b^dagger O Psi_k; orthonormal
        coordinates use W_b^dagger O W_k. The latter incorporates full
        physical state normalization and preserves weak/interfering entries.
        """
        if not isinstance(operator, FiniteCITransitionOperator) or not isinstance(
            ket, FiniteCorrelatedWave
        ):
            raise TypeError("bound finite transition and correlated ket wave required")
        if operator.bra is not self.parent or operator.ket is not ket.parent:
            raise ValueError("transition belongs to different parent snapshots")
        if (
            isinstance(memory_bytes, bool)
            or not isinstance(memory_bytes, (int, np.integer))
            or memory_bytes <= 0
        ):
            raise ValueError("positive integer memory cap required")
        nb, pb = self.wave_operator.shape
        nk, pk = ket.wave_operator.shape
        if (
            16 * (nb * nk + 3 * pb * nk + 12 * pb * pk + 6 * (pb * pb + pk * pk))
            > memory_bytes
        ):
            raise MemoryError("complete transition work exceeds cap")
        if representation == "bloch":
            return np.linalg.solve(
                self.metric,
                self.wave_operator.conj().T @ operator.matrix @ ket.wave_operator,
            )
        if representation == "orthonormal":
            return (
                self.orthonormal_wave_operator.conj().T
                @ operator.matrix
                @ ket.orthonormal_wave_operator
            )
        raise ValueError("explicit bloch or orthonormal representation required")
