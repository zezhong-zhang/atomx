"""Private full-coordinate directed Davidson control for standalone SD.

At each coupling the exact augmented pencil acts on (1, rho). Cached complete
native A*rho images are independent of coupling and deltaEv because diagrams
are evaluated at bare ev. The small projected pencil is only a numerical
subspace: actual full public-map raw/energy gates remain authoritative.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import get_blas_funcs

from atomx.methods.reduced_monovalent_sd import ReducedMonovalentSDResult


def solve(problem, key, *, residual_tolerance_Ha=1e-10, subspace=16, max_actions=160):
    tolerance = float(residual_tolerance_Ha)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-10:
        raise ValueError("positive private raw tolerance <=1e-10 Ha required")
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v < 2
        for v in (subspace, max_actions)
    ):
        raise ValueError("explicit positive private subspace/action budget required")
    key = tuple(key)
    count = problem.plan["amplitudes"][key]
    maximum = min(count, subspace)
    # Two complete basis/image families; sixteen work vectors are already in
    # the public plan. No magnetic/determinant tensor or physical cutoff.
    extra = 8 * (2 * maximum * count + 8 * (maximum + 1) ** 2)
    with problem.core._lock:
        problem._check(extra)
        prepared = problem._prepare(key)
        diagonal = prepared[2]
        start = problem._matvecs

        def action(vector, *, linear=True):
            if problem._matvecs - start >= max_actions:
                raise RuntimeError("private complete native action budget exhausted")
            return problem._evaluate(vector, prepared, linear=linear)

        constant, source = action(np.zeros(count), linear=False)
        axpy, dot, norm, scale = get_blas_funcs(
            ["axpy", "dot", "nrm2", "scal"], (source,)
        )
        vectors, images, energies = [], [], []
        projected_a = np.zeros((maximum, maximum))
        projected_d = np.zeros_like(projected_a)
        projected_b = np.zeros(maximum)
        anchor = np.zeros(count)
        path, diagnostics = [], []
        restarts = 0

        def append(candidate):
            nonlocal vectors, images, energies
            candidate = np.array(candidate, dtype=float, copy=True)
            initial_norm = norm(candidate)
            for _ in range(2):
                for previous in vectors:
                    candidate = axpy(previous, candidate, a=-dot(previous, candidate))
            length = norm(candidate)
            if not np.isfinite(length) or length <= np.finfo(float).eps * initial_norm:
                raise RuntimeError(
                    "private full-vector MGS2 numerical breakdown before raw gate"
                )
            candidate = scale(1 / length, candidate)
            if max((abs(dot(v, candidate)) for v in vectors), default=0) > 5e-12:
                raise RuntimeError("private full-vector orthogonality gate failed")
            energy, image = action(candidate)
            index = len(vectors)
            dv = diagonal * candidate
            for i, (v, av) in enumerate(zip(vectors, images, strict=True)):
                projected_a[i, index] = dot(v, image)
                projected_a[index, i] = dot(candidate, av)
                projected_d[i, index] = dot(v, dv)
                projected_d[index, i] = dot(candidate, diagonal * v)
            projected_a[index, index] = dot(candidate, image)
            projected_d[index, index] = dot(candidate, dv)
            projected_b[index] = dot(candidate, source)
            vectors.append(candidate)
            images.append(image)
            energies.append(energy)

        for coupling in np.linspace(0.125, 1, 8):
            iteration = 0
            while True:
                iteration += 1
                k = len(vectors)
                pencil = np.empty((k + 1, k + 1))
                pencil[0, 0] = coupling * constant
                pencil[0, 1:] = coupling * np.asarray(energies)
                pencil[1:, 0] = coupling * projected_b[:k]
                pencil[1:, 1:] = coupling * projected_a[:k, :k] - projected_d[:k, :k]
                values, roots = np.linalg.eig(pencil)
                previous = np.r_[1.0, [dot(v, anchor) for v in vectors]]
                overlaps = abs(previous @ roots) / (
                    np.linalg.norm(previous) * np.linalg.norm(roots, axis=0)
                )
                selected = int(np.argmax(overlaps))
                value, root = values[selected], roots[:, selected]
                if (
                    overlaps[selected] < 0.5
                    or abs(value.imag) > 1e-12
                    or np.max(abs(root.imag)) > 1e-12
                ):
                    raise RuntimeError(
                        "private real reference-connected numerical branch gate failed"
                    )
                if abs(root[0]) < 1e-8:
                    raise RuntimeError(
                        "private intermediate-normalization branch is singular"
                    )
                delta = float(value.real)
                coefficients = (root[1:] / root[0]).real
                rho, rhs = np.zeros(count), source.copy()
                for coefficient, v, av in zip(
                    coefficients, vectors, images, strict=True
                ):
                    rho = axpy(v, rho, a=coefficient)
                    rhs = axpy(av, rhs, a=coefficient)
                energy = constant + float(np.dot(energies, coefficients))
                shifted = diagonal + delta
                if np.any(abs(shifted) < 1e-8):
                    raise ValueError("small private feedback-shifted denominator")
                raw = coupling * rhs - shifted * rho
                if (
                    max(float(np.max(abs(raw))), abs(coupling * energy - delta))
                    <= tolerance / 4
                ):
                    actual_e, actual_a = action(rho)
                    image_error = max(
                        abs(actual_e - (energy - constant)),
                        float(np.max(abs(actual_a - (rhs - source)))),
                    )
                    actual_raw = coupling * (source + actual_a) - shifted * rho
                    measured_raw = float(np.max(abs(actual_raw)))
                    measured_energy = abs(coupling * (constant + actual_e) - delta)
                    if (
                        image_error > 1e-11
                        or max(measured_raw, measured_energy) > tolerance
                    ):
                        raise RuntimeError(
                            "private cached-image/public complete raw-energy gate failed"
                        )
                    path.append(
                        (
                            float(coupling),
                            delta,
                            iteration,
                            measured_raw,
                            measured_energy,
                        )
                    )
                    diagnostics.append(
                        dict(
                            coupling=float(coupling),
                            dimension=k,
                            overlap=float(overlaps[selected]),
                            image_error_Ha=image_error,
                        )
                    )
                    anchor = rho.copy()
                    break
                if k == maximum:
                    # Numerical restart, not physical coordinate truncation.
                    vectors, images, energies = [], [], []
                    projected_a.fill(0)
                    projected_d.fill(0)
                    projected_b.fill(0)
                    restarts += 1
                    if norm(rho) != 0:
                        append(rho)
                append(raw / shifted)
        reference, rhs = action(rho, linear=False)
        final_raw = float(np.max(abs(rhs - (diagonal + delta) * rho)))
        final_energy = abs(reference - delta)
        if max(final_raw, final_energy) > tolerance:
            raise RuntimeError(
                "private final actual public affine raw-energy gate failed"
            )
        problem._check(extra)
        provenance = dict(
            problem.core.provenance,
            method="private directed Davidson numerical root; public complete standalone equations",
            solver_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            cached_image_policy="Full bare-ev native images, MGS2; actual public complete raw/energy recheck at every coupling and final affine recheck",
            numerical_branch_metric="Euclidean augmented numerical coordinate overlap only; not physical SD-state norm or resonance identity",
            resources=dict(
                complete_payload_bound_bytes=problem.plan[
                    "complete_payload_bound_bytes"
                ]
                + extra,
                subspace=maximum,
                vector_families=2,
                max_actions=max_actions,
            ),
            diagnostics=diagnostics,
            restarts=restarts,
            scope="private numerical equation result; no public solver adoption, normalized production state/property, independent full-Yb oracle, physical precision or pole certificate",
        )
        return ReducedMonovalentSDResult(
            key,
            prepared[-1],
            delta,
            prepared[-1] + delta,
            problem.core_result.total_energy_Ha + prepared[-1] + delta,
            problem.plan["single_keys"][key],
            problem.plan["channels"][key],
            rho,
            diagonal + delta,
            final_raw,
            final_energy,
            problem._matvecs - start,
            tuple(path),
            provenance,
        )
