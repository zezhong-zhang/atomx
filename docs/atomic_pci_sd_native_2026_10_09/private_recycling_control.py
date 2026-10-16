"""Private same-column GCROT control with actual C=A*U refresh; not public SD.

Derived from the retained022 numerical candidate. Every physical operation is
inherited from current certified ValenceSD. Full native gates remain required;
this control cannot grant a public Sigma2/CI or production capability.
"""

import hashlib
import time
from pathlib import Path
from types import MappingProxyType

import numpy as np
from reorthogonal_gcrot import gcrotmk
from scipy.sparse.linalg import LinearOperator, gmres

from atomx.methods.core_sd import _owned
from atomx.methods.valence_sd import ValenceSD, ValenceSDResult
from atomx.methods.valence_sd import _source_identity as public_source

_source_identity = MappingProxyType(
    dict(
        public_source,
        private_recycling_control=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        private_reorthogonal_gcrot=hashlib.sha256(
            Path(__file__).with_name("reorthogonal_gcrot.py").read_bytes()
        ).hexdigest(),
    )
)


class RecycledValenceSD(ValenceSD):
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
        inverse_method="gmres",
        recycle_vectors=None,
        numerical_engine=None,
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
        Optional GCROT(m,k) recycles only within this one initialized column,
        across root/refinement/response right-hand sides of the same operator.
        Its outer-cycle budget is maxiter; restart supplies m. No recycled
        subspace persists across columns, energies, COMMON resets or solves.
        """
        if not isinstance(solve_response, bool):
            raise ValueError("explicit boolean response policy required")
        if iteration_callback is not None and not callable(iteration_callback):
            raise ValueError("callable iteration callback required")
        if inverse_method not in ("gmres", "gcrotmk"):
            raise ValueError("explicit gmres/gcrotmk inverse method required")
        if inverse_method == "gmres" and recycle_vectors is not None:
            raise ValueError("recycle vectors require gcrotmk")
        if numerical_engine is not None and (
            inverse_method != "gcrotmk" or not callable(numerical_engine.solve)
        ):
            raise ValueError("explicit private resident GCROT engine required")
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
        capacity = 0
        if inverse_method == "gcrotmk":
            capacity = (
                min(10, restart, count) if recycle_vectors is None else recycle_vectors
            )
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or not 1 <= capacity <= min(restart, count)
            ):
                raise ValueError(
                    "positive bounded integer gcrotmk recycle vectors required"
                )
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
        vectors = min(restart, count) + 18
        if inverse_method == "gcrotmk":
            # Initial inner FGMRES may use m+k vectors. Reserve both V/Z,
            # CU, QR/reorthogonalization scratch and full root/response work.
            vectors = 2 * (min(restart + capacity, count) + 1) + 6 * (capacity + 1) + 24
        extra = (
            8
            * count
            * (
                vectors
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
                    inverse_method=inverse_method,
                    iteration_boundary="GMRES restart"
                    if inverse_method == "gmres"
                    else "GCROT outer cycle",
                )
            zero = np.zeros(count)
            constant = self._evaluate(energy_Ha, zero, None, context)[1]

            def action(vector):
                self._matvecs += 1
                return vector - self._evaluate_value(energy_Ha, vector, context)[1]

            operator = LinearOperator((count, count), matvec=action, dtype=float)
            recycled = []

            def linear_solve(rhs, initial, inner, notify):
                if inverse_method == "gmres":
                    return gmres(
                        operator,
                        rhs,
                        x0=initial,
                        atol=inner,
                        rtol=0,
                        restart=restart,
                        maxiter=maxiter,
                        callback=notify,
                        callback_type="x",
                    )
                inverse = (
                    gcrotmk if numerical_engine is None else numerical_engine.solve
                )
                return inverse(
                    operator,
                    rhs,
                    x0=initial,
                    atol=inner,
                    rtol=0,
                    m=min(restart, count),
                    k=capacity,
                    maxiter=maxiter,
                    CU=recycled,
                    callback=notify,
                    discard_C=True,
                    truncate="oldest",
                )

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

                root, info = linear_solve(rhs, initial, inner, callback())
                for refinement in range(3):
                    defect = rhs - action(root)
                    if np.max(abs(defect)) <= inner:
                        break
                    delta, step_info = linear_solve(
                        defect, None, inner * 0.1, callback(root, refinement + 1)
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
                solver=f"matrix-free {inverse_method} and actual homogeneous-operator refinement",
                inverse_method=inverse_method,
                recycle_vectors_capacity=capacity,
                recycling_scope="one initialized column, same root/refinement/response operator; no cross-solve subspace",
                recycled_image_policy="discard_C=True: recompute actual C=A U before each inverse; retained012 negative case not erased",
                arnoldi_policy="two complete MGS passes against all C and V; complete coefficients retained, original budgets and native gates",
                private_candidate=True,
                numerical_engine=None
                if numerical_engine is None
                else dict(numerical_engine.identity),
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
