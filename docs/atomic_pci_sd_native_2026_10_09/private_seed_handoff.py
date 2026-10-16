"""Private reorthogonal numerical guesses followed by actual public certification.

Two explicit CUDA callback owners share one locked physical CoreSD. A private
result is never relabeled as a public result or passed to Sigma2. Consumers must
call the public solver again and retain its original gates and provenance.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from cuda_valence_bridge import BoundCUDAValenceControl
from private_recycling_control import RecycledValenceSD


def activate(control):
    """Select this exact callback owner before its owned COMMON initializer."""
    control.lib.atomx_cuda_term2_bind(control.callback)
    control.select("cuda")


def device_payload_bound(feature, dimensions, nvh):
    """Two complete field/density contexts, matching the actual CUDA helpers."""
    d = dimensions
    nodes, nv, ng = len(feature.radii), *feature.outer_indices.shape[1:]
    geometry = 8 * (
        8 * nodes
        + 3 * feature.outer_indices.size
        + 12 * len(feature.edge_left)
        + 6 * nv * ng
    )
    density = geometry + 8 * d["NXX"] ** 2 * (2 * nodes + 4 * nv)
    return int(
        2
        * (
            density
            + 8
            * (
                d["NHF"] * d["NX"] * d["NK"]
                + 6 * d["NXX"] ** 2 * nvh
                + 8 * d["NHF"] * d["NXX"]
                + 3 * d["NXX"] ** 2
            )
        )
    )


class PrivateSeedHandoff:
    def __init__(self, public, core_result, public_cuda, out, *, cuda_krylov=False):
        self.public, self.public_cuda, self.out = public, public_cuda, Path(out)
        self.out.mkdir(exist_ok=False)
        count = max(public.plan["raw_amplitudes"].values())
        # Reserve complete private V/Z/CU/QR work plus both phase results,
        # coordinates and copied core arrays; no scientific space truncation.
        self.host_bound = (
            public.plan["complete_payload_bound_bytes"]
            + 8 * count * (180 + 16)
            + 3 * public.core.source.nbytes
            + public.core._manifest["private_cuda_term2"][
                "host_additional_payload_bytes"
            ]
        )
        d = public.core.plan["dimensions"]
        shape = (d["NXX"], d["NXX"], public.plan["nvh"])
        one_device = public_cuda.density.preflight(d["NXX"] ** 2) + 8 * (
            d["NHF"] * d["NX"] * d["NK"]
            + 6 * np.prod(shape)
            + 8 * d["NHF"] * d["NXX"]
            + 3 * d["NXX"] ** 2
        )
        self.device_bound = int(2 * one_device)
        if self.device_bound != device_payload_bound(
            public.core.features, d, public.plan["nvh"]
        ):
            raise ValueError("complete planning/runtime device payload differs")
        if not isinstance(cuda_krylov, bool):
            raise ValueError("explicit private CUDA Krylov policy required")
        if self.host_bound > public.core.memory_bytes:
            raise MemoryError("complete private/public seed handoff host cap exceeded")
        if self.device_bound > public_cuda.cap:
            raise MemoryError(
                "both complete owned CUDA field contexts exceed device cap"
            )
        self.candidate = RecycledValenceSD(
            public.core, core_result, valence_keys=public.plan["valence_keys"]
        )
        self.candidate_cuda = BoundCUDAValenceControl(
            self.candidate, self.out / "cuda", memory_bytes=public_cuda.cap
        )
        self.numerical_engine = None
        if cuda_krylov:
            from cuda_krylov_control import CUDAKrylovControl

            self.numerical_engine = CUDAKrylovControl(
                self.candidate_cuda.cp,
                self.out / "krylov",
                operator_device_bound_bytes=self.device_bound,
            )
        activate(public_cuda)
        (self.out / "contract.json").write_text(
            json.dumps(
                dict(
                    role="complete private numerical guesses only; public GMRES must freshly certify source/root/response before screening",
                    host_bound_bytes=int(self.host_bound),
                    host_cap_bytes=public.core.memory_bytes,
                    combined_device_bound_bytes=self.device_bound,
                    device_cap_bytes=public_cuda.cap,
                    resident_krylov=cuda_krylov,
                    resident_krylov_pool_cap_bytes=0
                    if self.numerical_engine is None
                    else self.numerical_engine.cap,
                    total_declared_device_pool_cap_bytes=public_cuda.cap
                    + (
                        0
                        if self.numerical_engine is None
                        else self.numerical_engine.cap
                    ),
                    source_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    budget="original m30/k10/maxiter100/three refinements, two full MGS passes, complete actual C refresh; no fallback or relaxed gate",
                    callback_policy="explicit owner activation before each full phase initialization; final public callback restored even on propagated failure",
                    public_cuda_adopted=False,
                ),
                indent=2,
            )
            + "\n"
        )

    def guess(self, key, energy_Ha, *, iteration_callback=None):
        activate(self.candidate_cuda)
        try:
            result = self.candidate.solve(
                key,
                energy_Ha,
                inverse_method="gcrotmk",
                residual_tolerance_Ha=1e-11,
                iteration_callback=iteration_callback,
                numerical_engine=self.numerical_engine,
            )
        finally:
            # Restore callback identity; an exception still rejects the run.
            activate(self.public_cuda)
        label = f"{key[0]}_{key[1]}"
        np.savez_compressed(
            self.out / f"guess_{label}.npz",
            numerators=result.numerators,
            prime=result.numerator_derivative,
            denominators=result.denominators_Ha,
            sigma1=result.sigma1_Ha,
            sigma1_prime=result.sigma1_derivative,
        )
        (self.out / f"guess_{label}.json").write_text(
            json.dumps(
                dict(
                    key=key,
                    energy_Ha=energy_Ha,
                    raw_root_Ha=result.residual_max_Ha,
                    raw_prime_Ha=result.derivative_residual_max_Ha,
                    matrix_vector_products=result.matrix_vector_products,
                    seconds=result.wall_seconds,
                    provenance=dict(result.provenance),
                    public_certified=False,
                    role="full native private solve; complete arrays are numerical guesses for a fresh public solve only",
                ),
                indent=2,
            )
            + "\n"
        )
        return result
