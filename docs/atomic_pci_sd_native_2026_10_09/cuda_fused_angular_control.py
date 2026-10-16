"""Private fused disjoint angular accumulation; complete original order kept.

One GPU launch per original core subgroup replaces per-row multiply/add.
Output-channel uniqueness is proved from the full native plan before launch;
subgroups retain their original order. No atomic sums or physical truncation.
"""

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from cuda_batched_projection_control import (
    BoundBatchedCUDAControl,
    NativeBatchedCUDAControl,
)

KERNEL = r"""
extern "C" __global__ void angular_add(
 const double* value,const double* prime,const int* rows,const double* coeff,
 double* output,double* derivative,int count,int cm,int xx,int sm,int width,int response) {
 long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(tid>=(long long)count*cm*xx) return;
 int n=tid%xx, m=(tid/xx)%cm, row=tid/(cm*xx);
 int index=rows[4*row], sn=rows[4*row+1], cn=rows[4*row+2], off=rows[4*row+3];
 if(n>=cn) return;
 long long input=(long long)m*width+off+n;
 long long target=((long long)index*xx+sm+m)*xx+sn+n;
 output[target]=output[target]+coeff[row]*value[input];
 if(response) derivative[target]=derivative[target]+coeff[row]*prime[input];
}
"""


class FusedProjectionMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.projection_policy != "reachable":
            raise ValueError("explicit exactly reachable fused projection required")
        self.angular_identity = None
        self.angular_kernel = self.cp.RawKernel(
            KERNEL, "angular_add", options=("--fmad=false",)
        )

    def prepare_projection(self):
        super().prepare_projection()
        identity = self.projection_identity
        if identity == self.angular_identity:
            return
        self.angular_rows = self.angular_coeff = None
        compact = []
        ranges = {}
        for triple, members in itertools.groupby(self.rows, lambda row: tuple(row[:3])):
            for core, subgroup in itertools.groupby(
                members, lambda row: tuple(row[3:7])
            ):
                rows = list(subgroup)
                if len({int(row[8]) for row in rows}) != len(rows):
                    raise ValueError(
                        "native angular subgroup writes overlap; disjoint fused kernel rejected"
                    )
                selection = tuple(dict.fromkeys(int(row[7]) - 1 for row in rows))
                _, offsets = self.projection_fields[selection]
                start = len(compact)
                for row in rows:
                    inn = int(row[7]) - 1
                    cm, cr, cs, cn = (
                        self.phi[int(row[i]) - 1].shape[1] for i in (0, 1, 4, 7)
                    )
                    if (
                        tuple(map(int, row[9:])) != (cm, cr, cs, cn)
                        or not 0 < int(row[8]) <= self.expected_shape[2]
                    ):
                        raise ValueError(
                            "complete fused row shape/output coordinate differs"
                        )
                    compact.append(
                        (int(row[8]) - 1, int(self.starts[inn]), cn, offsets[inn][0])
                    )
                ranges[(tuple(triple), tuple(core))] = (start, len(compact))
        if len(compact) != len(self.rows):
            raise ValueError("complete ordered angular plan not covered")
        baseline = json.loads((self.out / "batched_projection.json").read_text())[
            "device_bound_bytes"
        ]
        bound = baseline + 24 * len(compact)
        free, _ = self.cp.cuda.runtime.memGetInfo()
        if bound > self.cap or bound > free:
            raise MemoryError(
                "complete fused geometry plus field cap/free preflight failed"
            )
        self.angular_rows = self.cp.asarray(np.asarray(compact, dtype=np.int32))
        self.angular_coeff = self.cp.asarray(self.coeff)
        self.angular_ranges, self.angular_identity = ranges, identity
        (self.out / "fused_angular.json").write_text(
            json.dumps(
                dict(
                    source_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    ordered_angular_rows=len(compact),
                    disjoint_subgroups=len(ranges),
                    full_device_payload_bound_bytes=bound,
                    device_cap_bytes=self.cap,
                    scope="every native row retained; original subgroup order; unique output channels per subgroup; one FP64 kernel without atomics/fmad; no controlled timing/public adoption",
                ),
                indent=2,
            )
            + "\n"
        )

    def compute(self, energy, numerator, prime, value, derivative, *, response=True):
        before = time.perf_counter()
        cp = self.cp
        if numerator.shape != self.expected_shape:
            raise ValueError("actual complete native CUDA COMMON shape differs")
        for array in (
            (numerator, prime, value, derivative) if response else (numerator, value)
        ):
            if (
                array.dtype != np.float64
                or array.shape != numerator.shape
                or not array.flags.f_contiguous
            ):
                raise ValueError(
                    "complete native Fortran float64 numerator/response/output required"
                )
        if not all(
            np.all(np.isfinite(a))
            for a in ((numerator, prime) if response else (numerator,))
        ) or not np.isfinite(energy):
            raise ValueError("finite complete native CUDA inputs required")
        self.device.use()
        self.prepare(numerator.shape)
        source = cp.asarray(np.ascontiguousarray(numerator.transpose(2, 0, 1)))
        source_prime = (
            cp.asarray(np.ascontiguousarray(prime.transpose(2, 0, 1)))
            if response
            else None
        )
        output = cp.zeros_like(source)
        output_prime = cp.zeros_like(source) if response else None
        dummy = cp.empty(1, dtype=cp.float64)
        h = self.d["NHF"]
        for triple, group in itertools.groupby(
            enumerate(self.rows), lambda item: tuple(item[1][:3])
        ):
            im, ir, rank = triple
            members = list(group)
            p_m, p_r = self.phi[im - 1], self.phi[ir - 1]
            cm, cr = p_m.shape[1], p_r.shape[1]
            densities = cp.ascontiguousarray(
                (p_r[:, :, None] * p_m[:, None, :]).reshape(h, cr * cm)
            )
            potential = self.density.apply(densities, int(rank))
            for core, sub in itertools.groupby(
                members, lambda item: tuple(item[1][3:7])
            ):
                j, iss, _, index1 = core
                actions = list(sub)
                p_s = self.phi[iss - 1]
                cs = p_s.shape[1]
                sm, sr, ss = (
                    self.starts[im - 1],
                    self.starts[ir - 1],
                    self.starts[iss - 1],
                )
                inverse = 1.0 / (
                    energy
                    + self.core_e[j - 1]
                    - self.energies[ir - 1][:, None]
                    - self.energies[iss - 1][None, :]
                )
                amplitude = source[index1 - 1, sr : sr + cr, ss : ss + cs] * inverse
                v1 = cp.ascontiguousarray(p_s @ amplitude.T)
                dv1 = dummy
                if response:
                    rho_prime = (
                        source_prime[index1 - 1, sr : sr + cr, ss : ss + cs] * inverse
                        - amplitude * inverse
                    )
                    dv1 = cp.ascontiguousarray(p_s @ rho_prime.T)
                uv = cp.empty((h, cm), dtype=cp.float64)
                duv = cp.empty_like(uv) if response else dummy
                self.uv(
                    ((h * cm + 127) // 128,),
                    (128,),
                    (
                        potential,
                        v1,
                        dv1,
                        uv,
                        duv,
                        np.int32(h),
                        np.int32(cr),
                        np.int32(cm),
                        np.int32(response),
                    ),
                )
                selection = tuple(range(self.d["NK"]))
                if self.projection_policy == "reachable":
                    selection = tuple(
                        dict.fromkeys(int(row[7]) - 1 for _, row in actions)
                    )
                fields, offsets = self.projection_fields[selection]
                projection_all = uv.T @ fields
                if response:
                    projection_prime_all = duv.T @ fields
                start, end = self.angular_ranges[(tuple(triple), tuple(core))]
                width = fields.shape[1]
                slots = (end - start) * cm * self.d["NXX"]
                self.angular_kernel(
                    ((slots + 127) // 128,),
                    (128,),
                    (
                        projection_all,
                        projection_prime_all if response else dummy,
                        self.angular_rows[start:end],
                        self.angular_coeff[start:end],
                        output,
                        output_prime if response else dummy,
                        np.int32(end - start),
                        np.int32(cm),
                        np.int32(self.d["NXX"]),
                        np.int32(sm),
                        np.int32(width),
                        np.int32(response),
                    ),
                )
        value[:] = cp.asnumpy(output).transpose(1, 2, 0)
        if response:
            derivative[:] = cp.asnumpy(output_prime).transpose(1, 2, 0)
        if not all(
            np.all(np.isfinite(a))
            for a in ((value, derivative) if response else (value,))
        ):
            raise AssertionError("complete actual native CUDA term2 output nonfinite")
        self.calls += 1
        with (self.out / "cuda_full_observations.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        call=self.calls,
                        state=self.state,
                        response=response,
                        seconds=time.perf_counter() - before,
                        scope="complete native angular FP64 CUDA term2 including input/output copies",
                    )
                )
                + "\n"
            )


class NativeFusedCUDAControl(FusedProjectionMixin, NativeBatchedCUDAControl):
    pass


class BoundFusedCUDAControl(FusedProjectionMixin, BoundBatchedCUDAControl):
    pass
