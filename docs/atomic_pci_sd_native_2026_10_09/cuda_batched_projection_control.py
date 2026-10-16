"""Private complete concatenated GPU projections; original angular order kept.

Every virtual partial is retained in one immutable field snapshot. Larger GEMM
projects each uv/d_uv once, then original signed angular rows consume exact
partial slices in unchanged order. No public backend or performance claim.
"""

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from cuda_full_term2_control import NativeCUDAFullTerm2Control
from cuda_valence_bridge import BoundCUDAValenceControl


class BatchedProjectionMixin:
    def __init__(self, *args, projection_policy="all", **kwargs):
        if projection_policy not in ("all", "reachable"):
            raise ValueError(
                "explicit complete/all or exactly reachable projections required"
            )
        self.projection_policy = projection_policy
        self.projection_identity = None
        self.projection_fields = {}
        super().__init__(*args, **kwargs)

    def prepare(self, shape):
        super().prepare(shape)
        self.prepare_projection()

    def prepare_projection(self):
        identity = (
            self.library_identity,
            str(self.out),
            self.projection_policy,
            *self.state,
        )
        if identity == self.projection_identity:
            return
        self.projection_fields = {}
        cp, d = self.cp, self.d
        total = sum(a.shape[1] for a in self.phi)
        if len(self.phi) != d["NK"] or not 0 < total <= d["NXX"] * d["NK"]:
            raise ValueError("complete owned projection fields differ")
        selections = {tuple(range(d["NK"]))}
        if self.projection_policy == "reachable":
            selections = set()
            for _, triple in itertools.groupby(self.rows, lambda row: tuple(row[:3])):
                for _, core in itertools.groupby(triple, lambda row: tuple(row[3:7])):
                    selections.add(
                        tuple(dict.fromkeys(int(row[7]) - 1 for row in core))
                    )
        if not selections or any(
            not group or any(not 0 <= i < d["NK"] for i in group)
            for group in selections
        ):
            raise ValueError("complete native angular projection coverage differs")
        widths = {
            group: sum(self.phi[i].shape[1] for i in group) for group in selections
        }
        additional = 8 * (
            d["NHF"] * sum(widths.values()) + 4 * d["NXX"] * max(widths.values())
        )
        baseline = self.density.preflight(d["NXX"] ** 2) + 8 * (
            d["NHF"] * d["NX"] * d["NK"]
            + 6 * np.prod(self.expected_shape)
            + 8 * d["NHF"] * d["NXX"]
            + 3 * d["NXX"] ** 2
        )
        bound = int(baseline + additional)
        free, _ = cp.cuda.runtime.memGetInfo()
        if bound > self.cap or bound > free:
            raise MemoryError("complete batched projection cap/free preflight failed")
        for group in sorted(selections):
            offsets = np.cumsum([0, *(self.phi[i].shape[1] for i in group)])
            # Singleton uses the immutable original field without a needless copy.
            fields = (
                self.phi[group[0]]
                if len(group) == 1
                else cp.concatenate([self.phi[i] for i in group], axis=1)
            )
            self.projection_fields[group] = (
                fields,
                {
                    i: (int(offsets[j]), int(offsets[j + 1]))
                    for j, i in enumerate(group)
                },
            )
        self.projection_identity = identity
        (self.out / "batched_projection.json").write_text(
            json.dumps(
                dict(
                    source_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    complete_virtual_columns=total,
                    projection_policy=self.projection_policy,
                    exact_native_reachable_groups=len(selections),
                    cached_field_columns=sum(widths.values()),
                    device_bound_bytes=bound,
                    device_cap_bytes=self.cap,
                    state=self.state,
                    policy="complete FP64 concatenated field projections; original ordered angular accumulation; invalidate at each owned context; no controlled performance claim",
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
                for inn, target in itertools.groupby(
                    actions, lambda item: int(item[1][7])
                ):
                    cn, sn = self.phi[inn - 1].shape[1], self.starts[inn - 1]
                    a, b = offsets[inn - 1]
                    projection = projection_all[:, a:b]
                    if response:
                        projection_prime = projection_prime_all[:, a:b]
                    for ordinal, row in target:
                        if tuple(map(int, row[9:])) != (cm, cr, cs, cn):
                            raise AssertionError(
                                "complete native projection dimensions differ"
                            )
                        index = int(row[8]) - 1
                        output[index, sm : sm + cm, sn : sn + cn] += (
                            projection * self.coeff[ordinal]
                        )
                        if response:
                            output_prime[index, sm : sm + cm, sn : sn + cn] += (
                                projection_prime * self.coeff[ordinal]
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


class NativeBatchedCUDAControl(BatchedProjectionMixin, NativeCUDAFullTerm2Control):
    pass


class BoundBatchedCUDAControl(BatchedProjectionMixin, BoundCUDAValenceControl):
    pass
