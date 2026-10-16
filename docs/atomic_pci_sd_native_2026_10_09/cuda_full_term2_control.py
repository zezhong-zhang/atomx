"""Private whole term2 CUDA contraction; native ordered angular plan, FP64 only.

All angular coefficients and reachable channels are emitted by a mechanical
specialization of the retained CPU kernel. No fitted/screened effective model.
"""

import ctypes
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent


def plan_source(original):
    """Emit the same native angular loop twice: count first, then fill all rows."""
    core = original.split("recursive subroutine valence_term2_streamed_core(", 1)[1]
    core = core.split("subroutine atomx_val_term2_streamed_map", 1)[0]
    declarations = "\n".join(
        line
        for line in core.splitlines()
        if line.startswith("  integer ")
        or line.startswith("  real(c_double) coef")
        or line.startswith("  real(c_double),external")
    )
    body = "  ka=vk" + core.split("  ka=vk", 1)[1]
    start = body.index("      do nss=nns,nmaxx(iss)")
    stop = body.index("      do inn=1,nk", start)
    body = body[:start] + body[stop:]
    start = body.index("        call dgemm('T','N'")
    stop = body.index("        coef=", start)
    body = body[:start] + body[stop:]
    accumulation = """          do n=1,cn
            do m=1,cm
              xvo(m+nnm-1,n+nnn-1,index)=xvo(m+nnm-1,n+nnn-1,index)+projected(m,n)*cfin
              if(calculate_response) dxvo(m+nnm-1,n+nnn-1,index)=dxvo(m+nnm-1,n+nnn-1,index)+dprojected(m,n)*cfin
            enddo
          enddo"""
    if body.count(accumulation) != 1:
        raise AssertionError("one original full native angular accumulation required")
    body = body.replace(
        accumulation,
        "          call record_row([im,ir,l,j,iss,k1,index1,inn,index,cm,cr,cs,cn],cfin)",
    )
    module = """module cuda_term2_plan
  use iso_c_binding
  use native_valence_bridge
  implicit none
  integer(c_int64_t) :: emitted=0,limit=0
  logical :: fill=.false.
  integer(c_int),pointer :: plan_rows(:,:)=>null()
  real(c_double),pointer :: plan_coeff(:)=>null()
contains
  subroutine record_row(row,coefficient)
    integer(c_int),intent(in) :: row(13)
    real(c_double),intent(in) :: coefficient
    emitted=emitted+1
    if(fill) then
      if(emitted>limit) error stop 'complete CUDA angular plan capacity exceeded'
      plan_rows(:,emitted)=row; plan_coeff(emitted)=coefficient
    endif
  end subroutine

  subroutine atomx_cuda_plan(phase,capacity,count,rows,coefficients) bind(C)
    integer(c_int),value :: phase
    integer(c_int64_t),value :: capacity
    integer(c_int64_t),intent(out) :: count
    integer(c_int),intent(out),target :: rows(13,capacity)
    real(c_double),intent(out),target :: coefficients(capacity)
    integer im,ir,j,km,kr,kapm,kapr,lm,lr,jm,jr,m0,r0,nnm,nnr,l,lmin,lhigh
    if(phase/=0.and.phase/=1) error stop 'explicit CUDA plan count/fill phase required'
    fill=phase==1; emitted=0; limit=capacity
    if(fill) then
      plan_rows=>rows; plan_coeff=>coefficients
    endif
    do im=1,nk
      call indk1(im,km,kapm,lm,jm,m0); call st(km,nnm)
      if(nmaxx(im)<nnm) cycle
      do ir=1,nk
        call indk1(ir,kr,kapr,lr,jr,r0); call st(kr,nnr)
        if(nmaxx(ir)<nnr) cycle
        lmin=abs(kapm-kapr); lhigh=kapm+kapr-1
        do l=lmin,lhigh
          if(mod(lm+lr+l,2)/=0) cycle
          do j=1,nn
            call cuda_plan_core(j,im,ir,l)
          enddo
        enddo
      enddo
    enddo
    count=emitted
    if(fill.and.emitted/=capacity) error stop 'incomplete CUDA angular plan'
    nullify(plan_rows,plan_coeff)
  end subroutine

  subroutine atomx_cuda_column(state) bind(C)
    integer(c_int),intent(out) :: state(2)
    state=[vp,vk]
  end subroutine

  subroutine atomx_cuda_fields(phi,energies,core_e,starts,ends) bind(C)
    real(c_double),intent(out) :: phi(nhf,nx,nk),energies(nx,nk),core_e(nn)
    integer(c_int),intent(out) :: starts(nk),ends(nk)
    integer im,km,kapm,lm,jm,m0
    phi=gg(:,:,:nk)+ff(:,:,:nk); energies=ee(:,:nk); core_e=eo(:nn); ends=nmaxx(:nk)
    do im=1,nk
      call indk1(im,km,kapm,lm,jm,m0); call st(km,starts(im))
      if(starts(im)/=first_virtual(im)) error stop 'complete native virtual range differs'
    enddo
  end subroutine
end module

recursive subroutine cuda_plan_core(j,im,ir,l)
  use native_valence_bridge
  use cuda_term2_plan, only: record_row
  implicit none
  integer,intent(in) :: j,im,ir,l
"""
    return module + declarations + "\n" + body


UV_KERNEL = r"""
extern "C" __global__ void ordered_uv(
 const double *uu,const double *v,const double *dv,double *u,double *du,
 int nodes,int cr,int cm,int response) {
 long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(tid>=(long long)nodes*cm) return;
 int i=tid/cm,m=tid%cm;
 double value=0.0,prime=0.0;
 for(int r=0;r<cr;++r) {
   double potential=uu[((long long)i*cr+r)*cm+m];
   value=value+v[(long long)i*cr+r]*potential;
   if(response) prime=prime+dv[(long long)i*cr+r]*potential;
 }
 u[tid]=value; if(response) du[tid]=prime;
}
"""


class NativeCUDAFullTerm2Control:
    """No intermediate host transfers; CPU native angular plan remains exact."""

    def __init__(self, feature, dimensions, out, library, *, memory_bytes=2 * 1024**3):
        sys.path.append(str(OWNER / "cuda_python_runtime"))
        import cupy as cp
        from cuda_density_control import CUDADensityControl

        self.cp, self.d, self.out, self.lib = cp, dict(dimensions), Path(out), library
        self.device = cp.cuda.Device(0)
        self.device.use()
        cp.get_default_memory_pool().set_limit(size=memory_bytes)
        self.density = CUDADensityControl(feature, memory_bytes=memory_bytes)
        self.uv = cp.RawKernel(UV_KERNEL, "ordered_uv", options=("--fmad=false",))
        self.cap, self.state, self.calls = memory_bytes, None, 0
        self.library_identity = hashlib.sha256(
            Path(library._name).read_bytes()
        ).hexdigest()
        manifest = json.loads(
            (Path(library._name).parent / "manifest.json").read_text()
        )
        if manifest["dimensions"] != self.d:
            raise ValueError("actual native CUDA dimensions/source binding differs")
        self.expected_shape = (
            self.d["NXX"],
            self.d["NXX"],
            manifest["valence_sd"]["nvh"],
        )
        self.plan_cap_bytes = 128 * 1024**2
        props = cp.cuda.runtime.getDeviceProperties(0)
        (self.out / "cuda_runtime.json").write_text(
            json.dumps(
                dict(
                    cupy_version=cp.__version__,
                    numpy_version=np.__version__,
                    numpy_path=np.__file__,
                    cuda_driver=cp.cuda.runtime.driverGetVersion(),
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                    device_name=props["name"].decode(),
                    actual_device_total_bytes=int(props["totalGlobalMem"]),
                    device_cap_bytes=self.cap,
                    cuda_path=os.environ.get("CUDA_PATH"),
                    kernel_cache=os.environ.get("CUPY_CACHE_DIR"),
                    device_report=subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap,utilization.gpu",
                            "--format=csv",
                        ],
                        text=True,
                    ),
                    scope="private complete GPU FP64 contractions; no CPU fallback/production adoption",
                ),
                indent=2,
            )
            + "\n"
        )

    def prepare(self, shape):
        cp, d = self.cp, self.d
        state = np.zeros(2, dtype=np.int32)
        integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")
        double = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        column = self.lib.atomx_cuda_column
        column.argtypes, column.restype = [integer], None
        column(state)
        current = tuple(map(int, state))
        if current == self.state:
            return
        plan = self.lib.atomx_cuda_plan
        plan.argtypes = [
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            integer,
            double,
        ]
        plan.restype = None
        count = ctypes.c_int64()
        plan(
            0,
            0,
            ctypes.byref(count),
            np.empty((13, 0), dtype=np.int32, order="F"),
            np.empty(0),
        )
        if count.value <= 0 or count.value * 60 > self.plan_cap_bytes:
            raise MemoryError(
                "complete native angular plan exceeds predeclared128MiB cap"
            )
        rows = np.zeros((13, count.value), dtype=np.int32, order="F")
        coeff = np.zeros(count.value)
        plan(1, count.value, ctypes.byref(count), rows, coeff)
        if not np.all(np.isfinite(coeff)):
            raise AssertionError("complete native angular coefficients nonfinite")
        h, x, k = d["NHF"], d["NX"], d["NK"]
        phi, energies, core_e = (
            np.empty((h, x, k), order="F"),
            np.empty((x, k), order="F"),
            np.empty(d["NN"]),
        )
        starts, ends = (np.zeros(k, dtype=np.int32) for _ in range(2))
        fields = self.lib.atomx_cuda_fields
        fields.argtypes, fields.restype = (
            [double, double, double, integer, integer],
            None,
        )
        fields(phi, energies, core_e, starts, ends)
        bound = self.density.preflight(d["NXX"] ** 2) + 8 * (
            h * x * k + 6 * np.prod(shape) + 8 * h * d["NXX"] + 3 * d["NXX"] ** 2
        )
        if bound > self.cap:
            raise MemoryError(
                "complete CUDA input/output/projection payload exceeds cap"
            )
        if not all(np.all(np.isfinite(a)) for a in (phi, energies, core_e)):
            raise AssertionError("actual complete native CUDA fields nonfinite")
        self.starts, self.ends = tuple(starts - 1), tuple(ends)
        self.phi = tuple(
            cp.asarray(np.ascontiguousarray(phi[:, a:b, i]))
            for i, (a, b) in enumerate(zip(self.starts, self.ends, strict=True))
        )
        self.energies = tuple(
            cp.asarray(energies[a:b, i])
            for i, (a, b) in enumerate(zip(self.starts, self.ends, strict=True))
        )
        self.core_e = core_e.copy()
        self.rows = rows.T.copy()
        self.coeff = coeff.copy()
        np.savez_compressed(
            self.out / f"cuda_full_fields_{current[0]}_{current[1]}.npz",
            phi=phi,
            energies=energies,
            core_e=core_e,
            starts=starts,
            ends=ends,
            rows=rows,
            coefficients=coeff,
        )
        (self.out / f"cuda_full_plan_{current[0]}_{current[1]}.json").write_text(
            json.dumps(
                dict(
                    state=current,
                    plan_rows=count.value,
                    device_bound_bytes=int(bound),
                    device_cap_bytes=self.cap,
                    host_plan_cap_bytes=self.plan_cap_bytes,
                    library_sha256=self.library_identity,
                    scope="native complete ordered angular plan; no amplitude/coefficient screening; full device FP64 contractions",
                ),
                indent=2,
            )
            + "\n"
        )
        self.state = current

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
                for inn, target in itertools.groupby(
                    actions, lambda item: int(item[1][7])
                ):
                    p_n = self.phi[inn - 1]
                    cn, sn = p_n.shape[1], self.starts[inn - 1]
                    projection = uv.T @ p_n
                    if response:
                        projection_prime = duv.T @ p_n
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
