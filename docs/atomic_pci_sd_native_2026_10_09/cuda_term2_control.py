"""Private native term2 GPU-density bridge; original full angular/response path."""

import ctypes
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent

BRIDGE = """module cuda_density_bridge
  use iso_c_binding
  use native_sd_bridge, only: nhf,nx,nk,nxx
  implicit none
  type(c_funptr) :: density_callback=c_null_funptr
  abstract interface
    subroutine cuda_callback(h,x,k,xx,im,ir,rank,nnm,nnr,cm,cr,phi,uu,status) bind(C)
      import c_int,c_double
      integer(c_int),value :: h,x,k,xx,im,ir,rank,nnm,nnr,cm,cr
      real(c_double),intent(in) :: phi(*)
      real(c_double),intent(out) :: uu(*)
      integer(c_int),intent(out) :: status
    end subroutine
  end interface
contains
  subroutine atomx_cuda_density_bind(callback) bind(C)
    type(c_funptr),value :: callback
    if(.not.c_associated(callback)) error stop 'explicit CUDA callback required'
    density_callback=callback
  end subroutine

  subroutine cuda_phi_prepare(phi)
    real(c_double),intent(in) :: phi(nhf,nx,nk)
    real(c_double) dummy(1)
    integer(c_int) status
    procedure(cuda_callback),pointer :: callback
    if(.not.c_associated(density_callback)) error stop 'CUDA density backend unbound'
    call c_f_procpointer(density_callback,callback)
    call callback(nhf,nx,nk,nxx,-1,-1,0,0,0,0,0,phi,dummy,status)
    if(status/=0) error stop 'complete CUDA phi preparation failed; no CPU fallback'
  end subroutine

  subroutine cuda_pair_action(phi,uu,im,ir,rank,nnm,nnr,cm,cr)
    real(c_double),intent(in) :: phi(nhf,nx,nk)
    real(c_double),intent(inout) :: uu(nhf,nxx,nxx)
    integer,intent(in) :: im,ir,rank,nnm,nnr,cm,cr
    integer(c_int) status
    procedure(cuda_callback),pointer :: callback
    if(.not.c_associated(density_callback)) error stop 'CUDA density backend unbound'
    call c_f_procpointer(density_callback,callback)
    call callback(nhf,nx,nk,nxx,im,ir,rank,nnm,nnr,cm,cr,phi,uu,status)
    if(status/=0) error stop 'complete CUDA density action failed; no CPU fallback'
  end subroutine
end module
"""


def kernel_source(original):
    """Replace one exact full virtual-pair action; retain all other arithmetic."""
    loop = """        !$omp do collapse(2) schedule(static)
        do m=nnm,nmaxx(im)
          do nr=nnr,nmaxx(ir)
            v=phi(:,nr,ir)*phi(:,m,im)
            call density_action(v,u,l)
            uu(:,nr,m)=u
          enddo
        enddo
        !$omp end do"""
    if original.count(loop) != 1 or original.count("  phi=gg+ff\n") != 1:
        raise AssertionError("one complete exact native pair loop required")
    original = original.replace(
        "  use native_valence_bridge\n",
        "  use native_valence_bridge\n  use cuda_density_bridge, only: cuda_phi_prepare,cuda_pair_action\n",
        1,
    )
    original = original.replace(
        "  phi=gg+ff\n", "  phi=gg+ff\n  call cuda_phi_prepare(phi)\n"
    )
    return original.replace(
        loop,
        """        !$omp single
        call cuda_pair_action(phi,uu,im,ir,l,nnm,nnr,nmaxx(im)-nnm+1,nmaxx(ir)-nnr+1)
        !$omp end single""",
    )


class NativeCUDADensityControl:
    """Resident immutable complete phi and shared native weighted geometry."""

    def __init__(
        self, feature, dimensions, out, *, memory_bytes=1024**3, fortran_output=False
    ):
        sys.path.append(str(OWNER / "cuda_python_runtime"))
        import cupy as cp
        from cuda_density_control import CUDADensityControl

        self.cp, self.out, self.dimensions = cp, Path(out), dict(dimensions)
        self.fortran_output = bool(fortran_output)
        self.device = cp.cuda.Device(0)
        self.device.use()
        cp.get_default_memory_pool().set_limit(size=memory_bytes)
        self.control = CUDADensityControl(feature, memory_bytes=memory_bytes)
        phi_bytes = 8 * dimensions["NHF"] * dimensions["NX"] * dimensions["NK"]
        bound = self.control.preflight(dimensions["NXX"] ** 2) + phi_bytes
        if self.fortran_output:
            bound += 8 * dimensions["NHF"] * dimensions["NXX"] ** 2
        if bound > memory_bytes:
            raise MemoryError("complete resident phi plus pair batch exceeds cap")
        if len(feature.radii) != dimensions["NHF"]:
            raise ValueError("complete GPU/native tagged coordinate identity differs")
        self.phi_hash = None
        self.phi = None
        self.calls = 0
        properties = cp.cuda.runtime.getDeviceProperties(0)
        runtime = dict(
            cupy_version=cp.__version__,
            numpy_version=np.__version__,
            numpy_path=np.__file__,
            cuda_path=os.environ.get("CUDA_PATH"),
            kernel_cache=os.environ.get("CUPY_CACHE_DIR"),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            device_name=properties["name"].decode(),
            device_capability=self.device.compute_capability,
            actual_device_total_bytes=int(properties["totalGlobalMem"]),
            device_cap_bytes=memory_bytes,
            complete_resident_device_bound_bytes=bound,
            feature_identity=dict(feature.identity),
            device_to_host_layout="Fortran contiguous"
            if self.fortran_output
            else "C contiguous",
            device_report=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap,utilization.gpu",
                    "--format=csv",
                ],
                text=True,
            ),
            packages={
                name: importlib.metadata.version(name)
                for name in (
                    "cupy-cuda13x",
                    "cuda-toolkit",
                    "nvidia-cublas",
                    "nvidia-cuda-nvrtc",
                    "nvidia-cuda-runtime",
                    "nvidia-nvjitlink",
                )
            },
        )
        (self.out / "cuda_runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        callback_type = ctypes.CFUNCTYPE(
            None,
            *([ctypes.c_int] * 11),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        )

        def callback(h, x, k, xx, im, ir, rank, nnm, nnr, cm, cr, phi, uu, status):
            status[0] = 1
            before = time.perf_counter()
            try:
                if (h, x, k, xx) != tuple(
                    dimensions[name] for name in ("NHF", "NX", "NK", "NXX")
                ):
                    raise ValueError("actual complete native CUDA dimensions changed")
                self.device.use()
                if im == -1 and ir == -1:
                    actual = np.ctypeslib.as_array(phi, shape=(h * x * k,)).reshape(
                        h, x, k, order="F"
                    )
                    current_hash = hashlib.sha256(actual.tobytes(order="F")).hexdigest()
                    if self.phi_hash is None:
                        self.phi = cp.asarray(np.ascontiguousarray(actual))
                        self.phi_hash = current_hash
                    elif current_hash != self.phi_hash:
                        raise ValueError("resident complete phi input/source changed")
                    entry = dict(
                        stage="complete_gpu_phi_binding",
                        phi_sha256=current_hash,
                        dimensions=self.dimensions,
                        seconds=time.perf_counter() - before,
                    )
                else:
                    if self.phi is None or not (
                        1 <= im <= k
                        and 1 <= ir <= k
                        and 0 <= rank <= dimensions["KK"]
                        and 1 <= nnm <= xx
                        and 1 <= nnr <= xx
                        and cm > 0
                        and cr > 0
                        and nnm - 1 + cm <= xx
                        and nnr - 1 + cr <= xx
                    ):
                        raise ValueError("unbound/incomplete native GPU pair range")
                    self.control.preflight(cm * cr)
                    left = self.phi[:, nnr - 1 : nnr - 1 + cr, ir - 1]
                    right = self.phi[:, nnm - 1 : nnm - 1 + cm, im - 1]
                    density = cp.ascontiguousarray(
                        (left[:, :, None] * right[:, None, :]).reshape(h, cr * cm)
                    )
                    acted = self.control.apply(density, rank)
                    enqueued_at = time.perf_counter()
                    if self.fortran_output:
                        packed = cp.asfortranarray(acted.reshape(h, cr, cm))
                        actual = cp.asnumpy(packed, order="F")
                        if not actual.flags.f_contiguous:
                            raise AssertionError(
                                "explicit native Fortran output layout lost"
                            )
                    else:
                        actual = cp.asnumpy(acted).reshape(h, cr, cm)
                    copied_at = time.perf_counter()
                    if not np.all(np.isfinite(actual)):
                        raise RuntimeError("nonfinite complete actual CUDA pair action")
                    result = np.ctypeslib.as_array(uu, shape=(h * xx * xx,)).reshape(
                        h, xx, xx, order="F"
                    )
                    result[:, nnr - 1 : nnr - 1 + cr, nnm - 1 : nnm - 1 + cm] = actual
                    self.calls += 1
                    entry = dict(
                        stage="complete_gpu_pair_action",
                        call=self.calls,
                        im=im,
                        ir=ir,
                        rank=rank,
                        pairs=cm * cr,
                        seconds=time.perf_counter() - before,
                        enqueue_seconds=enqueued_at - before,
                        synchronized_download_seconds=copied_at - enqueued_at,
                        finite_check_and_host_copy_seconds=time.perf_counter()
                        - copied_at,
                    )
                with (self.out / "cuda_observations.jsonl").open("a") as log:
                    log.write(json.dumps(entry) + "\n")
                status[0] = 0
            except BaseException as error:
                (self.out / "cuda_callback_failure.json").write_text(
                    json.dumps(
                        dict(type=type(error).__name__, message=str(error)), indent=2
                    )
                    + "\n"
                )

        self.callback = callback_type(callback)
        self.callback_type = callback_type

    def bind(self, library):
        function = library.atomx_cuda_density_bind
        function.argtypes = [self.callback_type]
        function.restype = None
        function(self.callback)
