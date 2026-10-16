"""Private additive whole-GPU term2 in complete native SD maps, explicit policy."""

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from cuda_full_term2_control import NativeCUDAFullTerm2Control, plan_source

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]

BRIDGE = """module cuda_full_term2_bridge
  use iso_c_binding
  use native_valence_bridge
  implicit none
  type(c_funptr) :: term2_callback=c_null_funptr
  integer :: selected=0
  abstract interface
    subroutine cuda_callback(energy,response,xx,channels,d,dd,rd,drd,status) bind(C)
      import c_int,c_double
      real(c_double),value :: energy
      integer(c_int),value :: response,xx,channels
      real(c_double),intent(in) :: d(*),dd(*)
      real(c_double),intent(out) :: rd(*),drd(*)
      integer(c_int),intent(out) :: status
    end subroutine
  end interface
contains
  subroutine atomx_cuda_term2_bind(callback) bind(C)
    type(c_funptr),value :: callback
    if(.not.c_associated(callback)) error stop 'explicit whole CUDA callback required'
    term2_callback=callback
  end subroutine
  subroutine atomx_cuda_term2_select(policy) bind(C)
    integer(c_int),value :: policy
    if(policy/=0.and.policy/=1) error stop 'explicit CPU/CUDA policy required'
    selected=policy
  end subroutine
end module

subroutine valence_term2_dispatch(energy,calculate_response)
  use cuda_full_term2_bridge
  implicit none
  real(c_double),intent(in) :: energy
  logical,intent(in) :: calculate_response
  real(c_double),allocatable :: value(:,:,:),prime(:,:,:)
  integer(c_int) status,request
  procedure(cuda_callback),pointer :: callback
  if(selected==0) then
    call valence_term2_streamed(energy,calculate_response)
    return
  endif
  if(.not.c_associated(term2_callback)) error stop 'whole CUDA term2 unbound'
  allocate(value(nxx,nxx,nvh),prime(nxx,nxx,nvh))
  call c_f_procpointer(term2_callback,callback)
  request=0
  if(calculate_response) request=1
  call callback(energy,request,nxx,nvh,xvi,dxvi,value,prime,status)
  if(status/=0) error stop 'whole CUDA term2 failed; no CPU fallback'
  ! Preserve every diagram already accumulated in the native map.
  xvo=xvo+value
  if(calculate_response) dxvo=dxvo+prime
  deallocate(value,prime)
end subroutine
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(directory):
    """Compile a source-recorded private dispatcher before the image is loaded."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    term2 = manifest["valence_sd"]["term2"]
    if term2["mode"] != "streamed_blas" or term2["threads"] != 8:
        raise ValueError("complete original eight-worker term2 required")
    wrapper = directory / "_pci_valence.f90"
    original = wrapper.read_text()
    if (
        original.count("call valence_term2_streamed(energy,.true.)") != 3
        or original.count("call valence_term2_streamed(energy,.false.)") != 1
    ):
        raise ValueError("complete three-response/one-value call coverage required")
    retained = directory / "pre_cuda_valence.f90.txt"
    retained.write_text(original)
    wrapper.write_text(
        original.replace("call valence_term2_streamed(", "call valence_term2_dispatch(")
    )
    kernel = ROOT / "src/atomx/methods/_pci_valence_term2_streamed.f90"
    (directory / "cuda_term2_plan.f90").write_text(plan_source(kernel.read_text()))
    (directory / "cuda_full_term2_bridge.f90").write_text(BRIDGE)
    command = manifest["command"][:]
    flags = command[: command.index("-shared")]
    compile_command = flags + [
        "-c",
        wrapper.name,
        "cuda_term2_plan.f90",
        "cuda_full_term2_bridge.f90",
    ]
    command[command.index("-o") : command.index("-o")] = [
        "cuda_term2_plan.o",
        "cuda_full_term2_bridge.o",
    ]
    with (directory / "cuda_valence_compiler.log").open("w") as log:
        for invocation in (compile_command, command):
            subprocess.run(
                invocation,
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    d = manifest["dimensions"]
    nvh = manifest["valence_sd"]["nvh"]
    manifest["private_cuda_term2"] = dict(
        original_wrapper_sha256=digest(retained),
        derived_wrapper_sha256=digest(wrapper),
        bridge_sha256=digest(directory / "cuda_full_term2_bridge.f90"),
        native_plan_sha256=digest(directory / "cuda_term2_plan.f90"),
        additive=True,
        host_additional_payload_bytes=16 * d["NXX"] ** 2 * nvh,
        selection="explicit paired CPU/CUDA control, never automatic fallback",
        scope="private same-diagram additive FP64 GPU term2, full source/root/response gates still required; public CUDA capability not adopted",
    )
    manifest["compile_commands"].extend([compile_command, command])
    manifest["command"] = command
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class BoundCUDAValenceControl(NativeCUDAFullTerm2Control):
    """One source-bound ValenceSD owner; invalidate at each COMMON initializer."""

    def __init__(self, valence, out, *, memory_bytes=2 * 1024**3):
        self.valence, self.owner_out, self.epoch = valence, Path(out), 0
        self.owner_out.mkdir(parents=True, exist_ok=True)
        manifest = valence.core._manifest
        if len(valence.core.features.radii) != valence.core.plan["dimensions"]["NHF"]:
            raise ValueError("complete tagged CUDA/native radial coordinates differ")
        extra = manifest["private_cuda_term2"]["host_additional_payload_bytes"]
        if (
            valence.plan["complete_payload_bound_bytes"] + extra
            > valence.core.memory_bytes
        ):
            raise MemoryError("complete additive CUDA host payload exceeds cap")
        super().__init__(
            valence.core.features,
            valence.core.plan["dimensions"],
            self.owner_out,
            valence.core._backend,
            memory_bytes=memory_bytes,
        )
        double_pointer = ctypes.POINTER(ctypes.c_double)
        callback_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            *([double_pointer] * 4),
            ctypes.POINTER(ctypes.c_int),
        )

        def callback(
            energy, response, xx, channels, numerator, prime, value, derivative, status
        ):
            try:
                if response not in (0, 1) or (xx, xx, channels) != self.expected_shape:
                    raise ValueError(
                        "complete bound CUDA dimensions/response policy differ"
                    )
                shape = (xx, xx, channels)
                count = int(np.prod(shape))
                arrays = tuple(
                    np.ctypeslib.as_array(pointer, shape=(count,)).reshape(
                        shape, order="F"
                    )
                    for pointer in (numerator, prime, value, derivative)
                )
                self.compute(energy, *arrays, response=bool(response))
                status[0] = 0
            except BaseException as error:
                (self.owner_out / "callback_failure.json").write_text(
                    json.dumps(
                        dict(
                            error_type=type(error).__name__,
                            error=str(error),
                            epoch=self.epoch,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                status[0] = 1

        self.callback = callback_type(callback)
        binder = self.lib.atomx_cuda_term2_bind
        binder.argtypes, binder.restype = [callback_type], None
        binder(self.callback)
        self.selector = self.lib.atomx_cuda_term2_select
        self.selector.argtypes, self.selector.restype = [ctypes.c_int], None
        original = valence._init

        def initialize(*arguments):
            self.epoch += 1
            self.state = None
            for name in ("phi", "energies", "core_e", "rows", "coeff"):
                if hasattr(self, name):
                    delattr(self, name)
            self.out = self.owner_out / f"context_{self.epoch:06d}"
            self.out.mkdir(exist_ok=False)
            original(*arguments)
            (self.out / "binding.json").write_text(
                json.dumps(
                    dict(
                        core_identity_sha256=valence._core_identity,
                        core_amplitudes_sha256=valence._core_amplitudes_sha256,
                        slot=int(arguments[0]),
                        kappa=int(arguments[1]),
                        library_sha256=self.library_identity,
                        policy="fresh complete native fields/plan after every owned COMMON initialization",
                    ),
                    indent=2,
                )
                + "\n"
            )

        valence._init = initialize

    def select(self, variant):
        if variant not in ("cpu", "cuda"):
            raise ValueError("explicit cpu/cuda variant required")
        self.selector(int(variant == "cuda"))
