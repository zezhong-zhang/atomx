"""Private binary128 arithmetic control over the same valence equations."""

import json
import subprocess
from pathlib import Path

from build_wide_valence_control import build_wide, digest


def build_quad(directory):
    directory = Path(directory)
    build_wide(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    wrapper = (directory / "wide_valence.f90").read_text()
    declaration = "real(c_long_double),value,intent(in) :: energy"
    if wrapper.count(declaration) != 6:
        raise AssertionError("all six actual bound energy declarations required")
    # Binary128 has no compatible NumPy/ctypes scalar on this x86 host. Pass
    # the exact scalar by reference, like the array storage, without narrowing.
    wrapper = wrapper.replace(declaration, "real(c_long_double),intent(in) :: energy")
    diagrams = (directory / "wide_valence_diagrams.f").read_text()
    helpers = (directory / "wide_density.f90").read_text()
    conversion = """
subroutine atomx_quad_pack(n,width,characters,values) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int),value :: n,width
  character(c_char),intent(in) :: characters(width,n)
  real(c_float128),intent(out) :: values(n)
  character(len=width) :: word
  integer i,j,status
  do i=1,n
    do j=1,width
      word(j:j)=characters(j,i)
    enddo
    read(word,*,iostat=status) values(i)
    if(status/=0) error stop 'invalid explicit binary128 decimal input'
  enddo
end subroutine

subroutine atomx_quad_unpack(n,width,values,characters) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int),value :: n,width
  real(c_float128),intent(in) :: values(n)
  character(c_char),intent(out) :: characters(width,n)
  character(len=width) :: word
  integer i,j,status
  if(width<48) error stop 'binary128 decimal output width below48'
  do i=1,n
    write(word,'(ES48.36E4)',iostat=status) values(i)
    if(status/=0) error stop 'invalid explicit binary128 decimal output'
    do j=1,width
      characters(j,i)=word(j:j)
    enddo
  enddo
end subroutine
"""
    for name, body in (
        ("quad_valence.f90", wrapper),
        ("quad_valence_diagrams.f", diagrams),
        ("quad_density.f90", helpers),
    ):
        body = (
            body.replace("wide_", "quad_")
            .replace("c_long_double", "c_float128")
            .replace("real*10", "real*16")
        )
        if name == "quad_density.f90":
            body += conversion
        (directory / name).write_text(body)
    command = manifest["command"][:]
    flags = command[: command.index("-shared")]
    compile_command = flags + [
        "-c",
        "quad_valence.f90",
        "quad_valence_diagrams.f",
        "quad_density.f90",
    ]
    command[command.index("-o") : command.index("-o")] = [
        "quad_valence.o",
        "quad_valence_diagrams.o",
        "quad_density.o",
    ]
    with (directory / "quad_compiler.log").open("w") as log:
        for invocation in (compile_command, command):
            subprocess.run(
                invocation,
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    manifest["compile_commands"].extend([compile_command, command])
    manifest["command"] = command
    manifest["private_quad_valence"] = dict(
        policy="same14original valence routines, binary128 owned/local/density arithmetic; core fields/angular helpers retain original double ABI",
        transport="explicit113-significant-bit scalar/array storage by reference, decimal pack/unpack; no NumPy longdouble reinterpretation",
        scope="private finite arithmetic reference, neither new physical coverage nor scalable public backend",
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
