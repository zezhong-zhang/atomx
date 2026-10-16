"""Private exact standalone term31 terminal contraction specialization.

Retain original ordered physical-single trial densities, Coulomb potentials,
angular coefficients and all channels. Only the terminal P/Q radial sum is
factored into a full FP64 GEMM. This does not change the public builder/default.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scratch_bytes(dimensions):
    return 16 * dimensions["NHF"] * dimensions["NXX"] + 8 * dimensions["NHF"] * (
        dimensions["NX"] + 16
    )


def specialize(original):
    if original.lower().count("subroutine pci_val_term31") != 1:
        raise ValueError("complete original standalone singles diagram required")
    begin = original.index("                do 165 n=nnn,nmaxx(indn)")
    end = original.index("165             continue", begin) + len(
        "165             continue"
    )
    retained = original[begin:end]
    if (
        retained.count("do ") != 3
        or retained.count("v1(ij)*gg(ij,m,indm)") != 1
        or retained.count("v2(ij)*ff(ij,m,indm)") != 1
        or retained.count("uu(ij,n)*rp(ij)") != 1
        or retained.count("c*rint(vv,1,max,11,h)") != 1
    ):
        raise ValueError("exact complete term31 terminal P/Q sum required")
    replacement = """                cm=nmaxx(indm)-nnm+1
                cn=nmaxx(indn)-nnn+1
                if(cm.gt.0.and.cn.gt.0) then
                  do m=nnm,nmaxx(indm)
                    do ij=1,max
                      wm(ij,m)=v1(ij)*gg(ij,m,indm)
     *                           +v2(ij)*ff(ij,m,indm)
                    enddo
                  enddo
                  do n=nnn,nmaxx(indn)
                    do ij=1,max
                      wu(ij,n)=uu(ij,n)*rp(ij)
                    enddo
                  enddo
                  call dgemm('T','N',cm,cn,max,c,wm(1,nnm),
     *              nhf,wu(1,nnn),nhf,1.d0,xvo(nnm,nnn,index),nxx)
                endif"""
    result = original.replace(
        '      include "all.par"',
        '      include "all.par"\n      integer cm,cn\n      dimension wm(nhf,nxx),wu(nhf,nxx)',
        1,
    )
    return result.replace(retained, replacement, 1)


def install(directory, *, memory_bytes):
    """Before load, bind the private scratch and actual LP64 inner-one BLAS."""
    directory = Path(directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if "private_monovalent_term31_blas" in manifest:
        raise ValueError("one explicit private term31 specialization per fresh backend")
    if (
        manifest.get("monovalent_sd", {}).get("core_denominators")
        != "unshifted closed-core; all diagrams at bare ev"
        or manifest.get("blas", {}).get("runtime", {}).get("parallel") != 2
        or manifest["blas"]["runtime"].get("threads") != 1
    ):
        raise ValueError(
            "source-bound standalone and actual inner-one OpenMP BLAS required"
        )
    extra = scratch_bytes(manifest["dimensions"])
    if extra > memory_bytes:
        raise MemoryError(
            "complete private term31 static scratch exceeds explicit allowance"
        )
    source = directory / "monovalent_diagram.f"
    for filename, expected in manifest["files"].items():
        if Path(filename).name != filename or digest(directory / filename) != expected:
            raise ValueError(
                "actual pre-specialization backend source/artifact differs"
            )
    if digest(source) != manifest["monovalent_sd"]["diagram_sha256"]:
        raise ValueError("complete original pinned standalone diagram differs")
    retained = directory / "pre_term31_blas_monovalent_diagram.f.txt"
    original = source.read_text()
    retained.write_text(original)
    source.write_text(specialize(original))
    # Keep a callable original signed component in the same COMMON owner;
    # complete original/GEMM maps can be compared without another core replay.
    (directory / "monovalent_original_diagram.f").write_text(
        original.replace(
            "subroutine pci_val_term31", "subroutine pci_val_term31_original", 1
        )
    )
    control = """module native_monovalent_term31_control
  use native_valence_bridge
  implicit none
contains
  subroutine atomx_monovalent_term31_control(mode,s,result) bind(C)
    integer(c_int),value,intent(in) :: mode
    real(c_double),intent(in) :: s(nx)
    real(c_double),intent(out) :: result(nxx,nxx,nvh)
    rvi=s; drvi=0d0; xvo=0d0; rvo=0d0
    if(mode==0) then
      call pci_val_term31_original
    else if(mode==1) then
      call pci_val_term31
    else
      error stop 'explicit original or BLAS component required'
    endif
    result=xvo
  end subroutine
end module
"""
    (directory / "monovalent_term31_control.f90").write_text(control)
    commands = [c for c in manifest["compile_commands"] if "monovalent_diagram.f" in c]
    if len(commands) != 1 or "-fopenmp" in commands[0]:
        raise ValueError("original serial-static standalone compilation required")
    compiler = commands[0][: commands[0].index("-c")]
    compile_command = compiler + [
        "-c",
        "monovalent_diagram.f",
        "monovalent_original_diagram.f",
        "monovalent_term31_control.f90",
    ]
    link_command = manifest["command"][:]
    index = link_command.index("-o")
    link_command[index:index] = [
        "monovalent_original_diagram.o",
        "monovalent_term31_control.o",
    ]
    with (directory / "term31_blas_compiler.log").open("w") as log:
        for command in (compile_command, link_command):
            subprocess.run(
                command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    manifest["private_monovalent_term31_blas"] = dict(
        original_sha256=digest(retained),
        derived_sha256=digest(source),
        original_component_sha256=digest(directory / "monovalent_original_diagram.f"),
        component_control_sha256=digest(directory / "monovalent_term31_control.f90"),
        helper_sha256=digest(__file__),
        extra_payload_bytes=extra,
        specialization="complete original P/Q trial density and angular/Coulomb rows; full terminal W^T U FP64 GEMM",
        scope="private arithmetic control; no public adoption, GPU, full-space root or controlled performance claim",
    )
    manifest["compile_commands"].extend((compile_command, link_command))
    manifest["command"] = link_command
    manifest["files"] = {
        p.name: digest(p) for p in directory.iterdir() if p.is_file() and p != path
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
