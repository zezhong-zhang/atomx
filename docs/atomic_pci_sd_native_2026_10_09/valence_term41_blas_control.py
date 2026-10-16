"""Private original term41 contraction specialization; every row retained.

Only the final nr/n/m sum becomes two complete FP64 GEMMs. All original
direct/exchange angular loops, core numerators and analytic denominators stay.
This serial-static unit retains the backend's actual inner-one BLAS binding.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specialize(original):
    head, term = original.split("      subroutine pci_val_term41(evt)", 1)
    body, tail = term.split("      subroutine pci_val_", 1)
    begin = body.index("                    do 127 nr=nnr,nmaxx(indr)")
    end = body.index("127                 continue", begin) + len(
        "127                 continue"
    )
    retained = body[begin:end]
    if (
        retained.count("x(nr,n)*y(m,nr)") != 1
        or retained.count("x(nr,n)*dy(m,nr)") != 1
        or retained.count("do ") != 3
        or retained.count("xvo(m,n,index)") != 4
        or retained.count("dxvo(m,n,index)") != 2
    ):
        raise ValueError("exact original complete term41 terminal sums required")
    replacement = """                    cm=nmaxx(indm)-nnm+1
                    cn=nmaxx(indn)-nnn+1
                    cr=nmaxx(indr)-nnr+1
                    if(cm.gt.0.and.cn.gt.0.and.cr.gt.0) then
                     call dgemm('N','N',cm,cn,cr,1.d0,
     *                    y(nnm,nnr),nx,x(nnr,nnn),nx,1.d0,
     *                    xvo(nnm,nnn,index),nxx)
                     call dgemm('N','N',cm,cn,cr,1.d0,
     *                    dy(nnm,nnr),nx,x(nnr,nnn),nx,1.d0,
     *                    dxvo(nnm,nnn,index),nxx)
                    endif"""
    body = body.replace(
        '      include "all.par"', '      include "all.par"\n      integer cm,cn,cr', 1
    )
    body = body.replace(retained, replacement, 1)
    return (
        head
        + "      subroutine pci_val_term41(evt)"
        + body
        + "      subroutine pci_val_"
        + tail
    )


def install(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "private_term41_blas" in manifest:
        raise ValueError("one explicit term41 specialization per fresh backend")
    source = directory / "valence_diagrams.f"
    original = source.read_text()
    retained = directory / "pre_term41_blas_valence_diagrams.f.txt"
    retained.write_text(original)
    derived = specialize(original)
    source.write_text(derived)
    commands = [c for c in manifest["compile_commands"] if "valence_diagrams.f" in c]
    if len(commands) != 1 or "-fopenmp" in commands[0]:
        raise ValueError("original serial-static valence compilation required")
    compiler = commands[0][: commands[0].index("-c")]
    compile_command = compiler + ["-c", "valence_diagrams.f"]
    link_command = manifest["command"][:]
    with (directory / "term41_blas_compiler.log").open("w") as log:
        for command in (compile_command, link_command):
            subprocess.run(
                command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    manifest["private_term41_blas"] = dict(
        original_sha256=digest(retained),
        derived_sha256=digest(source),
        helper_sha256=digest(__file__),
        specialization="exact full FP64 y@x/dy@x terminal sums; every original core/direct/exchange/phase/denominator row retained",
        extra_payload_bytes=0,
        scope="private arithmetic specialization; requires full signed/root/response comparison, no controlled speedup or public adoption",
    )
    manifest["compile_commands"].extend((compile_command, link_command))
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
