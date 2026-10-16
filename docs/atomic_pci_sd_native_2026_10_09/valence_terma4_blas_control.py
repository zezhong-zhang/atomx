"""Private complete valence singles contraction with exact BLAS factorization.

The original complete signed angular/direct/exchange rows and native RINT
projection are unchanged. Full P and Q intermediate contractions and analytic
response use four GEMMs before the original ordered n reduction.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scratch_bytes(dimensions):
    return 32 * dimensions["NHF"] * dimensions["NX"]


def specialize(original):
    head, term = original.split("      subroutine pci_val_terma4(evt)", 1)
    body, separator, tail = term.partition("      subroutine pci_val_")
    if not separator and not body.rstrip().endswith("end"):
        raise ValueError("complete final native singles routine required")
    start = body.index("                 do 109 n=nnn,nmaxx(indn)")
    end = body.index("109              continue", start) + len(
        "109              continue"
    )
    retained = body[start:end]
    if retained.count("do ") != 3 or any(
        retained.count(product) != 1
        for product in (
            "gg(ij,nr,indr)*x(nr,n)*uu(ij,n)",
            "ff(ij,nr,indr)*x(nr,n)*uu(ij,n)",
            "gg(ij,nr,indr)*dx(nr,n)*uu(ij,n)",
            "ff(ij,nr,indr)*dx(nr,n)*uu(ij,n)",
        )
    ):
        raise ValueError("exact original complete P/Q singles contraction required")
    replacement = """                 cn=nmaxx(indn)-nnn+1
                 cr=nmaxx(indr)-nnr+1
                 if(cn.gt.0.and.cr.gt.0) then
                  call dgemm('N','N',max,cn,cr,1.d0,
     *                 gg(1,nnr,indr),nhf,x(nnr,nnn),nx,
     *                 0.d0,xg,nhf)
                  call dgemm('N','N',max,cn,cr,1.d0,
     *                 ff(1,nnr,indr),nhf,x(nnr,nnn),nx,
     *                 0.d0,xf,nhf)
                  call dgemm('N','N',max,cn,cr,1.d0,
     *                 gg(1,nnr,indr),nhf,dx(nnr,nnn),nx,
     *                 0.d0,dxg,nhf)
                  call dgemm('N','N',max,cn,cr,1.d0,
     *                 ff(1,nnr,indr),nhf,dx(nnr,nnn),nx,
     *                 0.d0,dxf,nhf)
                  do n=nnn,nmaxx(indn)
                   do ij=1,max
                    v1(ij)=v1(ij)+xg(ij,n-nnn+1)*uu(ij,n)
                    v2(ij)=v2(ij)+xf(ij,n-nnn+1)*uu(ij,n)
                    dv1(ij)=dv1(ij)+dxg(ij,n-nnn+1)*uu(ij,n)
                    dv2(ij)=dv2(ij)+dxf(ij,n-nnn+1)*uu(ij,n)
                   enddo
                  enddo
                 endif"""
    body = body.replace(
        '      include "all.par"',
        '      include "all.par"\n      integer cn,cr\n      dimension xg(nhf,nx),xf(nhf,nx),dxg(nhf,nx),dxf(nhf,nx)',
        1,
    )
    body = body.replace(retained, replacement, 1)
    return (
        head
        + "      subroutine pci_val_terma4(evt)"
        + body
        + separator
        + tail
    )


def install(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "private_terma4_blas" in manifest:
        raise ValueError("one explicit singles specialization per fresh backend")
    source = directory / "valence_diagrams.f"
    retained = directory / "pre_terma4_blas_valence_diagrams.f.txt"
    original = source.read_text()
    retained.write_text(original)
    source.write_text(specialize(original))
    commands = [c for c in manifest["compile_commands"] if "valence_diagrams.f" in c]
    if not commands or any("-fopenmp" in c for c in commands):
        raise ValueError("original serial-static valence compilation required")
    compiler = commands[0][: commands[0].index("-c")]
    if any(c[: c.index("-c")] != compiler for c in commands):
        raise ValueError("actual valence compiler identity changed")
    compile_command = compiler + ["-c", "valence_diagrams.f"]
    link_command = manifest["command"][:]
    with (directory / "terma4_blas_compiler.log").open("w") as log:
        for command in (compile_command, link_command):
            subprocess.run(
                command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    manifest["private_terma4_blas"] = dict(
        original_sha256=digest(retained),
        derived_sha256=digest(source),
        helper_sha256=digest(__file__),
        extra_payload_bytes=scratch_bytes(manifest["dimensions"]),
        specialization="complete P/Q x/dx GEMMs plus original ordered n reduction/native RINT; every original angular/direct/exchange/denominator row retained",
        scope="private arithmetic specialization; complete raw/signed/response gates required; no controlled timing/public adoption",
    )
    manifest["compile_commands"].extend((compile_command, link_command))
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
