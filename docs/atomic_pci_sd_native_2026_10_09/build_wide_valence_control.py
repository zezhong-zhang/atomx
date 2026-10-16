"""Private finite native precision experiment; no public precision policy."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

OWNED_COMMON = (
    "val",
    "xval1",
    "xval2",
    "irhov",
    "rhovin",
    "rhovout",
    "drhovin",
    "drhovout",
    "ipv",
)
WIDE_VARIABLES = {
    "xv1",
    "xv2",
    "xvi",
    "rvi",
    "xvo",
    "rvo",
    "dxvi",
    "drvi",
    "dxvo",
    "drvo",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_variables(text):
    """Only explicit COMMON coordinates; all other implicit locals may widen."""
    names = set()
    for match in re.finditer(r"^\s*common\s*/\w+/\s*(.*)$", text, re.M | re.I):
        depth, begin = 0, 0
        declaration = match.group(1)
        for i, char in enumerate(declaration + ","):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                item = declaration[begin:i].strip()
                name = re.match(r"[A-Za-z]\w*", item)
                if name is None:
                    raise AssertionError("unparsed explicit COMMON coordinate")
                names.add(name[0].lower())
                begin = i + 1
        if depth:
            raise AssertionError("multiline COMMON must be explicitly handled")
    return names


def build_wide(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["valence_sd"]["term2"]["mode"] != "original":
        raise ValueError("finite wide control requires original scalar diagrams")
    source = (directory / "valence_diagrams.f").read_text()
    starts = list(re.finditer(r"^      subroutine \w+\b", source, re.M | re.I))
    if len(starts) != 14:
        raise AssertionError("complete pinned valence routine set required")
    bodies = []
    common_records = {}
    for i, start in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(source)
        body = source[start.start() : end]
        names = common_variables(body)
        common_records[body.splitlines()[0].strip()] = sorted(names)
        # Core fields/weighted native coordinates and angular helpers keep
        # their original ABI. Valence-owned arrays and local contractions use
        # REAL*10. Integers keep their original implicit I--N convention.
        narrow = sorted(
            n for n in names if n not in WIDE_VARIABLES and not "i" <= n[0] <= "n"
        )
        declarations = "      double precision s,d6j\n"
        declarations += "".join(f"      double precision {n}\n" for n in narrow)
        body, changed = re.subn(
            r'(      include "all.par"\n)', r"\1" + declarations, body, flags=re.I
        )
        if changed != 1:
            raise AssertionError("one complete parameter include per routine required")
        body, changed = re.subn(
            r"implicit double precision \(a-h,o-z\)",
            "implicit real*10 (a-h,o-z)",
            body,
            flags=re.I,
        )
        if changed != 1:
            raise AssertionError("explicit local precision declaration required")
        bodies.append(body)
    diagrams = "\n".join(bodies)
    wrapper = (directory / "_pci_valence.f90").read_text()
    wrapper = wrapper.replace(
        "native_valence_fock_bridge", "native_valence_wide_fock_bridge"
    )
    wrapper = wrapper.replace("native_valence_bridge", "native_valence_wide_bridge")
    wrapper = wrapper.replace("atomx_val_", "atomx_wide_val_")
    wrapper = wrapper.replace("real(c_double)", "real(c_long_double)")
    for old, new in (
        ("real(c_long_double) gv(nhf),fv(nhf),ev", "real(c_double) gv(nhf),fv(nhf),ev"),
        (
            "real(c_long_double),intent(in) :: t(total)",
            "real(c_double),intent(in) :: t(total)",
        ),
    ):
        if wrapper.count(old) != 1:
            raise AssertionError("explicit retained double core/val ABI required")
        wrapper = wrapper.replace(old, new)
    for name in OWNED_COMMON:
        pattern = r"/" + name + r"/"
        wrapper = re.sub(pattern, "/wide_" + name + "/", wrapper, flags=re.I)
        diagrams = re.sub(pattern, "/wide_" + name + "/", diagrams, flags=re.I)
    wrapper = wrapper.replace("pci_val_", "wide_val_")
    diagrams = diagrams.replace("pci_val_", "wide_val_")
    diagrams = re.sub(r"\byfun\b", "wide_yfun", diagrams, flags=re.I)
    diagrams = re.sub(r"\brint\b", "wide_rint", diagrams, flags=re.I)

    native = (directory / "native_bridge.f90").read_text()
    match = re.search(
        r"  recursive subroutine density_action\(x,y,k\).*?  end subroutine",
        native,
        re.S,
    )
    if match is None:
        raise AssertionError("actual bound native weighted density action required")
    density = match[0].replace("density_action", "wide_density_action")
    density = density.replace("real(c_double)", "real(c_long_double)")
    # Edge weights remain actual double inputs, promoted before multiplication.
    density = density.replace(
        "w=edge_weight(e,k)", "w=real(edge_weight(e,k),c_long_double)"
    )
    helpers = (
        """module native_wide_density
  use iso_c_binding
  use native_sd_bridge, only: nhf,nv,ng,ne,outer,rising,falling,edge_left,edge_right,edge_weight
  implicit none
contains
"""
        + density
        + """
end module

subroutine wide_yfun(x,y,k,m,*)
  use iso_c_binding
  use native_sd_bridge, only: nhf,kk
  use native_wide_density, only: wide_density_action
  implicit none
  integer k,m
  real(c_long_double) x(nhf),y(nhf)
  if(k<0.or.k>kk.or.m/=nhf) error stop 'invalid complete wide native density action'
  call wide_density_action(x,y,k)
end subroutine

function wide_rint(f,na,nb,nq,h) result(value)
  use iso_c_binding
  implicit none
  integer na,nb,nq
  real(c_long_double) f(*),value
  real(c_double) h
  value=sum(f(na:nb))
end function

subroutine atomx_wide_val_precision(bits,significant_digits) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int),intent(out) :: bits,significant_digits
  real(c_long_double) value
  bits=storage_size(value); significant_digits=digits(value)
end subroutine
"""
    )
    for name, body in (
        ("wide_valence.f90", wrapper),
        ("wide_valence_diagrams.f", diagrams),
        ("wide_density.f90", helpers),
    ):
        (directory / name).write_text(body)
    command = manifest["command"][:]
    flags = command[: command.index("-shared")]
    # Keep original large local arrays static; no OpenMP promotion here.
    compile_command = flags + [
        "-c",
        "wide_valence.f90",
        "wide_valence_diagrams.f",
        "wide_density.f90",
    ]
    command[command.index("-o") : command.index("-o")] = [
        "wide_valence.o",
        "wide_valence_diagrams.o",
        "wide_density.o",
    ]
    with (directory / "wide_compiler.log").open("w") as log:
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
    manifest["private_wide_valence"] = dict(
        policy="original complete valence local/owned/density contractions REAL*10; core fields and angular helpers keep double ABI; all valence COMMON names separate",
        original_common_variables=common_records,
        scope="finite native precision diagnosis only, public ABI and default unchanged",
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
