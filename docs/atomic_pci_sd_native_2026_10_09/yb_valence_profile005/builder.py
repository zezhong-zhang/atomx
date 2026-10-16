"""Instrument a private complete backend without changing its arithmetic."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

OWNER = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent = OWNER / "yb_full_static003/backend"
    out = OWNER / "yb_valence_profile005"
    out.mkdir(exist_ok=False)
    manifest = json.loads((parent / "manifest.json").read_text())
    contract = dict(
        hypothesis="Original full valence source diagrams dominate its complete map; measure each unchanged diagram before selecting an optimization",
        fixed="original Yb68core/414virtual/ns123/Fermi/l<=3/5Ha and run008 signed core amplitudes, same complete native original valence diagrams, same real DC/Fock/operator",
        changes="private native stage clocks only; no skipped terms, amplitude screening, altered loops, threads or arithmetic",
        comparison="complete signed singles/doubles and both derivatives against retained uninstrumented stack_new004 map",
        gate_Ha=1e-11,
        stopping="any complete signed mismatch or nonfinite result rejects; one profiling map then decide from observed stages",
        cap_bytes=32 * 1024**3,
        load=os.getloadavg(),
        performance_scope="diagnostic single-map stage timing; no controlled speedup assertion",
        parent_manifest_sha256=digest(parent / "manifest.json"),
        parent_manifest=manifest,
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    shutil.copy2(__file__, out / "builder.py")
    backend = out / "backend"
    shutil.copytree(parent, backend)
    wrapper = backend / "_pci_valence.f90"
    body = wrapper.read_text()
    body = body.replace(
        "  use iso_c_binding\n",
        "  use iso_c_binding\n  use iso_fortran_env, only: error_unit\n",
        1,
    )
    start = body.index("  subroutine atomx_val_map(")
    stop = body.index("  end subroutine", start)
    section = body[start:stop]
    declaration = "    integer(c_int64_t) clock_before,clock_after,clock_rate\n"
    section = section.replace(
        "    rvo=s;",
        declaration + "    call system_clock(clock_before,clock_rate)\n    rvo=s;",
        1,
    )
    lines = []
    for line in section.splitlines():
        lines.append(line)
        if line.strip().startswith("call pci_val_") or line.strip().startswith(
            "rs=rvo;"
        ):
            name = line.strip().split("(")[0].removeprefix("call ")
            if name.startswith("rs="):
                name = "copy_source"
            lines.extend(
                [
                    "    call system_clock(clock_after)",
                    f"    write(error_unit,*) 'ATOMX_PROFILE {name}',real(clock_after-clock_before,c_double)/real(clock_rate,c_double)",
                    "    flush(error_unit)",
                    "    clock_before=clock_after",
                ]
            )
    body = body[:start] + "\n".join(lines) + "\n" + body[stop:]
    start = body.index("  subroutine atomx_val_map_full_fock(")
    stop = body.index("  end subroutine", start)
    section = body[start:stop]
    section = section.replace(
        "    call atomx_val_map", declaration + "    call atomx_val_map", 1
    )
    section = section.replace(
        "    call valence_fock_map",
        "    call system_clock(clock_before,clock_rate)\n    call valence_fock_map",
        1,
    )
    section += "    call system_clock(clock_after)\n    write(error_unit,*) 'ATOMX_PROFILE full_fock_and_copy',real(clock_after-clock_before,c_double)/real(clock_rate,c_double)\n    flush(error_unit)\n"
    body = body[:start] + section + body[stop:]
    wrapper.write_text(body)
    command = manifest["command"]
    compile_command = command[: command.index("-shared")] + ["-c", wrapper.name]
    with (backend / "profile_compiler.log").open("w") as log:
        for invocation in (compile_command, command):
            subprocess.run(
                invocation,
                cwd=backend,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    manifest["compile_commands"].extend([compile_command, command])
    manifest["valence_sd"]["template_sha256"] = digest(wrapper)
    manifest["diagnostic_only"] = contract
    manifest["files"] = {
        p.name: digest(p)
        for p in backend.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (backend / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                backend=str(backend),
                executable_sha256=digest(backend / "native_core_sd.so"),
            )
        )
    )


if __name__ == "__main__":
    main()
