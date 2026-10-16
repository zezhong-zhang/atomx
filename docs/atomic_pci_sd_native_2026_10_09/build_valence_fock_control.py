"""Source-bound isolated valence extension, without changing original diagrams."""

import json
import shutil
import subprocess
from pathlib import Path

from build_valence_control import build_control, digest


def build_fock_control(source, directory, dimensions, compiler, *, nxv, nvh, runtime):
    directory = Path(directory)
    manifest = build_control(
        source, directory, dimensions, compiler, nxv=nxv, nvh=nvh, runtime=runtime
    )
    template = Path(__file__).with_name("valence_fock_control.f90")
    shutil.copy2(template, directory / template.name)
    command = manifest["command"][:]
    command.insert(command.index("-o"), template.name)
    run = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    (directory / "valence_fock_compiler.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    manifest["command"] = command
    manifest["valence_fock_control"] = dict(
        template_sha256=digest(template),
        equation="complete scalar one-body commutator with connected core attachment difference",
        bare_one_body="F_mv excluded from correlation Sigma1; retained in H1",
        status="unverified isolated extension; independent full signed and fermion gates required",
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
