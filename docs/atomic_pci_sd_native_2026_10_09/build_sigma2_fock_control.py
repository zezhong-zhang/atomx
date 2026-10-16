"""Fresh source-bound complete finite noncanonical screening diagnostic."""

import json
import shutil
import subprocess
from pathlib import Path

from build_sigma2_control import build_sigma2, digest


def build_sigma2_fock(
    source, directory, dimensions, compiler, *, sizes, runtime, **valence_options
):
    directory = Path(directory)
    manifest = build_sigma2(
        source,
        directory,
        dimensions,
        compiler,
        sizes=sizes,
        runtime=runtime,
        **valence_options,
    )
    template = Path(__file__).with_name("sigma2_fock_control.f90")
    shutil.copy2(template, directory / template.name)
    command = manifest["command"][:]
    command.insert(command.index("-o"), template.name)
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    (directory / "sigma2_fock_compiler.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    manifest["command"] = command
    manifest["sigma2_fock_control"] = dict(
        template_sha256=digest(template),
        equation="irreducible core-virtual Fock contraction of complete valence doubles",
        derivative="joint energy shift, denominator slope2 and full supplied response",
        status="isolated; independent signed and fermion gates required",
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
