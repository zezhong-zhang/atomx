"""Build separate native standalone SD on the pinned core/valence libraries.

The fixed-energy CI+all-order wrapper remains unchanged. All standalone
diagrams are evaluated at bare ev; feedback is a separate consumer equation.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from build_pci_valence_backend import VALENCE_SHA, build_valence, digest


def build_monovalent(source, directory, dimensions, compiler, *, nxv, nvh, **options):
    source, directory = Path(source), Path(directory)
    original = source / "lib/all-order/valsd-ci.f"
    if digest(original) != VALENCE_SHA:
        raise ValueError("standalone valence source differs from pinned pCI")
    template = (
        Path(__file__).resolve().parents[1] / "src/atomx/methods/_pci_monovalent.f90"
    )
    manifest = build_valence(
        source, directory, dimensions, compiler, nxv=nxv, nvh=nvh, **options
    )
    text = original.read_text()
    match = re.search(
        r"^      subroutine term31\b.*?(?=^      subroutine term32\b)",
        text,
        re.M | re.S | re.I,
    )
    if match is None:
        raise ValueError("complete standalone singles diagram not found")
    body, count = re.subn(r"\bterm31\b", "pci_val_term31", match[0], flags=re.I)
    if count != 1:
        raise ValueError("standalone singles diagram namespacing differs")
    (directory / "monovalent_diagram.f").write_text(body)
    shutil.copy2(template, directory / template.name)
    command = manifest["command"][:]
    flags = command[: command.index("-shared")]
    compile_command = flags + ["-c", template.name, "monovalent_diagram.f"]
    index = command.index("-o")
    command[index:index] = [template.stem + ".o", "monovalent_diagram.o"]
    with (directory / "monovalent_compiler.log").open("w") as log:
        for invocation in (compile_command, command):
            subprocess.run(
                invocation,
                cwd=directory,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    manifest["compile_commands"].extend([compile_command, command])
    manifest["command"] = command
    manifest["monovalent_sd"] = dict(
        source_sha256=VALENCE_SHA,
        template_sha256=digest(template),
        builder_sha256=digest(Path(__file__)),
        diagram_sha256=digest(directory / "monovalent_diagram.f"),
        extra_static_bytes=8
        * (2 * dimensions["NXX"] ** 2 * nvh + 4 * dimensions["NX"]),
        equation="standalone linearized SD, fixed core, physical valence singles and all core-valence doubles",
        core_denominators="unshifted closed-core; all diagrams at bare ev",
        valence_coordinates="physical rho; native double input multiplied by unshifted D0",
        local_storage="serial static wrapper and original term31; same CoreSD lock",
        scope="real native DC; no production physical state/properties, triples or GPU certificate",
    )
    manifest["files"] = {
        path.name: digest(path)
        for path in directory.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions-json", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--nxv", type=int, required=True)
    parser.add_argument("--nvh", type=int, required=True)
    parser.add_argument("--runtime-directory", type=Path)
    args = parser.parse_args()
    build_monovalent(
        args.source,
        args.output,
        json.loads(args.dimensions_json.read_text()),
        args.compiler,
        nxv=args.nxv,
        nvh=args.nvh,
        runtime_directory=args.runtime_directory,
    )


if __name__ == "__main__":
    main()
