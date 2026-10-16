"""Isolated fixed-energy pCI valence numerator/derivative convention control.

This extends a fresh original core backend. No public valence method is implied;
noncanonical valence Fock terms and Sigma2 require separate independent gates.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_pci_sd_backend import build, digest  # noqa: E402

VALENCE_SHA = "6bf791841f1d9f6c0e0c2fd176d4517007ac622d2aea69fd1278107a414ab07b"
ROUTINES = (
    "xval",
    "rhoval",
    "outin",
    "term1",
    "term2",
    "term32",
    "term41",
    "term42",
    "term51",
    "term52",
    "terma1",
    "terma2",
    "terma3",
    "terma4",
)


def build_control(source, directory, dimensions, compiler, *, nxv, nvh, runtime):
    """Keep every original valence diagram, namespaced in the same case image."""
    source, directory = Path(source), Path(directory)
    upstream = source / "lib/all-order/valsd-ci.f"
    if digest(upstream) != VALENCE_SHA:
        raise ValueError("valence source differs from pinned pCI")
    if dimensions["NX"] != dimensions["NXX"]:
        raise ValueError("prototype COMMON requires NX=NXX; reject padding mismatch")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (nxv, nvh)):
        raise ValueError("complete valence channel dimensions required")
    manifest = build(source, directory, dimensions, compiler, runtime_directory=runtime)
    shutil.copyfile(upstream, directory / "valsd-ci.original.f")
    original = upstream.read_text()
    starts = list(re.finditer(r"^      subroutine (\w+)\b", original, re.M | re.I))
    blocks = {}
    for i, match in enumerate(starts):
        end = (
            starts[i + 1].start()
            if i + 1 < len(starts)
            else original.index('      include "inidat.f"')
        )
        blocks[match.group(1).lower()] = original[match.start() : end]
    body = "\n".join(blocks[name] for name in ROUTINES)
    # Symbol renaming only; all source equations and COMMON layouts are retained.
    for name in sorted(ROUTINES, key=len, reverse=True):
        body = re.sub(r"\b" + name + r"\b", "pci_val_" + name, body, flags=re.I)
    (directory / "valence_diagrams.f").write_text(body)
    parameters = (directory / "all.par").read_text()
    for name, size in (("NXV", nxv), ("NVH", nvh)):
        parameters = parameters.replace(
            f"PARAMETER ({name}=1)", f"PARAMETER ({name}={size})"
        )
    (directory / "all.par").write_text(parameters)
    template = Path(__file__).with_name("valence_control.f90")
    shutil.copyfile(template, directory / "valence_control.f90")
    old = manifest["command"]
    insert = old.index("-o")
    command = (
        old[:insert] + ["valence_control.f90", "valence_diagrams.f"] + old[insert:]
    )
    run = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    (directory / "valence_compiler.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    manifest.update(
        command=command,
        valence_control=dict(
            source_sha256=VALENCE_SHA,
            nxv=nxv,
            nvh=nvh,
            template_sha256=digest(template),
            approximation="original CI+all-order fixed-energy numerator map; no term31 or energy feedback",
            status="independent convention control; not a public native valence API",
        ),
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
