"""Unpromoted exact BLAS term31/32 control, retaining original native routines."""

import json
import shutil
import subprocess
from pathlib import Path

from build_pci_sd_backend import build, digest


def build_control(source, directory, dimensions, compiler, **options):
    threads = options.get("term21_threads", 0)
    if not threads or options.get("blas_library") is None:
        raise ValueError("explicit native LP64 BLAS and positive workers required")
    manifest = build(source, directory, dimensions, compiler, **options)
    directory = Path(directory)
    worker = Path(__file__).with_name("core_single_to_doubles_blas.f90")
    shutil.copy2(worker, directory / "single_to_doubles.original.f90")
    (directory / "single_to_doubles.f90").write_text(
        worker.read_text().replace("CASE_THREADS", str(manifest["term21"]["threads"]))
    )
    bridge = directory / "native_bridge.f90"
    body = bridge.read_text()
    for which in (31, 32):
        old = f"    call term{which}\n"
        if body.count(old) != 2:
            raise AssertionError("original action and profiler coverage differs")
        body = body.replace(old, f"    call core_single_to_doubles_blas({which})\n")
    bridge.write_text(body)
    flags = manifest["compile_commands"][0][:-2]
    commands = [
        flags + ["-c", "native_bridge.f90"],
        flags + ["-fopenmp", "-c", "single_to_doubles.f90"],
    ]
    command = manifest["command"][:]
    command.insert(command.index("-o"), "single_to_doubles.o")
    commands.append(command)
    with (directory / "single_to_doubles_compiler.log").open("w") as log:
        for invocation in commands:
            run = subprocess.run(
                invocation, cwd=directory, text=True, capture_output=True
            )
            log.write(run.stdout + run.stderr)
            log.flush()
            run.check_returncode()
    manifest.update(
        command=command,
        single_to_doubles=dict(
            mode="unpromoted exact native weighted BLAS term31/32",
            workers=manifest["term21"]["threads"],
            template_sha256=digest(worker),
            scope="full scalar core singles into doubles; original angular selection and all radial rows",
            extra_scratch_bound_bytes=8
            * dimensions["NHF"]
            * (
                dimensions["NXX"]
                * (dimensions["NK"] + 1 + manifest["term21"]["threads"])
                + dimensions["NN"]
                + 2 * manifest["term21"]["threads"]
            ),
        ),
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
