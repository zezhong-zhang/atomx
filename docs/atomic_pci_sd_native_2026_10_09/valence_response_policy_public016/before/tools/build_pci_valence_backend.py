"""Build full native core/fixed-energy valence SD with pinned pCI diagrams.

Original GPL-3.0 valence equations are retained. Namespacing and the explicit
core COMMON shape reconciliation change representation, not physical diagrams.
All valence/core pools must match the caller's complete resource preflight.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from build_pci_sd_backend import build, digest

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


def build_valence(
    source,
    directory,
    dimensions,
    compiler,
    *,
    nxv,
    nvh,
    valence_term2_blas=False,
    valence_term2_threads=1,
    **core_options,
):
    source, directory = Path(source), Path(directory)
    upstream = source / "lib/all-order/valsd-ci.f"
    if digest(upstream) != VALENCE_SHA:
        raise ValueError("valence diagram source differs from pinned pCI")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (nxv, nvh)):
        raise ValueError("positive complete valence channel dimensions required")
    if not isinstance(valence_term2_blas, bool):
        raise ValueError("explicit boolean valence term2 BLAS policy required")
    if (
        isinstance(valence_term2_threads, bool)
        or not isinstance(valence_term2_threads, int)
        or valence_term2_threads < 1
        or not valence_term2_blas
        and valence_term2_threads != 1
    ):
        raise ValueError(
            "positive valence term2 worker count; original path requires one"
        )
    if valence_term2_blas and (
        core_options.get("blas_library") is None
        or core_options.get("term21_threads", 0) < 1
    ):
        raise ValueError("valence term2 BLAS requires a bound core BLAS/runtime path")
    template = (
        Path(__file__).resolve().parents[1] / "src/atomx/methods/_pci_valence.f90"
    )
    manifest = build(source, directory, dimensions, compiler, **core_options)
    shutil.copy2(upstream, directory / "valsd-ci.original.f")
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
    for name in sorted(ROUTINES, key=len, reverse=True):
        body = re.sub(r"\b" + name + r"\b", "pci_val_" + name, body, flags=re.I)
    # The core bridge owns rmi(NXX,NN), whereas the original valence program
    # declares rmi(NX,NN). Reconcile the COMMON ABI, retaining all NXX active
    # radial rows. NX may be larger solely to store the full partial-wave list.
    body, common_patches = re.subn(r"\brmi\(nx,nn\)", "rmi(nxx,nn)", body, flags=re.I)
    if common_patches != 8:
        raise AssertionError(f"unexpected core COMMON reconciliation: {common_patches}")
    (directory / "valence_diagrams.f").write_text(body)
    parameters = (directory / "all.par").read_text()
    for name, count in (("NXV", nxv), ("NVH", nvh)):
        parameters = parameters.replace(
            f"PARAMETER ({name}=1)", f"PARAMETER ({name}={count})"
        )
    (directory / "all.par").write_text(parameters)
    (directory / "valence_control_sizes.inc").write_text(
        f"  integer, parameter :: nxv={nxv}, nvh={nvh}\n"
    )
    shutil.copy2(template, directory / template.name)
    term2 = dict(mode="original", threads=1, scratch_bytes=0)
    term2_object = []
    if valence_term2_blas:
        kernel = template.with_name("_pci_valence_term2_streamed.f90")
        (directory / kernel.name).write_text(
            kernel.read_text().replace("CASE_THREADS", str(valence_term2_threads))
        )
        wrapper = (directory / template.name).read_text()
        wrapper, changed = re.subn(
            r"\bcall pci_val_term2\(energy\)",
            "call valence_term2_streamed(energy)",
            wrapper,
        )
        if changed != 3:
            raise AssertionError("complete affine/linear/term2 call coverage differs")
        (directory / template.name).write_text(wrapper)
        d = dimensions
        scratch = 8 * (
            d["NHF"] * (d["NX"] * d["NK"] + d["NXX"] ** 2)
            + valence_term2_threads
            * (d["NHF"] * (4 * d["NXX"] + 2) + 4 * d["NXX"] ** 2)
        )
        term2 = dict(
            mode="streamed_blas",
            threads=valence_term2_threads,
            template_sha256=digest(kernel),
            scratch_bytes=scratch,
            replaced_calls=changed,
            scope="exact shared complete virtual-pair potential, disjoint core-channel heap workers, inner-one BLAS projections and full analytic response; no screening",
        )
        term2_object = [kernel.stem + ".o"]
    command = manifest["command"][:]
    index = command.index("-o")
    # OpenMP implies recursive local storage. The original large valence
    # arrays must stay static, as for the original core diagrams; only the
    # explicitly heap-backed core workers are compiled with OpenMP.
    flags = command[: command.index("-shared")]
    compile_command = flags + ["-c", template.name, "valence_diagrams.f"]
    invocations = [compile_command]
    if valence_term2_blas:
        invocations.append(flags + ["-fopenmp", "-c", kernel.name])
    command[index:index] = [template.stem + ".o", "valence_diagrams.o"] + term2_object
    invocations.append(command)
    with (directory / "valence_compiler.log").open("w") as log:
        for invocation in invocations:
            run = subprocess.run(
                invocation,
                cwd=directory,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            run.check_returncode()
    manifest["compile_commands"].extend(invocations)
    manifest.update(
        command=command,
        valence_sd=dict(
            source_sha256=VALENCE_SHA,
            template_sha256=digest(template),
            nxv=nxv,
            nvh=nvh,
            core_common_patches=common_patches,
            dimension_binding="owned complete NXV/NVH include; independent of precompiled core placeholders",
            local_storage="serial static original arrays and wrapper; external per-backend CoreSD lock; optional heap-only term2 unit compiled with OpenMP",
            term2=term2,
            approximation="CI+all-order fixed-energy numerator equations; original main-loop diagrams without term31; no valence-singles feedback or energy self-consistency",
            fock="full scalar core-core, core-virtual, virtual-virtual commutator; bare F_mv remains H1",
            linear_action="source separated from exact original homogeneous diagrams",
            scope="real native DC, complete declared closed core and virtual pool; no Sigma2/Breit/QED/complex coverage",
        ),
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
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
    parser.add_argument("--blas-library", type=Path)
    parser.add_argument("--term21-threads", type=int, default=0)
    parser.add_argument(
        "--term21-mode", choices=("compact", "streamed"), default="compact"
    )
    parser.add_argument("--core-singles-blas", action="store_true")
    parser.add_argument("--core-single-to-doubles-blas", action="store_true")
    parser.add_argument("--valence-term2-blas", action="store_true")
    parser.add_argument("--valence-term2-threads", type=int, default=1)
    args = parser.parse_args()
    dimensions = json.loads(args.dimensions_json.read_text())
    build_valence(
        args.source,
        args.output,
        dimensions,
        args.compiler,
        nxv=args.nxv,
        nvh=args.nvh,
        runtime_directory=args.runtime_directory,
        blas_library=args.blas_library,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
        core_single_to_doubles_blas=args.core_single_to_doubles_blas,
        valence_term2_blas=args.valence_term2_blas,
        valence_term2_threads=args.valence_term2_threads,
    )


if __name__ == "__main__":
    main()
