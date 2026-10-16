"""Build source-bound complete native core/valence and reduced Sigma2 diagrams.

Original GPL-3.0 routines are retained with namespace/layout reconciliation.
This is fixed-energy linearized SD screening; no CI symmetry approximation,
energy extrapolation, Breit/QED or public physical eigenproblem is added.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from build_pci_valence_backend import build_valence, digest

SIGMA2_SHA = "828ea5d34c70e63bb21ceca5f6e6652da7dc3007ec9ba1021d0d6835c9726105"
ROUTINES = (
    "xcnvw",
    "xfor4",
    "ini",
    "term1",
    "term51",
    "term52",
    "term41a",
    "term41c",
    "term41d",
    "term42a",
    "term42c",
    "term42d",
    "s1",
)


def build_sigma2(source, directory, dimensions, compiler, *, sizes, **core_options):
    source, directory = Path(source), Path(directory)
    upstream = source / "lib/all-order/sdvw-ci.f"
    if digest(upstream) != SIGMA2_SHA:
        raise ValueError("screening diagram source differs from pinned pCI")
    required = {"NZ", "NVWH", "NVH2", "NXV2", "NXCN", "NXV", "NVH"}
    if set(sizes) != required or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in sizes.values()
    ):
        raise ValueError(
            "complete positive screening representation dimensions required"
        )
    manifest = build_valence(
        source,
        directory,
        dimensions,
        compiler,
        nxv=sizes["NXV"],
        nvh=sizes["NVH"],
        **core_options,
    )
    shutil.copy2(upstream, directory / "sdvw-ci.original.f")
    original = upstream.read_text()
    starts = list(
        re.finditer(r"^[ \t]+(?:subroutine|function) (\w+)\b", original, re.M | re.I)
    )
    blocks = {}
    for i, match in enumerate(starts):
        end = (
            starts[i + 1].start()
            if i + 1 < len(starts)
            else original.index('      include "d6j.f"')
        )
        blocks[match.group(1).lower()] = original[match.start() : end]
    body = "\n".join(blocks[name] for name in ROUTINES)
    for name in sorted(ROUTINES, key=len, reverse=True):
        body = re.sub(r"\b" + name + r"\b", "pci_vw_" + name, body, flags=re.I)
    for name in ("val", "xcore1", "xcore2", "irhov"):
        body = re.sub("/" + name + "/", "/vw_" + name + "/", body, flags=re.I)
    body, lookup_patches = re.subn(r"(do 661 i5=0,)10\b", r"\1kk", body, flags=re.I)
    body, common_patches = re.subn(r"\brmi\(nx,nn\)", "rmi(nxx,nn)", body, flags=re.I)
    if lookup_patches != 2 or common_patches != 3:
        raise AssertionError("upstream screening initializer/COMMON identity differs")
    (directory / "sigma2_diagrams.f").write_text(body)
    names = ("NZ", "NVWH", "NVH2", "NXV2", "NXCN")
    (directory / "allvw.par").write_text(
        '      include "all.par"\n'
        + "\n".join(f"      PARAMETER ({name}={sizes[name]})" for name in names)
        + "\n"
    )
    (directory / "vw_control_sizes.inc").write_text(
        "\n".join(f"  integer, parameter :: {name}={sizes[name]}" for name in names)
        + "\n"
    )
    template = Path(__file__).resolve().parents[1] / "src/atomx/methods/_pci_sigma2.f90"
    shutil.copy2(template, directory / template.name)
    command = manifest["command"][:]
    flags = command[: command.index("-shared")]
    compile_command = flags + ["-c", template.name, "sigma2_diagrams.f"]
    command[command.index("-o") : command.index("-o")] = [
        template.stem + ".o",
        "sigma2_diagrams.o",
    ]
    with (directory / "sigma2_compiler.log").open("w") as log:
        for invocation in (compile_command, command):
            result = subprocess.run(
                invocation,
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            result.check_returncode()
    manifest["compile_commands"].extend([compile_command, command])
    manifest.update(
        command=command,
        sigma2_sd=dict(
            source_sha256=SIGMA2_SHA,
            template_sha256=digest(template),
            sizes=sizes,
            core_common_patches=common_patches,
            lookup_initializer_patches=lookup_patches,
            local_storage="serial static original arrays and joint-response temporaries; shared owner lock",
            equation="original complete noniterative Coulomb screening plus irreducible full scalar Fov contraction of valence doubles",
            derivative="common shift of all valence energies; fixed core numerators; supplied full implicit responses; explicit denominator slope2",
            normalization="original parity-free S1; multiply (-1)^k S1(m,v)S1(n,w) to recover direct rho_k",
            symmetry="directed values retained; no approximate pCI Ksym swaps, Hermitian averaging or energy extrapolation",
            scope="real native DC fixed-energy linearized SD; no physical eigenenergy/CI metric/Breit/QED/complex coverage",
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
    parser.add_argument("--sizes-json", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
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
    manifest = build_sigma2(
        args.source,
        args.output,
        json.loads(args.dimensions_json.read_text()),
        args.compiler,
        sizes=json.loads(args.sizes_json.read_text()),
        runtime_directory=args.runtime_directory,
        blas_library=args.blas_library,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
        core_single_to_doubles_blas=args.core_single_to_doubles_blas,
        valence_term2_blas=args.valence_term2_blas,
        valence_term2_threads=args.valence_term2_threads,
    )
    print(
        json.dumps(
            dict(
                directory=str(args.output),
                pin=manifest["pin"],
                library_sha256=manifest["files"]["native_core_sd.so"],
            )
        )
    )


if __name__ == "__main__":
    main()
