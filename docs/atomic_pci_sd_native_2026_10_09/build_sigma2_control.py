"""Source-bound original pCI two-valence screening convention control."""

import json
import re
import shutil
import subprocess
from pathlib import Path

from build_valence_control import build_control, digest

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


def build_sigma2(
    source,
    directory,
    dimensions,
    compiler,
    *,
    sizes,
    runtime,
    valence_builder=build_control,
):
    source, directory = Path(source), Path(directory)
    upstream = source / "lib/all-order/sdvw-ci.f"
    if digest(upstream) != SIGMA2_SHA:
        raise ValueError("two-valence source differs from pinned pCI")
    if set(sizes) != {"NZ", "NVWH", "NVH2", "NXV2", "NXCN", "NXV", "NVH"}:
        raise ValueError("complete two-valence representation dimensions required")
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in sizes.values()
    ):
        raise ValueError("positive complete screening dimensions required")
    manifest = valence_builder(
        source,
        directory,
        dimensions,
        compiler,
        nxv=sizes["NXV"],
        nvh=sizes["NVH"],
        runtime=runtime,
    )
    shutil.copyfile(upstream, directory / "sdvw-ci.original.f")
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
    # Different COMMON layouts cannot be overlaid in the single case image.
    for name in ("val", "xcore1", "xcore2", "irhov"):
        body = re.sub(r"/" + name + r"/", "/vw_" + name + "/", body, flags=re.I)
    # Three upstream lookup initializers assume KK=10. Bind initialization to
    # the declared representation; every equation and channel remains intact.
    body, patches = re.subn(r"(do 661 i5=0,)10\b", r"\1kk", body, flags=re.I)
    if patches != 2:
        raise AssertionError("unexpected upstream lookup initializer coverage")
    # Match the radial shape owned by native_sd_bridge, even when the separate
    # partial-wave-list dimension NX exceeds the actual radial dimension NXX.
    body, core_common_patches = re.subn(
        r"\brmi\(nx,nn\)", "rmi(nxx,nn)", body, flags=re.I
    )
    if core_common_patches != 3:
        raise AssertionError("unexpected screening core COMMON coverage")
    (directory / "sigma2_diagrams.f").write_text(body)
    parameters = (
        '      include "all.par"\n'
        + "\n".join(
            f"      PARAMETER ({name}={sizes[name]})"
            for name in ("NZ", "NVWH", "NVH2", "NXV2", "NXCN")
        )
        + "\n"
    )
    (directory / "allvw.par").write_text(parameters)
    (directory / "vw_control_sizes.inc").write_text(
        "\n".join(
            f"  integer, parameter :: {name}={sizes[name]}"
            for name in ("NZ", "NVWH", "NVH2", "NXV2", "NXCN")
        )
        + "\n"
    )
    template = Path(__file__).with_name("sigma2_control.f90")
    shutil.copyfile(template, directory / template.name)
    old = manifest["command"]
    insert = old.index("-o")
    command = old[:insert] + [template.name, "sigma2_diagrams.f"] + old[insert:]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    (directory / "sigma2_compiler.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    manifest.update(
        command=command,
        sigma2_control=dict(
            source_sha256=SIGMA2_SHA,
            sizes=sizes,
            template_sha256=digest(template),
            namespaces=["val", "xcore1", "xcore2", "irhov"],
            lookup_initializer_patches=patches,
            core_common_patches=core_common_patches,
            equation="original noniterative CI+all-order two-valence diagrams",
            derivative_coverage="upstream derivative recorded but not accepted as joint energy derivative",
            status="isolated finite screening control; operator/response gates required, no public production claim",
        ),
    )
    manifest["files"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
