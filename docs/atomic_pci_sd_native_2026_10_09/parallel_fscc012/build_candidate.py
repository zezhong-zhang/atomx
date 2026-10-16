"""Compile one isolated FSCC equation candidate; never mutate validated inputs."""

import argparse
import difflib
import hashlib
import json
import os
import resource
import shlex
import shutil
import subprocess
import time
from pathlib import Path

OWNER = Path(__file__).resolve().parent.parent
SOURCE = OWNER / "expt_compact_public_source001"
BUILD = OWNER / "expt_compact_public_source001-build"
EXPECTED = "1605f4d9daf0adb5ecc543fc785da4cd5e32d96121260dc00e9951bdbdd0d33f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = SOURCE / "src/rcc/models/sector01.c"
    link_file = BUILD / "CMakeFiles/expt.x.dir/link.txt"
    flags_file = BUILD / "CMakeFiles/expt.x.dir/flags.make"
    link = shlex.split(link_file.read_text())
    files = [
        Path(__file__),
        original,
        flags_file,
        link_file,
        BUILD / "expt.x",
        BUILD / "atomx_backend.json",
        Path(link[0]),
    ]
    files += [BUILD / token for token in link if token.endswith((".o", ".a"))]
    files += [p for p in (SOURCE / "src/rcc").rglob("*") if p.is_file()]
    binding = {str(path): sha(path) for path in files}
    if sha(BUILD / "expt.x") != EXPECTED:
        raise ValueError("original validated backend changed")
    contract = dict(
        hypothesis="The dgd2b_2 open Fov*T2 term is already antisymmetric in outgoing particles; perm(34) duplicates it. Removing only that extra permutation must close the unchanged-H finite Bloch equation.",
        evidence="audit003 complete signed residual equals positive open Fov*T2 to 1.776e-15 Ha; no fitted coefficients or changed Hamiltonian.",
        caps=dict(threads=1, address_bytes=4 * 1024**3, compile_seconds=120),
        gates=dict(oracle_Ha=1e-11, strict_backend_Ha=1e-10),
        load=os.getloadavg(),
        available_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        source_and_objects=binding,
        stop="One isolated candidate build. Original backend/source/objects must remain unchanged; no deployment.",
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2))
    shutil.copytree(SOURCE / "src/rcc", out / "source/src/rcc")
    shutil.copy2(SOURCE / "LICENSE", out / "source/LICENSE")
    candidate = out / "source/src/rcc/models/sector01.c"
    old = original.read_text()
    before = """    // dgd2b_2
    reorder("t2c", "r1", "2341");
    mult("vh", "r1", "r2", 1);
    perm("r2", "(34)");
    update("s2_0", -1.0, "r2");"""
    after = """    // dgd2b_2: open core-valence Fock contraction.
    // ATOMX_FOV_T2_001: t2c already antisymmetric in outgoing particles.
    // A second P(34) would double this contribution in noncanonical orbitals.
    reorder("t2c", "r1", "2341");
    mult("vh", "r1", "r2", 1);
    update("s2_0", -1.0, "r2");"""
    if old.count(before) != 1:
        raise ValueError("candidate patch source predicate changed")
    new = old.replace(before, after)
    candidate.write_text(new)
    patch = "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="a/src/rcc/models/sector01.c",
            tofile="b/src/rcc/models/sector01.c",
        )
    )
    (out / "equation.patch").write_text(patch)
    flags = {}
    for line in flags_file.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    includes = [
        token.replace(
            str(SOURCE / "src/rcc/include"), str(out / "source/src/rcc/include")
        )
        for token in flags["C_INCLUDES"]
    ]
    compile_cmd = [
        "/usr/bin/cc",
        *flags["C_DEFINES"],
        *includes,
        *flags["C_FLAGS"],
        "-o",
        str(out / "sector01.c.o"),
        "-c",
        str(candidate),
    ]
    link_cmd = [
        token if not token.endswith((".o", ".a")) else str(BUILD / token)
        for token in link
    ]
    old_object = str(BUILD / "CMakeFiles/expt.x.dir/src/rcc/models/sector01.c.o")
    link_cmd[link_cmd.index(old_object)] = str(out / "sector01.c.o")
    link_cmd[link_cmd.index("-o") + 1] = str(out / "expt.x")
    (out / "commands.json").write_text(
        json.dumps(dict(compile=compile_cmd, link=link_cmd), indent=2)
    )
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    start = time.perf_counter()
    with (out / "build.log").open("w") as log:
        for cmd in (compile_cmd, link_cmd):
            subprocess.run(
                cmd,
                cwd=BUILD,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=True,
            )
    if not all(sha(Path(path)) == digest for path, digest in binding.items()):
        raise ValueError("validated build/source inputs changed during candidate build")
    metadata = json.loads((BUILD / "atomx_backend.json").read_text())
    metadata.update(
        executable=str(out / "expt.x"),
        executable_sha256=sha(out / "expt.x"),
        original_executable_sha256=EXPECTED,
        parent_patch_sha256=metadata["patch_sha256"],
        patch_sha256=sha(out / "equation.patch"),
        experimental_equation_patch="ATOMX_FOV_T2_001",
        relinked_object_sha256=sha(out / "sector01.c.o"),
        build_contract_sha256=sha(out / "contract.json"),
    )
    metadata["source_sha256"]["src/rcc/models/sector01.c"] = sha(candidate)
    (out / "atomx_backend.json").write_text(json.dumps(metadata, indent=2))
    result = dict(
        status="candidate_built_unverified",
        highest_level_reached=False,
        candidate_sha256=sha(out / "expt.x"),
        candidate_manifest_sha256=sha(out / "atomx_backend.json"),
        original_backend_unchanged=True,
        original_source_and_objects_unchanged=True,
        patch_sha256=sha(out / "equation.patch"),
        elapsed_compile_and_link_seconds=time.perf_counter() - start,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
