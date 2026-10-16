"""Source-bound private compact-index lookup and full native CCSDT replay.

Compile one changed engine object and link the unchanged pinned backend
objects. Preserve their hashes and the full LGPL source/patch. No reference
checkout or existing backend is mutated. This is a storage correctness gate,
without production triples, enumeration scalability or timing claims.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
from expt_compact_index_patch import patch_diagram

from atomx.methods.fscc import _axial_irrep, _model_matrix, read_heff

OWNER = Path(__file__).resolve().parent
HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include "diagram.h"
size_t n_spinor_blocks;
static size_t allocated;
void *cc_malloc(size_t n) { allocated += n; void *p=malloc(n); if(!p) exit(90); return p; }
void cc_free(void *p) { free(p); }
void errquit(const char *fmt, ...) { fprintf(stderr,"EXPECTED_REJECTION: %s\n",fmt); exit(87); }
void diagram_init_inverse_index(diagram_t *, size_t, block_t **);
static void coordinates(size_t key,int rank,int base,int *a) {
    for(int j=rank-1;j>=0;j--) { a[j]=key%base; key/=base; }
}
int main(int argc,char **argv) {
    if(argc>1) {
        diagram_t d={0}; block_t b[2]={0}; block_t *v[]={b,b+1};
        d.rank=6; d.n_blocks=2; d.blocks=v; n_spinor_blocks=3;
        if(!strcmp(argv[1],"overflow")) { n_spinor_blocks=SIZE_MAX; b[0].spinor_blocks[0]=2; }
        diagram_init_inverse_index(&d,2,v); return 91;
    }
    size_t queries=0;
    for(int rank=2;rank<=6;rank+=2) for(int base=2;base<=5;base++) {
        n_spinor_blocks=base; size_t total=1; for(int k=0;k<rank;k++) total*=base;
        diagram_t d={0}; d.rank=rank;
        block_t *all=calloc(total,sizeof(block_t));
        block_t **v=calloc(total,sizeof(block_t *)); size_t count=0;
        /* Deliberately reverse creation order and omit exact structural zeros. */
        for(size_t k=total;k-->0;) if(k%7!=0) {
            coordinates(k,rank,base,all[k].spinor_blocks); v[count++]=all+k;
        }
        d.blocks=v; d.n_blocks=count; allocated=0;
        diagram_init_inverse_index(&d,count,v);
        if(allocated!=2*sizeof(size_t)*count) return 92;
        for(size_t k=0;k<total;k++) {
            int a[CC_DIAGRAM_MAX_RANK]={0}; coordinates(k,rank,base,a);
            block_t *expected=k%7 ? all+k : NULL;
            if(diagram_get_block(&d,a)!=expected) return 93;
            queries++;
        }
        cc_free(d.inv_index); free(v); free(all);
    }
    diagram_t d={0}; d.rank=6; d.n_blocks=5; n_spinor_blocks=64;
    block_t b[5]={0}; block_t *v[5]; size_t total=(size_t)64*64*64*64*64*64;
    size_t keys[]={total-1,0,4095,15,1};
    for(int i=0;i<5;i++) { coordinates(keys[i],6,64,b[i].spinor_blocks); v[i]=b+i; }
    d.blocks=v; allocated=0; diagram_init_inverse_index(&d,5,v);
    if(allocated!=80) return 94;
    for(size_t k=0;k<4096;k++) {
        int a[CC_DIAGRAM_MAX_RANK]={0}; coordinates(k,6,64,a); block_t *expected=NULL;
        for(int i=0;i<5;i++) if(keys[i]==k) expected=b+i;
        if(diagram_get_block(&d,a)!=expected) return 95;
        queries++;
    }
    int a[CC_DIAGRAM_MAX_RANK]={63,63,63,63,63,63};
    if(diagram_get_block(&d,a)!=b) return 96;
    cc_free(d.inv_index); d.n_blocks=0;
    if(diagram_get_block(&d,a)!=NULL) return 97;
    printf("{\"status\":\"pass\",\"complete_small_queries\":%zu,\"large_base\":64,\"rank\":6,\"cartesian_extent\":%zu,\"dense_index_bytes\":%zu,\"compact_five_block_bytes\":%zu}\n",queries,total,total*sizeof(size_t),allocated);
    return 0;
}
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run_name):
    out = OWNER / run_name
    out.mkdir(exist_ok=False)

    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")

    if max(os.getloadavg()) > 27:
        raise RuntimeError("predeclared load27 gate rejects compiled replay launch")
    base = OWNER / "expt_axial_public_source001"
    build = base.with_name(base.name + "-build")
    parent = OWNER / "expt_axial_control002"
    if json.loads((parent / "result.json").read_text())["status"] != "pass":
        raise ValueError("complete original native/complex C1/axial gate required")
    old = base / "src/rcc/engine/diagram.c"
    modified, patch = patch_diagram(old.read_text())
    save(
        "contract.json",
        dict(
            hypothesis="complete compact block indices preserve original signed CCSDT and every valid lookup while allocating metadata only for actual blocks",
            fixed="same full native/complex Be 2core8virtual10magnetic CCSDT0h0p/0h1p, C1/axial; unchanged amplitude equations, integrals, tile4, raw1e-12 and residual bound1e-9 gates",
            signed_gate_Ha=1e-11,
            spectral_gate_Ha=1e-9,
            build_host_cap_bytes=2 * 1024**3,
            native_run_cap_bytes=1024**3,
            native_threads=1,
            build_workers=1,
            load_guard=27,
            storage="sorted full-key/original-block-number size_t pairs; dynamic original-order block list; ATXDG002 explicit private diagram format",
            unchanged="complete Cartesian traversal and every original symmetry/zero/permutation test; no block, spinor or amplitude selection",
            stop="first source, finite, signed, raw or resource violation rejects; duplicate/key overflow/old diagram formats explicitly rejected",
            scope="private metadata storage controls; no scalable enumeration/full production triples/performance/public adoption claim",
        ),
    )
    shutil.copy2(Path(__file__), out / "probe_source.py")
    shutil.copy2(OWNER / "expt_compact_index_patch.py", out / "patch_source.py")
    shutil.copy2(old, out / "diagram_original.c")
    shutil.copy2(base / "LICENSE", out / "LICENSE")
    (out / "diagram.c").write_text(modified)
    (out / "compact_index.patch").write_text(patch)
    flags = [
        "-DCOMPILER_GNU",
        "-O3",
        "-std=gnu99",
        "-fopenmp",
        "-DBLAS_OPENBLAS",
        "-DRETURN_BY_STACK",
    ]
    includes = [
        "-I" + str(p)
        for p in (
            build / "openblas",
            base / "openblas",
            base / "src/rcc/include",
            base / "src/rcc/engine",
        )
    ]
    obj = out / "diagram.o"
    commands = [["cc", *flags, *includes, "-c", str(out / "diagram.c"), "-o", str(obj)]]
    link = shlex.split((build / "CMakeFiles/expt.x.dir/link.txt").read_text())
    inherited = {
        str((build / x).resolve()): digest(build / x)
        for x in link
        if x.endswith((".o", ".a"))
    }
    link[link.index("CMakeFiles/expt.x.dir/src/rcc/engine/diagram.c.o")] = str(obj)
    link[link.index("-o") + 1] = str(out / "expt.x")
    commands.append(link)
    save(
        "source_binding.json",
        dict(
            original_diagram=digest(old),
            original_header=digest(base / "src/rcc/engine/diagram.h"),
            patch=digest(out / "compact_index.patch"),
            inherited_objects=inherited,
            probe=digest(__file__),
            patch_helper=digest(OWNER / "expt_compact_index_patch.py"),
            original_control=digest(parent / "result.json"),
            original_inputs=digest(parent / "inputs.npz"),
        ),
    )
    save("build_commands.json", commands)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    libs = Path(
        "/home/zzhang/code/atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu"
    )
    env["LD_LIBRARY_PATH"] = f"{libs}:{libs}/openblas-serial:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    with (out / "build.log").open("w") as log:
        for command in commands:
            subprocess.run(
                command,
                cwd=build,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
    (out / "harness.c").write_text(HARNESS)
    with (out / "lookup_build.log").open("w") as log:
        subprocess.run(
            [
                "cc",
                *flags,
                *includes,
                "-ffunction-sections",
                "-fdata-sections",
                str(out / "diagram.c"),
                str(out / "harness.c"),
                "-Wl,--gc-sections",
                "-o",
                str(out / "lookup.x"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    lookup = subprocess.check_output([str(out / "lookup.x")], text=True, timeout=30)
    save("lookup.json", json.loads(lookup))
    for case in ("duplicate", "overflow"):
        run = subprocess.run(
            [str(out / "lookup.x"), case], capture_output=True, text=True, timeout=30
        )
        if run.returncode != 87 or "EXPECTED_REJECTION" not in run.stderr:
            raise AssertionError("incomplete duplicate/key overflow refusal")
        save(
            case + "_rejection.json", dict(returncode=run.returncode, stderr=run.stderr)
        )

    records = []
    fci = np.load(parent / "inputs.npz", allow_pickle=False)["complete_fci_energies"][
        :8
    ]
    for case in ("native", "complex"):
        for mode in ("C1", "axial"):
            original = parent / f"{case}_{mode}"
            directory = out / f"{case}_{mode}"
            directory.mkdir()
            for name in (
                "cc.inp",
                "ATOMX_ONE",
                "ATOMX_TWO",
                "integrals.json",
                *(("ATOMX_AXIAL",) if mode == "axial" else ()),
            ):
                shutil.copy2(original / name, directory / name)
            save(
                f"{case}_{mode}_inputs.json",
                {
                    name: digest(directory / name)
                    for name in ("cc.inp", "ATOMX_ONE", "ATOMX_TWO", "integrals.json")
                },
            )
            with np.load(original / "full_signed_heff.npz", allow_pickle=False) as data:
                expected = data["matrix"].copy()
                expected_core = data["core_energy"].item()
                symmetry = data["symmetry"].copy()
            for phase in ("fresh", "restart"):
                if phase == "restart":
                    with (directory / "cc.inp").open("a") as f:
                        f.write("reuse 0h0p 0h1p\n")
                with (directory / (phase + ".log")).open("w") as log:
                    run = subprocess.run(
                        [str(out / "expt.x"), "--no-clean", "cc.inp"],
                        cwd=directory,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    )
                log = (directory / (phase + ".log")).read_text()
                statuses = {
                    s: (int(ok), int(it), *(float(v) for v in (d1, d2, d3)))
                    for s, ok, it, d1, d2, d3 in re.findall(
                        r"ATOMX_CC_STATUS (\dh\dp) (\d+) (\d+) (\S+) (\S+) (\S+)", log
                    )
                }
                if (
                    run.returncode
                    or set(statuses) != {"0h0p", "0h1p"}
                    or any(v[0] != 1 or max(v[2:]) > 1e-12 for v in statuses.values())
                ):
                    raise AssertionError("complete raw original CCSDT equations failed")
                if (
                    phase == "restart"
                    and log.count(" Trying to read amplitudes from disk") != 2
                ):
                    raise AssertionError(
                        "actual compact diagram restart was not executed"
                    )
                blocks = read_heff(directory / "scratch/HEFF")
                numbers = None if mode == "C1" else [_axial_irrep(*s) for s in symmetry]
                matrix = _model_matrix(blocks, "0h1p", tuple(range(2, 10)), numbers)
                core = blocks["0h0p", 1][0, 0]
                error = max(
                    float(np.max(abs(matrix - expected))), abs(core - expected_core)
                )
                spectrum = np.sort(np.linalg.eigvals(matrix).real) + core.real
                fci_error = float(np.max(abs(spectrum - fci)))
                # Use the same original full Fock energy scale for the fresh raw bound.
                with np.load(parent / "inputs.npz", allow_pickle=False) as inputs:
                    scale = 6 * float(np.max(abs(np.diag(inputs["fock"]))))
                residual_bound = scale * max(max(v[2:]) for v in statuses.values())
                if (
                    error > 1e-11
                    or fci_error > 1e-9
                    or residual_bound > 1e-9
                    or not np.all(np.isfinite(matrix))
                ):
                    raise AssertionError(
                        "complete signed/FCI/raw-bound original gate failed"
                    )
                np.savez_compressed(
                    directory / (phase + "_signed_heff.npz"),
                    matrix=matrix,
                    core_energy=core,
                    symmetry=symmetry,
                )
                records.append(
                    dict(
                        case=case,
                        mode=mode,
                        phase=phase,
                        signed_error_Ha=error,
                        fci_error_Ha=fci_error,
                        residual_bound_Ha=residual_bound,
                        statuses=statuses,
                    )
                )
                save("partial_result.json", records)
    if any(digest(path) != sha for path, sha in inherited.items()):
        raise ValueError("inherited backend object mutated during complete replay")
    save(
        "result.json",
        dict(
            status="pass",
            records=records,
            lookup=json.loads(lookup),
            executable_sha256=digest(out / "expt.x"),
            scope="private storage only; Cartesian enumeration and production triples remain open",
            public_adopted=False,
            highest_level_achieved=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    try:
        main(args.run_name)
    except BaseException as error:
        (OWNER / args.run_name / "failure.json").write_text(
            json.dumps(dict(type=type(error).__name__, error=str(error)), indent=2)
            + "\n"
        )
        raise
