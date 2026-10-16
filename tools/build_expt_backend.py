"""Build the pinned EXP-T integral backend in a separate, patched checkout.

Does not download code, modify the reference checkout or install globally.
The upstream LGPL-3.0 source and license remain in the copied tree. The
generated patch records the complete interface change for redistribution.
"""
import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

PIN = "aa8d809ad5818d4aa618dcdadbd7f12257ecbc28"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"upstream interface mismatch: {old[:80]!r}")
    return text.replace(old, new)


def patch_source(source, *, axial_parity=False, compact_block_index=False,
                 noncanonical_attachment_fix=False, axial_irrep_capacity=256,
                 compact_axial_eri=False, exact_nonzero_division=False):
    if axial_irrep_capacity not in (256, 512) or (not axial_parity and axial_irrep_capacity != 256):
        raise ValueError('explicit exact axial capacity256/512 requires axial mode')
    if compact_axial_eri and not axial_parity:
        raise ValueError('compact integral storage requires exact axial mode')
    changes = []
    p = source / "src/rcc/engine/mult.c"
    old = s = p.read_text()
    fix = '#if defined BLAS_OPENBLAS\n  #include "cblas.h"\n#endif\n'
    if fix not in s:
        s = replace_once(s, '#include "tt.h"\n', '#include "tt.h"\n\n'+fix)
        p.write_text(s)
        changes.append((p.relative_to(source), old, s))
    p = source / "src/rcc/interfaces/pyscf_interface.c"
    old = s = p.read_text()
    s = replace_once(s, "size_t *pyscf_eri_index = NULL;", "size_t *pyscf_eri_index = NULL;\ndouble complex *atomx_fock = NULL;")
    s = replace_once(s, 'printf("\\n begin interface to pyscf\\n");', 'printf("\\n ATOMX_INTEGRALS_V1 signed-antisymmetric complex Fock+ERI\\n");')
    s = replace_once(s, '    printf(" switching tile size to %d\\n", CC_MAX_SPINORS);\n    opts->tile_size = CC_MAX_SPINORS;', '    printf(" retained requested tile size %d\\n", opts->tile_size);')
    s = replace_once(s, '    *nspinors = 0;\n    io_read(f, nspinors, sizeof(int));', '''    char magic[8];
    io_read(f, magic, 8);
    if (memcmp(magic, "ATXCC001", 8) != 0 || cc_opts->use_goldstone) {
        errquit("Atomx backend requires ATXCC001 antisymmetrized integrals");
    }
    *nspinors = 0;
    io_read(f, nspinors, sizeof(int));
    if (*nspinors < 2 || *nspinors > CC_MAX_SPINORS) {
        errquit("Atomx spinor count exceeds this EXP-T build limit");
    }''')
    s = replace_once(s, '    /*\n     * create index for fast random access', '''    atomx_fock = cc_calloc(nspinors64 * nspinors64, sizeof(double complex));
    io_read(f, atomx_fock, nspinors64 * nspinors64 * sizeof(double complex));

    /*
     * create index for fast random access''')
    s = replace_once(s, '/**eri = cc_calloc', '*eri = cc_calloc')
    s = replace_once(s, '    }*/\n\n    /*\n     * read two-electron', '    }\n\n    /*\n     * read two-electron')
    s = replace_once(s, '    /*printf("begin reading two electron part\\n");', '    printf("begin reading two electron part\\n");')
    s = replace_once(s, '    io_close(f);*/', '    io_close(f);')
    p.write_text(s)
    changes.append((p.relative_to(source), old, s))
    p = source / "src/rcc/sorting/sort_pyscf.c"
    old = s = p.read_text()
    s = replace_once(s, 'extern size_t *pyscf_eri_index;', 'extern size_t *pyscf_eri_index;\nextern double complex *atomx_fock;')
    start, end = s.index('void sort_pyscf_one_electron()'), s.index('void fill_block_twoelec_complex_pyscf(block_t *block)\n{')
    s = s[:start] + '''void sort_pyscf_one_electron()
{
    for (int i = 0; i < n_requests; i++) {
        diagram_t *dg = sorting_requests[i].dg;
        if (dg->rank != 2) continue;
        for (int j = 0; j < dg->n_blocks; j++) {
            block_t *sb = dg->blocks[j];
            if (!sb->is_unique) continue;
            block_load(sb);
            fill_block_one_elec(sb, atomx_fock, 1);
            block_store(sb);
        }
    }
}

void sort_pyscf_two_electron()
{
    for (int i = 0; i < n_requests; i++) {
        sorting_request_t *req = &sorting_requests[i];
        diagram_t *dg = req->dg;
        if (dg->rank != 4) continue;
        for (int j = 0; j < dg->n_blocks; j++) {
            block_t *sb = dg->blocks[j];
            if (!sb->is_unique) continue;
            block_load(sb);
            fill_block_twoelec_complex_pyscf(sb);
            if (is_pppp_diagram(req->hp, req->valence, req->order)) {
                conj_vector(sb->size, sb->buf);
            }
            block_store(sb);
        }
    }
}

''' + s[end:]
    s = replace_once(s, 'void pyscf_data_free()\n{', 'void pyscf_data_free()\n{\n    cc_free(atomx_fock);\n    atomx_fock = NULL;')
    p.write_text(s)
    changes.append((p.relative_to(source), old, s))
    # Machine-readable raw fixed-point differences, before DIIS/damping.
    p = source / "src/rcc/models/ccutils.c"
    old = s = p.read_text()
    s = replace_once(s, '        if (converged || diverged) {\n            goto end_of_iter;', '''        if (converged || diverged) {
            printf("\\nATOMX_CC_STATUS %dh%dp %d %d %.16e %.16e %.16e\\n",
                sector_h, sector_p, converged, iter,
                do_singles ? diff1 : 0.0, do_doubles ? diff2 : 0.0,
                do_triples ? diff3 : 0.0);
            goto end_of_iter;''')
    p.write_text(s)
    changes.append((p.relative_to(source), old, s))
    p = source / "src/rcc/readinp/readinp.c"
    old = s = p.read_text()
    s = replace_once(s, '    opts->actsp_defined = CC_ACT_SPACE_SPEC_BINARY;',
                     '    put_back(token_type);\n    opts->actsp_defined = CC_ACT_SPACE_SPEC_BINARY;')
    p.write_text(s)
    changes.append((p.relative_to(source), old, s))
    p = source / "src/rcc/heff/formatted_heff_file.c"
    old = s = p.read_text()
    if '%21.12E' not in s:
        raise ValueError('upstream HEFF precision format mismatch')
    s = s.replace('%21.12E', '%26.17E')
    p.write_text(s)
    changes.append((p.relative_to(source), old, s))
    patch = ''.join(''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile='a/'+str(p), tofile='b/'+str(p))) for p, a, b in changes)
    if axial_parity:
        from expt_axial_patch import patch_axial
        patch += patch_axial(source, irrep_capacity=axial_irrep_capacity)
    if compact_axial_eri:
        from expt_compact_eri_patch import patch_compact_eri
        patch += patch_compact_eri(source)
    if compact_block_index:
        from expt_compact_index_patch import patch_diagram
        # The separable zero predicate is part of this exact representation
        # change. A different upstream implementation needs a new derivation.
        spinors = (source / 'src/rcc/spinors.c').read_text()
        begin = spinors.index('int is_symblock_zero(')
        end = spinors.index('\n\n\nint spinor_attr_cmp_by_eps', begin)
        if hashlib.sha256(spinors[begin:end].encode()).hexdigest() != '813a2b097ef864460c941f93c54d1bffd2c15a0b17ec2db597782c2cf2fe5f89':
            raise ValueError('upstream independent block-axis zero predicate differs')
        diagram = source / 'src/rcc/engine/diagram.c'
        value, compact_patch = patch_diagram(diagram.read_text(), reachable_constructor=True)
        diagram.write_text(value)
        patch += compact_patch
    if noncanonical_attachment_fix:
        from expt_noncanonical_patch import patch_noncanonical_attachment
        patch += patch_noncanonical_attachment(source)
    if exact_nonzero_division:
        from expt_numerator_division_patch import patch_exact_nonzero_division
        patch += patch_exact_nonzero_division(source)
    return patch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--toolchain', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--fortran-compiler', type=Path)
    ap.add_argument('--axial-parity', action='store_true',
                    help='Enable optional exact signed M/parity tensor blocks')
    ap.add_argument('--axial-irrep-capacity', type=int, choices=(256, 512), default=256,
                    help='Explicit compiled exact symmetry-table capacity; default unchanged')
    ap.add_argument('--compact-block-index', action='store_true',
                    help='Experimental exact compact indices and nonempty block axes; explicit ATXDG002 diagram files')
    ap.add_argument('--compact-axial-eri', action='store_true',
                    help='Explicit exact M/parity integral-bank storage; requires axial mode')
    ap.add_argument('--noncanonical-attachment-fix', action='store_true',
                    help='Opt in to the independently verified open Fov*T2 antisymmetry fix in 0h1p equations')
    ap.add_argument('--exact-nonzero-division', action='store_true',
                    help='Divide every nonzero numerator, including values below the legacy1e-14 skip threshold')
    args = ap.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN or args.destination.exists():
        raise ValueError('requires pinned reference and fresh destination')
    upstream_diff = subprocess.check_output(['git', '-C', str(args.source), 'diff', '--binary'], text=True)
    changed = subprocess.check_output(['git', '-C', str(args.source), 'diff', '--name-only'], text=True).splitlines()
    if changed and changed != ['src/rcc/engine/mult.c']:
        raise ValueError('reference checkout has unreviewed modifications')
    shutil.copytree(args.source, args.destination, ignore=shutil.ignore_patterns('.git'))
    patch = patch_source(args.destination, axial_parity=args.axial_parity,
                         compact_block_index=args.compact_block_index,
                         noncanonical_attachment_fix=args.noncanonical_attachment_fix,
                         axial_irrep_capacity=args.axial_irrep_capacity,
                         compact_axial_eri=args.compact_axial_eri,
                         exact_nonzero_division=args.exact_nonzero_division)
    (args.destination / 'atomx_interface.patch').write_text(upstream_diff+patch)
    # Existing reference build records the GNU/OpenBLAS declaration fix.
    env = dict(os.environ)
    libs = args.toolchain / 'usr/lib/x86_64-linux-gnu'
    env['LD_LIBRARY_PATH'] = f"{libs}:{libs}/openblas-serial:" + env.get('LD_LIBRARY_PATH', '')
    cmd = ['uv', 'run', '--no-project', '--with', 'cmake==3.31.6', 'cmake']
    build = args.destination.with_name(args.destination.name+'-build')
    compiler = args.fortran_compiler or args.toolchain.parent/'gfortran'
    if not compiler.exists():
        compiler = shutil.which('gfortran')
    if not compiler:
        raise ValueError('Fortran compiler missing; specify --fortran-compiler')
    subprocess.run(cmd+['-S', str(args.destination), '-B', str(build),
        '-DCMAKE_Fortran_COMPILER='+str(compiler),
        '-DCMAKE_PREFIX_PATH='+str(args.toolchain/'usr'),
        '-DCMAKE_LIBRARY_PATH='+str(libs/'openblas-serial'), '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch', '-DBUILD_TESTING=OFF', '-DTARGET=HASWELL'], env=env, check=True)
    subprocess.run(cmd+['--build', str(build), '--parallel', str(args.workers)], env=env, check=True)
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    (build/'atomx_backend.json').write_text(json.dumps(dict(upstream_revision=revision,
        abi='ATXCC001', executable=str(build/'expt.x'), executable_sha256=sha(build/'expt.x'),
        patch_sha256=sha(args.destination/'atomx_interface.patch'),
        tensor_symmetries=['C1', 'axial'] if args.axial_parity else ['C1'],
        axial_max_omega_x2=args.axial_irrep_capacity//4-2 if args.axial_parity else None,
        eri_storage_layouts=['packed', 'axial_compact'] if args.compact_axial_eri else ['packed'],
        block_index_layout='complete sorted key/index pairs' if args.compact_block_index else 'pinned dense Cartesian inverse index',
        diagram_format='ATXDG002' if args.compact_block_index else 'pinned EXP-T diagram format',
        constructor_axes='exact original nonempty axes' if args.compact_block_index else 'complete original Cartesian axes',
        noncanonical_attachment_equation='ATOMX_FOV_T2_001' if args.noncanonical_attachment_fix else 'pinned original equations',
        numerator_division=('ATOMX_EXACT_NONZERO_DIVEPS_001' if args.exact_nonzero_division
                            else 'ATOMX_LEGACY_NUMERATOR_SKIP_1E14'),
        source_sha256={str(p.relative_to(args.destination)): sha(p) for p in sorted((args.destination/'src').rglob('*'))
                       if p.is_file()}), indent=2))


if __name__ == '__main__':
    main()
