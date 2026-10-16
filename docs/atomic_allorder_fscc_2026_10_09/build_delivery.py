"""Reviewable source plus small replay inputs; large captures remain sealed."""
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
SELECTED = ['test_spinor_integrals_fscc.py', 'test_finite_ccsd.py', 'test_ci_sd_space.py',
            'test_ci_effective_qspace.py', 'test_ci_effective_one_body.py', 'test_mbpt_matrix.py',
            'test_architecture_boundaries.py', 'test_architecture_scientific_fidelity.py']
TEST_HELPERS = ['test_review_regressions.py', 'test_mbpt_sigma2.py']


def main():
    entries = {}
    def add(path, name):
        if path.is_file():
            entries[name] = path
    for path in sorted((ROOT/'src').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path, 'atomx/'+str(path.relative_to(ROOT)))
    for name in ('README.md', 'LICENSE', 'AGENTS.md', 'pyproject.toml', 'tools/build_expt_backend.py'):
        add(ROOT/name, 'atomx/'+name)
    for path in sorted((ROOT/'docs').glob('*.md')):
        add(path, 'atomx/'+str(path.relative_to(ROOT)))
    for name in SELECTED + TEST_HELPERS:
        add(ROOT/'tests'/name, 'atomx/tests/'+name)
    # The baseline loader and exact V68 inputs support the Yb reproduction.
    old = ROOT/'docs/rare_earth_yb_spectrum_2026_10_08/results/v68_default'
    for name in ('result.pkl', 'status.json', 'core.json', 'structure.json', 'orbitals_workspace.npz'):
        add(old/name, 'atomx/'+str((old/name).relative_to(ROOT)))
    p = ROOT/'docs/yb_diagnosis_repair_2026_10_08/common.py'
    add(p, 'atomx/'+str(p.relative_to(ROOT)))
    excluded = {'artifact_manifest.json', 'package_manifest.json', 'delivery_artifacts.json',
                'bundle_validation.json', 'bundle_tests.log', 'package_validation.log', 'delivery_receipt_report.json',
                'delivery_receipt_source.json'}
    large = []
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.zip', '.dg') or path.name in excluded:
            continue
        if path.stat().st_size > 32*1024**2:
            large.append(dict(path=str(path), bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            continue
        add(path, 'atomx/'+str(path.relative_to(ROOT)))
    upstream = BASE/'EXP-T-atomx'
    source_tar = HERE/'EXP-T-atomx-source.tar.xz'
    # A solid source archive removes cross-file Fortran repetition without
    # omitting the build-essential OpenBLAS source or its original licenses.
    entries.pop('atomx/'+str(source_tar.relative_to(ROOT)), None)
    roots = [upstream/n for n in ('src', 'openblas', 'docs', 'CMakeLists.txt', 'LICENSE',
                                  'README.md', 'WHATISNEW', 'atomx_interface.patch')]
    source_paths = [p for top in roots for p in (top.rglob('*') if top.is_dir() else [top]) if p.is_file()]
    source_identity = {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    reusable = False
    if source_tar.exists():
        with tarfile.open(source_tar, 'r:xz') as tar:
            stored = {m.name.removeprefix('EXP-T-atomx/'): hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                      for m in tar.getmembers() if m.isfile()}
        reusable = stored == source_identity
    if not reusable:
        with tarfile.open(source_tar, 'w:xz') as tar:
            for top in roots:
                tar.add(top, arcname='EXP-T-atomx/'+str(top.relative_to(upstream)))
    archive_identity = HERE/'backend_source_archive_identity.json'
    archive_identity.write_text(json.dumps(dict(files=source_identity,
        archive_sha256=hashlib.sha256(source_tar.read_bytes()).hexdigest()), indent=2)+'\n')
    add(archive_identity, 'atomx/'+str(archive_identity.relative_to(ROOT)))
    add(source_tar, 'external/EXP-T-atomx-source.tar.xz')
    for name in ('expt.x', 'atomx_backend.json'):
        add(BASE/'EXP-T-atomx-build'/name, 'external/backend-linux-x86_64/'+name)
    manifest = dict(files={name: dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                           for name, path in entries.items()},
                    native_source_modules=len(list((ROOT/'src/atomx').rglob('*.py'))),
                    large_inputs_kept_locally=large,
                    scope=['Complete current Atomx source; current guides; eight affected test modules and their two helper dependencies',
                           'Pinned modified EXP-T build-essential source, OpenBLAS source, original licenses, docs and binary/source metadata',
                           'Native anchors and retained Yb V68 inputs; small signed input files and full logs/effective matrices',
                           'Large >32MiB captured integral files remain at manifest paths; Ne can be regenerated from probe_band.py',
                           'Backend scratch diagrams/amplitudes remain local; required SHA/inputs/results are recorded',
                           'Compiled binary is Linux x86_64/HASWELL and needs GNU libgfortran.so.5; runtime/toolchain is not bundled'])
    (HERE/'package_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    instructions = '''Atomx tensor CCSD / FS-CCSDT source and measured evidence

Read atomx/docs/ATOMIC_FSCC.md and
atomx/docs/atomic_allorder_fscc_2026_10_09/README.md first.

Atomx is GPL-3.0-only. EXP-T is LGPL-3.0; retain its LICENSE and
copyright notices. Its internal OpenBLAS carries its own source licenses.
Extract external/EXP-T-atomx-source.tar.xz to obtain the backend source.
It includes every build-essential patched file;
the pinned upstream revision and patch SHA are in atomx_backend.json.

The extracted source is already patched. Configure it directly with CMake
and a GNU Fortran compiler, or use the build tool with an original pinned
git checkout to reproduce the patch independently.

The supplied backend executable is Linux x86_64 / HASWELL. It needs a GNU
Fortran runtime (libgfortran.so.5) and normal GNU/OpenMP/C libraries. Rebuild
the included source on other systems, or use tools/build_expt_backend.py
with a pinned checkout and local toolchain. No accounts/configuration or
credentials are included.

After preparing Atomx's Python dependencies and the runtime library path:
  chmod +x external/backend-linux-x86_64/expt.x
  export ATOMX_EXPT_EXECUTABLE=/absolute/bundle/external/backend-linux-x86_64/expt.x
  cd atomx
  PYTHONPATH=src python -m pytest -q tests/test_spinor_integrals_fscc.py

That runs actual electronic CCSDT against independent full CI, including
complex non-diagonal inputs and attachment/detachment. Without the variable
the external gate is explicitly skipped.

Long Yb/Ne inputs larger than 32MiB are not copied into this ZIP. Their exact
stable local paths, byte sizes and SHA are in BUNDLE_MANIFEST.json. All backend
temporary .dg diagrams are retained locally rather than shipped. Full Ne
native input can be regenerated from the declared band script. The V68 Yb
reference is included; choose a fresh output directory for reruns. Published
results carry their actual source/binary/input identity; experimental and
space gates remain failed.
'''
    dest = HERE/'Atomx_全阶SD_FSCCSDT源码与验证.zip'
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
        archive.writestr('REPLAY.txt', instructions)
    print(json.dumps(dict(files=len(entries), bytes=dest.stat().st_size,
                         native_modules=manifest['native_source_modules'], omitted_large_inputs=len(large))))


if __name__ == '__main__':
    main()
