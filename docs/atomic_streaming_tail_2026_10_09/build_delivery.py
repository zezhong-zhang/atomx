"""Package final source and replayable evidence; preserve large local captures."""
import json
from pathlib import Path
import zipfile

from verify_evidence import sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'docs/atomic_allorder_fscc_2026_10_09'
BASE = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
TESTS = ['test_spinor_integrals_fscc.py', 'test_finite_ccsd.py', 'test_ci_sd_space.py',
         'test_ci_effective_qspace.py', 'test_ci_effective_one_body.py', 'test_mbpt_matrix.py',
         'test_architecture_boundaries.py', 'test_architecture_scientific_fidelity.py',
         'test_streaming_spinor_integrals.py', 'test_ci_radial_snapshot.py']
HELPERS = ['test_review_regressions.py', 'test_mbpt_sigma2.py']
ZIPNAME = 'Atomx_按需积分与虚轨道扩展源码.zip'


def main():
    entries = {}
    omitted = {}
    def add(path, name=None):
        if not path.is_file():
            raise FileNotFoundError(path)
        name = name or 'atomx/'+str(path.relative_to(ROOT))
        if path.suffix == '.dg' or path.stat().st_size > 32*1024**2:
            omitted[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path),
                                     reason='Backend scratch/amplitude or >32-MiB integral capture retained locally')
        else:
            entries[name] = path
    for path in sorted((ROOT/'src').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path)
    for name in ('README.md', 'LICENSE', 'AGENTS.md', 'pyproject.toml', 'tools/build_expt_backend.py'):
        add(ROOT/name)
    for path in sorted((ROOT/'docs').glob('*.md')):
        add(path)
    for name in TESTS+HELPERS:
        add(ROOT/'tests'/name)
    baseline = ROOT/'docs/rare_earth_yb_spectrum_2026_10_08/results/v68_default'
    for name in ('result.pkl', 'status.json', 'core.json', 'structure.json', 'orbitals_workspace.npz'):
        add(baseline/name)
    add(ROOT/'docs/yb_diagnosis_repair_2026_10_08/common.py')
    for name in ('README.md', 'results.json', 'ne_full_band_summary.json', 'backend_source_archive_identity.json',
                 'EXP-T_atomx_interface.patch'):
        add(OLD/name)
    for directory in ('be_ccsdt_tile4_final', 'yb_o22_r2', 'ne_full_band'):
        for path in sorted((OLD/directory).rglob('*')):
            if path.is_file():
                add(path)
    excluded = {'artifact_manifest.json', 'package_manifest.json', 'delivery_artifacts.json',
                'bundle_validation.json', 'bundle_tests.log', 'package_validation.log',
                'delivery_receipt_report.json', 'delivery_receipt_source.json', 'seal_validation.json',
                'package_build.log'}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.zip' or path.name in excluded or 'previews' in path.parts:
            continue
        add(path)
    # Read-only reuse of the previously verified LGPL/OpenBLAS source archive.
    add(OLD/'EXP-T-atomx-source.tar.xz', 'external/EXP-T-atomx-source.tar.xz')
    for name in ('expt.x', 'atomx_backend.json'):
        add(BASE/'EXP-T-atomx-build'/name, 'external/backend-linux-x86_64/'+name)
    source = json.loads((HERE/'final_source_sha256.json').read_text())
    assert all(sha(ROOT/k) == v for k, v in source.items())
    manifest = dict(files={name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in entries.items()},
                    native_source_modules=len(source), large_or_scratch_files_kept_locally=omitted,
                    tests=TESTS, test_helpers=HELPERS,
                    scope=['Complete final 338-module Atomx source and current guides; ten selected test modules and two imported helpers',
                           'Pinned patched EXP-T executable and full build-essential LGPL/OpenBLAS source from the unchanged prior seal',
                           'Exact Yb V68 baseline, old Be/Yb electronic anchors, Ne manifests and all current small replay evidence',
                           'Integral captures >32MiB and backend .dg diagrams remain local with byte/hash identities',
                           'Linux x86_64/HASWELL executable needs GNU libgfortran.so.5; runtime/toolchain not bundled'])
    (HERE/'package_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    instructions = '''Atomx native on-demand integrals and controlled virtual-tail continuation

Read atomx/docs/ATOMIC_FSCC.md and
atomx/docs/atomic_streaming_tail_2026_10_09/README.md.
The current 338-module source is bound by final_source_sha256.json. Initial
and final implementation captures distinguish the 54-virtual solve from
its final-source export bridge. Earlier owning-topic evidence is unchanged.

Atomx is GPL-3.0-only; EXP-T is LGPL-3.0. Preserve all copyright/licenses.
Extract external/EXP-T-atomx-source.tar.xz for the already patched source,
including build-essential OpenBLAS source and licenses. Configure directly
with CMake/GNU Fortran, or use tools/build_expt_backend.py with an original
pinned checkout to reproduce the patch independently. Backend identity and
source-file hashes are under atomic_allorder_fscc_2026_10_09/.

The supplied executable is Linux x86_64/HASWELL and requires libgfortran.so.5
and normal GNU/OpenMP/C libraries. Runtime/toolchain are not bundled; rebuild
on other architectures. No accounts, configuration or credentials are included.

After preparing Python dependencies and the runtime library path:
  chmod +x external/backend-linux-x86_64/expt.x
  export ATOMX_EXPT_EXECUTABLE=/absolute/bundle/external/backend-linux-x86_64/expt.x
  cd atomx
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q \\
    tests/test_streaming_spinor_integrals.py tests/test_spinor_integrals_fscc.py

The external test gate runs actual signed electronic FS-CCSDT against full CI;
without ATOMX_EXPT_EXECUTABLE these external tests are explicitly skipped.

Choose fresh output names; saved evidence directories are not rerun destinations:
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python \\
    docs/atomic_streaming_tail_2026_10_09/probe_ne.py \\
    --representation on_demand --name fresh_ne --solve
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python \\
    docs/atomic_streaming_tail_2026_10_09/probe_yb.py \\
    --radials 2 --name fresh_yb36
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python \\
    docs/atomic_streaming_tail_2026_10_09/probe_yb.py \\
    --radials 3 --name fresh_yb54
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python \\
    docs/atomic_streaming_tail_2026_10_09/probe_yb.py \\
    --radials 4 --name fresh_yb72

These regenerate complete declared inputs and use the supplied executable.
Ne's ~1.12GB integral captures and larger Yb integrals are retained locally;
the probes verify regenerated signed bytes against included manifests. The
54-virtual export-only bridge consumes its original G capture, so regenerate
that input first or use a fresh solve. The exact 36-virtual Yb and Be electronic
inputs are included. Every omitted .dg or >32MiB capture has a stable local
path, size and SHA in BUNDLE_MANIFEST.json; the ZIP does not claim to include
those files. Performance ratios apply only to the named native export phase.

Complete Yb operator construction retained all 1894 virtual spinors below
5Ha/l<=3, but full C1 CC solving remains over budget. Native memory savings
do not certify complete-band CC, physical resonance poles, or experimental
accuracy. The reported Yb space and experimental gates still fail.
'''
    destination = HERE/ZIPNAME
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
        archive.writestr('REPLAY.txt', instructions)
    print(json.dumps(dict(files=len(entries), bytes=destination.stat().st_size,
                         native_modules=len(source), omitted_files=len(omitted))))


if __name__ == '__main__':
    main()
