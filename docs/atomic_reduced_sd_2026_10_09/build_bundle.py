"""Full native source plus scoped SD evidence and original pCI reference archive."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
entries={}
def add(p,name=None):
    if not p.is_file():raise FileNotFoundError(p)
    entries[name or 'atomx/'+str(p.relative_to(ROOT))]=p
for p in sorted((ROOT/'src').rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts:add(p)
for name in ('pyproject.toml','LICENSE','README.md','AGENTS.md','PROJECT_INDEX.md'):add(ROOT/name)
add(ROOT/'tools/build_expt_backend.py')
for p in (ROOT/'docs').glob('*.md'):add(p)
tests=['test_linearized_sd.py','test_ci_sd_space.py','test_finite_ccsd.py','test_ci_effective_qspace.py','test_architecture_boundaries.py','test_architecture_scientific_fidelity.py']
for name in tests+['test_spinor_integrals_fscc.py']:add(ROOT/'tests'/name)
exclude={'artifact_manifest.json','package_manifest.json','package_build.log','bundle_validation.json','bundle_tests.log','delivery_artifacts.json','seal_validation.json','delivery_receipt_pdf.json','delivery_receipt_source.json'}
for p in sorted(HERE.rglob('*')):
    if p.is_file() and not any(x in p.parts for x in ['__pycache__','fresh_numba_cache','previews','extracted_bundle','extracted_pci']) and p.suffix!='.zip' and p.name not in exclude:add(p)
add(ROOT/'docs/atomic_streaming_tail_2026_10_09/full_yb_band.json')
add(ROOT/'docs/atomic_streaming_tail_2026_10_09/README.md')
add(BASE/'pCI_pristine_pinned.tar.gz','external/pCI_pristine_pinned.tar.gz')
source=json.loads((HERE/'final_source_sha256.json').read_text());assert all(sha(ROOT/k)==v for k,v in source.items())
manifest=dict(files={k:dict(sha256=sha(p),bytes=p.stat().st_size) for k,p in entries.items()},source_modules=len(source),tests=tests,scope='Full current native Atomx source; current guides and owning small SD evidence; original pinned GPLv3 pCI archive. No EXP-T executable, runtime/toolchain, large old integrals, compiler caches or full historical documentation/data collection.',licensing='Atomx GPL-3.0-only; original pCI GPLv3 copyright/licenses retained in its unmodified source archive.')
(HERE/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
instructions='''Atomx native scalar closed-core linearized SD

Read atomx/docs/ATOMIC_LINEARIZED_SD.md and
atomx/docs/atomic_reduced_sd_2026_10_09/README.md.
339-module final source is bound by final_source_sha256.json and all packaged
files by BUNDLE_MANIFEST.json. Current native APIs and no fitted corrections.

After installing Atomx Python dependencies:
  cd atomx
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_linearized_sd.py
Use a fresh NUMBA_CACHE_DIR for isolated full regression when required.
The six selected modules listed in the manifest comprise the 56-test panel.
No external backend is needed for that panel. The included spinor test module
supplies a native fixture, and is not part of that 56-test collection.

Regenerate measurements into NEW paths:
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python docs/atomic_reduced_sd_2026_10_09/probe_native.py --output-dir /fresh/native_sd
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python docs/atomic_reduced_sd_2026_10_09/probe_plan.py --band docs/atomic_streaming_tail_2026_10_09/full_yb_band.json --output /fresh/yb_plan.json

For the independent reference angular check, extract external/pCI_pristine_pinned.tar.gz.
Compile that original lib/all-order/include/libD.f with the provided driver:
  gfortran -O2 -std=legacy -ffixed-line-length-none -I /path/pCI/lib/all-order/include atomx/docs/atomic_reduced_sd_2026_10_09/pci_angular_driver.f90 /path/pCI/lib/all-order/include/libD.f -o /fresh/pci_angular
Then run probe_pci_angular.py with --source /path/pCI --executable /fresh/pci_angular --output-dir /fresh/angular --revision 4d685d82bfdf27876098a87808dadc5fcf3038d1 --library-path /optional/runtime/path.
The archive source has a fixed upstream commit recorded in the evidence. When
the directory lacks .git, explicit --revision and original source hashes are
verified against the pinned angular evidence before the run.
Original pCI code is GPLv3; retain its license and copyrights. The small supplied
angular executable is Linux x86_64 and needs libgfortran.so.5. Rebuild on other
platforms. EXP-T builds remain separately available from the preceding package.

Do not infer Yb accuracy from a closed-core scalar count, residual or narrow Ne
box-root pool. Direct large-pool contractions, monovalent SD/Bloch feedback,
CI+all-order Sigma1/Sigma2, dressed properties and full physical convergence
remain unfinished. 310-fold coordinate reduction is not a runtime speedup.
Some current guide links refer to older repository topics not fully packaged.
'''
with zipfile.ZipFile(HERE/'Atomx_原生约化核SD源码与验证.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in entries.items():z.write(p,name)
    z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n');z.writestr('REPLAY.txt',instructions)
print(json.dumps(dict(files=len(entries),source_modules=len(source),bytes=(HERE/'Atomx_原生约化核SD源码与验证.zip').stat().st_size)))
