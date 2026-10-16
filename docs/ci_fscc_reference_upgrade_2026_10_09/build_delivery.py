"""Package current native source, consumer tests, docs and bounded evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]
if __name__=='__main__':
    files=set((REPO/'src/atomx').rglob('*.py'))
    files.update(REPO/n for n in ('pyproject.toml','conftest.py','README.md','LICENSE','AGENTS.md'))
    files.update(REPO/'tests'/n for n in ('test_ci_effective_one_body.py','test_mbpt_matrix.py',
                                         'test_ci_mbpt_sigma1.py','test_review_regressions.py','test_mbpt_sigma2.py'))
    files.update(REPO/'docs'/n for n in ('README.md','ARCHITECTURE_CAPABILITIES.md','ARCHITECTURE_DESIGN.md',
                                        'TESTING.md','VERIFICATION.md','AGENT_REFERENCE.md','AGENT_INSIGHTS.md'))
    files.add(REPO/'docs/rare_earth_yb_spectrum_2026_10_08/render_report.py')
    for p in OUT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        if p.suffix in ('.pyc','.zip','.int','.basis','.levels','.png'):
            continue
        if p.name in ('pCI_primary_paper.pdf','pCI_primary_paper.txt','package_manifest.json','artifact_manifest.json','delivery_artifacts.json'):
            continue
        if p.name.startswith('delivery_receipt') or p.name=='minitest.py' or p.parent.name=='expt_vibrot' and p.suffix=='.py':
            continue
        files.add(p)
    entries={}
    destination=OUT/'Atomx_CI_MBPT_FS_RCC源码与验证.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(files):
            data=p.read_bytes();name=str(p.relative_to(REPO));entries[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            archive.writestr(name,data)
        manifest=dict(files=entries,excluded={
            'upstream_source_and_toolchain':'Official pinned trees, pristine tarballs, builds and isolated compiler stay in the recorded local directory; reproduce using download_references.py and explicit build ports',
            'copyrighted_paper_fulltext':'Public primary paper linked in README; locally downloaded PDF/text omitted from source bundle',
            'copied_upstream_test_scripts':'EXP-T test.py/minitest.py remain in the pinned upstream tree; numerical outputs retained',
            'old_Yb_matrix_snapshots':'Ten large poles snapshots and old reference archives remain sealed locally; replay requires their recorded paths/hashes',
            'scope':'Contains complete current dirty native source, not only this task delta; before/final source hashes distinguish the five changed native files',
        })
        archive.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
    (OUT/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(path=str(destination),files=len(entries),bytes=destination.stat().st_size)))
