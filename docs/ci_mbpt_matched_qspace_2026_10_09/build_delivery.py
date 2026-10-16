"""Complete dirty native source, finite-model inputs, tests and evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    files=set((ROOT/'src/atomx').rglob('*.py'))
    files.update(ROOT/n for n in ('pyproject.toml','conftest.py','README.md','LICENSE','AGENTS.md'))
    tests=('test_ci_effective_qspace.py','test_mbpt_matrix.py','test_mbpt_transitions.py',
           'test_ci_effective_one_body.py','test_ci_mbpt_sigma1.py','test_architecture_boundaries.py',
           'test_numerical_design_contracts.py','test_review_regressions.py','test_mbpt_sigma2.py')
    files.update(ROOT/'tests'/n for n in tests)
    files.update(ROOT/'docs'/n for n in ('README.md','ARCHITECTURE_CAPABILITIES.md','ARCHITECTURE_DESIGN.md',
                                       'TESTING.md','VERIFICATION.md','AGENT_REFERENCE.md','AGENT_INSIGHTS.md'))
    files.add(ROOT/'docs/yb_diagnosis_repair_2026_10_08/common.py')
    files.add(ROOT/'docs/rare_earth_yb_spectrum_2026_10_08/render_report.py')
    old=ROOT/'docs/rare_earth_yb_spectrum_2026_10_08/results/v68_default'
    files.update(old/n for n in ('result.pkl','orbitals_workspace.npz','status.json','core.json','structure.json'))
    setup=ROOT/'docs/ci_fscc_reference_upgrade_2026_10_09'
    files.update(setup/n for n in ('download_references.py','build_references.py','toolchain.json',
                                 'downloads.json','reference_builds.json','README.md',
                                 'pCI_gnu_string_constructor.patch','EXP-T_gnu15_openblas_declaration.patch'))
    excluded_names={'package_manifest.json','artifact_manifest.json','delivery_artifacts.json',
                    'ambit_sigma_probe','bundle_records_validation.log','package_build.log',
                    'bundle_run.log','final_bundle_run.log','final_package_validation.json',
                    'final_package_validation.log'}
    for f in HERE.rglob('*'):
        if (f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.zip','.o','.pyc')
                and f.name not in excluded_names and not f.name.startswith('delivery_receipt')):
            files.add(f)
    entries={}
    dest=HERE/'Atomx_有限Q空间源码与验证.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(files):
            data=f.read_bytes(); name=str(f.relative_to(ROOT))
            entries[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            z.writestr(name,data)
        manifest=dict(files=entries,scope='Complete current dirty native source (335 modules), selected affected tests, native Yb frozen input, all declared finite Q matrices and controls; not only this task delta',
            exclusions=['External upstream trees, compiler, static libraries and probe executable remain at their pinned recorded local paths; source adapter and setup metadata included',
                        'Large old Sigma pole captures and six complete archived evidence packages remain sealed locally; not needed for finite Yb replay',
                        'Delivery receipts, archive self-manifests, extracted-bundle verification outputs and generated object files are excluded'],
            replay='PYTHONPATH=src:. with compatible installed scientific dependencies; finite Yb scripts can replay directly in an extracted copy; matched AMBiT adapter additionally requires the recorded pinned build and toolchain')
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    (HERE/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(entries),modules=sum(n.startswith('src/atomx/') for n in entries),bytes=dest.stat().st_size)))

if __name__=='__main__': main()
