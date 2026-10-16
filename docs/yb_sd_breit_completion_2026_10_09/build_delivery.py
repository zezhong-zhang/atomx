"""Ship complete dirty native source and replayable finite physical captures."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = (
    'test_ci_sd_space.py', 'test_ci_effective_qspace.py', 'test_finite_ccsd.py',
    'test_scaffolds.py', 'test_breit_open_shell_reference.py',
    'test_static_breit_retardation.py', 'test_ci_effective_one_body.py',
    'test_mbpt_matrix.py', 'test_mbpt_transitions.py', 'test_ci_mbpt_sigma1.py',
    'test_csf_sparse_projection.py', 'test_architecture_boundaries.py',
    'test_numerical_design_contracts.py', 'test_mcdhf_ci_silent_zeros.py',
    'test_review_regressions.py', 'test_mbpt_sigma2.py', '_be_reference.py',
)
PANEL = [f'tests/{name}' for name in TESTS[:13]] + [
    'tests/test_mcdhf_ci_silent_zeros.py::TestSameSubshellDoubles',
    'tests/test_mcdhf_ci_silent_zeros.py::TestMissingOrbitalGuard',
]


def main():
    files = set((ROOT / 'src/atomx').rglob('*.py'))
    files.update(ROOT / n for n in ('pyproject.toml', 'conftest.py', 'README.md', 'LICENSE', 'AGENTS.md'))
    files.update(ROOT / 'tests' / n for n in TESTS)
    files.update((ROOT / 'tests/fixtures/grasp_breit_open_shell').glob('*'))
    files.update((ROOT / 'docs').glob('*.md'))
    files.add(ROOT / 'docs/yb_diagnosis_repair_2026_10_08/common.py')
    files.add(ROOT / 'docs/rare_earth_yb_spectrum_2026_10_08/render_report.py')
    old = ROOT / 'docs/rare_earth_yb_spectrum_2026_10_08/results/v68_default'
    files.update(old / n for n in ('result.pkl', 'orbitals_workspace.npz', 'status.json', 'core.json', 'structure.json'))
    for name in ('ci_mbpt_matched_qspace_2026_10_09', 'ci_fscc_reference_upgrade_2026_10_09',
                 'yb_matrix_self_energy_2026_10_08', 'yb_correlation_tail_2026_10_08',
                 'yb_literature_repair_strategy_2026_10_08', 'yb_energy_dependent_sigma_2026_10_08',
                 'yb_diagnosis_repair_2026_10_08', 'rare_earth_yb_spectrum_2026_10_08'):
        files.add(ROOT / 'docs' / name / 'README.md')
    files.update((ROOT / 'docs/ci_mbpt_matched_qspace_2026_10_09').glob('*_qspace.json'))
    exclude = {'package_manifest.json', 'artifact_manifest.json', 'delivery_artifacts.json',
               'package_build.log', 'package_build_final.log', 'bundle_validation.json',
               'bundle_tests.log', 'bundle_replay.json', 'bundle_replay.log',
               'final_package_validation.json'}
    for f in HERE.rglob('*'):
        if (f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.zip', '.pyc')
                and f.name not in exclude and not f.name.startswith('delivery_receipt')):
            files.add(f)
    entries = {}
    dest = HERE / 'Atomx_有限CCSD_SD_Breit源码与验证.zip'
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for f in sorted(files):
            data = f.read_bytes()
            name = str(f.relative_to(ROOT))
            entries[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            archive.writestr(name, data)
        manifest = dict(
            files=entries, source_modules=sum(n.startswith('src/atomx/') for n in entries),
            scope='Complete current dirty native source, affected tests and dependencies, frozen Yb inputs, all five DC/DCB and native CCSD matrices, current guides and this increment evidence',
            tests=PANEL,
            replay='PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python docs/yb_sd_breit_completion_2026_10_09/replay_saved.py',
            full_assembly='run_panel.py rebuilds the five Yb sectors; probe_ccsd.py rebuilds the native CCSD anchor; dependencies from pyproject.toml required',
            exclusions=['No installed environment or upstream trees/binaries; previous pinned reference builds are retained locally',
                        'Older complete evidence topics remain sealed locally; their README and required one-hole records are included, not all linked archival data',
                        'No full repository test/doc archive; selected affected tests and root guides included',
                        'Delivery receipts, self-manifests, extracted-bundle verification and generated caches excluded'],
            scientific_status='119 affected tests passed; complete declared finite model numerical gates pass; Yb experimental and space targets fail; no scalable atomic SD or FS-CCSDT claim')
        archive.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    (HERE / 'package_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(files=len(entries), source_modules=manifest['source_modules'], bytes=dest.stat().st_size)))


if __name__ == '__main__':
    main()
