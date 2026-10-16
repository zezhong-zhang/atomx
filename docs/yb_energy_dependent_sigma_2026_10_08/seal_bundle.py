"""Seal full local evidence and build a bounded replay/source delivery bundle."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PRIOR=ROOT.parent/'yb_diagnosis_repair_2026_10_08'
OLD=ROOT.parent/'rare_earth_yb_spectrum_2026_10_08'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    names=['test_mbpt_transitions.py','test_ci_mbpt_sigma1.py','test_review_regressions.py',
        'test_mbpt2.py','test_mbpt_sigma2.py','test_qed_self_energy_table.py',
        'test_qed_backend_auto.py','test_qed_indelicato_vs_fac.py','test_qed_orbital_shifts.py',
        'test_qed_approximation_coverage.py','test_velocity_gauge_full_order.py',
        'test_finite_frequency_electric.py']
    with zipfile.ZipFile(ROOT/'Affected_tests_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p=REPO/'tests'/name;z.write(p,str(p.relative_to(REPO)))
    patch=subprocess.run(['git','diff','--','src/atomx/methods/mbpt_transitions.py',
        'tests/test_mbpt_transitions.py'],cwd=REPO,capture_output=True,check=True,text=True).stdout
    (ROOT/'source_changes_from_HEAD.patch').write_text(patch)
    source=json.loads((ROOT/'final_source_sha256.json').read_text())
    current={name:sha(REPO/name) for name in source}
    if source!=current:raise ValueError('Final validated source changed before sealing')
    extra=[ROOT/'artifact_manifest.json',ROOT/'bundle_identity.json',PRIOR/'common.py',PRIOR/'README.md']
    for l in (4,6):
        extra += [PRIOR/f'results/l{l}_e50_all'/n for n in ('results.json','identity.json','contract.json','status.json')]
    extra += [OLD/'results/v68_default'/n for n in
        ('result.pkl','core.json','structure.json','status.json','orbitals_workspace.npz')]
    replay='''Yb energy-dependent self-energy replay bundle, 2026-10-08.

Extract this bundle into an independent directory. Extract docs/yb_energy_dependent_sigma_2026_10_08/Atomx_Dyson_final_source.zip and Affected_tests_source.zip into the same directory, preserving their src/ and tests/ paths. Use the repository virtual environment or install pyproject.toml dependencies in a new environment; do not replace numerical dependencies to render a PDF.

Summaries and captured baseline inputs are included. The exact large pole arrays are sealed in the local artifact manifest and remain in the original calculation directory. To recapture, retain the bundled scientific results and move the existing results/l6_e50_all and replay_and_ci folders aside before running run_dyson.py and analyze_and_inject.py in a new checkout. Drivers reject overwriting existing evidence. Preserve all original outputs. No experimental adjustment is used.

PDF/README describe numerical, spatial and experimental gates separately. Frozen-orbital scalar Dyson solving does not implement full Brueckner orbital response, self-consistent GF2, all-order screening or correlated radiative rates.
'''
    (ROOT/'REPLAY.txt').write_text(replay)
    exclusions={'artifact_manifest.json','bundle_identity.json','delivery_receipts.json',
        'Yb能量自洽_复现包.zip'}
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file()
        and '__pycache__' not in p.parts and p.name not in exclusions]
    manifest={str(p.relative_to(REPO)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'bundle_identity.json').write_text(json.dumps(dict(
        parent_source_archive_sha256=sha(PRIOR/'Atomx_final_source.zip'),
        measurement_source_archive_sha256=sha(ROOT/'Atomx_Dyson_source.zip'),
        current_HEAD=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip(),
        source_changed_during_capture=False, final_source_unchanged_at_sealing=True,
        final_source_archive_sha256=sha(ROOT/'Atomx_Dyson_final_source.zip'),
        source_changed_after_capture='Finite real scalar input guards; matched five-root replay and 137-test panel passed',
        full_local_manifest_sha256=sha(ROOT/'artifact_manifest.json'),
        delivery_pole_policy='Exact full NPZ arrays are retained locally and sealed in the manifest; the delivery bundle contains complete source, controls, baseline and summaries for recapture, without the large pole arrays.'),indent=2)+'\n')
    deliver=[p for p in files if p.suffix!='.npz']+extra+[ROOT/'REPLAY.txt']
    seen=set()
    bundle=ROOT/'Yb能量自洽_复现包.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in deliver:
            name=str(p.relative_to(REPO))
            if name not in seen:z.write(p,name);seen.add(name)
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('Bundle CRC failure')
        for p in deliver:
            if hashlib.sha256(z.read(str(p.relative_to(REPO)))).hexdigest()!=sha(p):
                raise ValueError('Bundle payload hash failure')
    print('SEALED',len(manifest),'local files; delivery',bundle.stat().st_size,'bytes',len(seen),'files')


if __name__=='__main__':main()
