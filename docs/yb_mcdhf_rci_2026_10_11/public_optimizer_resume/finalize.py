"""Archive already-obtained interface proof; performs no solve or test rerun."""
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OWN / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


source = ROOT / 'src/atomx/numerics/bspline/orbital_optimization.py'
test = ROOT / 'tests/test_orbital_optimizer_resume.py'
assert sha(source) == '8216b92338a208414804b908662ff704413d8968437d50c359b14b7997988a8b'
assert sha(test) == 'e3a4cdb43f90d430d12d8a8f52229b14d725e18e192e57acb71a2107e7b83114'
(OWN / 'source_after').mkdir(exist_ok=True)
shutil.copy2(source, OWN / 'source_after/orbital_optimization.py')
manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src/atomx').rglob('*.py'))}
assert len(manifest) == 353
save('source_after_manifest.json', manifest)
before = json.loads((OWN / 'source_before_manifest.json').read_text())
changed = [p for p in sorted(manifest) if manifest[p] != before.get(p)]
assert changed == ['src/atomx/numerics/bspline/orbital_optimization.py']
patch = difflib.unified_diff((OWN / 'source_before/orbital_optimization.py').read_text().splitlines(True),
                             source.read_text().splitlines(True),
                             fromfile='source_before/orbital_optimization.py',
                             tofile='source_after/orbital_optimization.py')
(OWN / 'optimizer.patch').write_text(''.join(patch))
(OWN / 'focused.log').write_text('Captured from actual tool execution; not a rerun.\n'
    'Command: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest -q tests/test_orbital_optimizer_resume.py\n'
    '........................................                                 [100%]\n40 passed in 0.50s\n')
(OWN / 'ruff.log').write_text('Captured from actual tool execution; not a rerun.\n'
    'Command: .venv/bin/ruff check src/atomx/numerics/bspline/orbital_optimization.py tests/test_orbital_optimizer_resume.py\n'
    'All checks passed!\n')
save('focused_attempt001.json', {
    'record': 'actual earlier unfrozen test harness tool result',
    'passed': 34, 'failed': 3, 'seconds': .51,
    'failures': [
        {'test': 'test_original_metric_roundtrip_and_full_recomputation[True]',
         'actual_Ha': -1.0000130822888613, 'independent_Ha': -1.0000130822888615,
         'cause': 'test exact equality with independently strided matrix contraction'},
        {'test': 'test_accepted_resume_step_has_independent_rayleigh_gram_and_shared_snapshots',
         'actual_Ha': -1.0000130822888613, 'independent_Ha': -1.0000130822888615,
         'cause': 'same test exact equality control'},
        {'test': 'test_invalid_geometry_is_rejected_without_repair[negative_branch]',
         'cause': 'test requested nonexistent DiracResult.negative_coefficients attribute'}],
    'resolution': 'independent Rayleigh tolerance2e-15Ha and separate full scipy generalized eigensolve; no production gate change',
    'unfrozen_attempt_source_hash_unavailable': True})
save('documentation_impact.json', {
    'type': 'opt-in lower-level public numerical continuation, no new physics/backend',
    'search_terms': ['optimize_orbitals', 'preconditioner_factory', 'original metric', 'continuation', 'restart'],
    'dispositions': [
        {'path': 'src/atomx/numerics/bspline/orbital_optimization.py', 'action': 'update',
         'reason': 'document actual state/checkpoint/budget lifecycle; remove preexisting false automatic-refresh/last-step statements without adding such behavior'},
        {'path': str((OWN / 'README.md').relative_to(ROOT)), 'action': 'new_authoritative_topic',
         'reason': 'source-bound public interface, independent proof, scientific identity limits and actual obtained controls'},
        {'path': 'docs/ARCHITECTURE_CAPABILITIES.md', 'action': 'root_owned_update',
         'reason': 'root added public accepted-point resume capability row; no solver.solve dispatch claim'},
        {'path': 'docs/AGENT_REFERENCE.md', 'action': 'root_owned_registration',
         'reason': 'root integrates current navigation/evidence; outside this child ownership'},
        {'path': 'docs/yb_mcdhf_rci_2026_10_11/mixed_ci_stationarity_probe/PUBLIC_OPTIMIZATION_SUCCESSOR.md',
         'action': 'retain_source_epoch_evidence', 'reason': 'immutable earlier absent-interface/prospective algebra evidence; new README explicitly qualifies recorded-source claim'},
        {'path': 'docs/yb_mcdhf_rci_2026_10_11/native_run', 'action': 'retain',
         'reason': 'old source/input/failed and successful numerical evidence unchanged'},
        {'path': 'docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run', 'action': 'other_agent_owned_consumer',
         'reason': 'separate prospective real full531 continuation; no new solve here'},
        {'path': 'docs/HPC_NUMERICS_RESEARCH_2026-09-27.md', 'action': 'retain',
         'reason': 'earlier profiling evidence, no continuation API dependency or new timing claim'},
        {'path': 'docs/ARCHITECTURE_DESIGN.md', 'action': 'retain',
         'reason': 'numerics retains only numerical/std dependencies, no scientific policy/driver import'},
        {'path': 'docs/TESTING.md', 'action': 'retain',
         'reason': 'new local tests and single root affected panel follow existing testing guidance'}],
    'new_global_insight': False,
    'distilled_boundary': 'Reuse original metric only through validated immutable state; fresh endpoint initialization is a different residual yardstick. New objective identities require consumer physical binding and full reevaluation, never inherited certification.',
    'old_scientific_evidence_modified': False})
root_regression = ROOT / 'docs/yb_mcdhf_rci_2026_10_11/public_resume_integration'
root_check = json.loads((root_regression / 'verification.json').read_text())
assert root_check['passed'] and root_check['returncode'] == 0
root_log = (root_regression / 'pytest.log').read_text()
assert re.search(r'115 passed', root_log)
save('result.json', {
    'status': 'implemented_public_interface_independently_tested',
    'interface': 'OrbitalOptimizationState.from_arrays + optimize_orbitals accepted-point resume',
    'source_sha256': sha(source), 'test_sha256': sha(test),
    'source_python_count': 353, 'source_changes_from_contract': changed,
    'focused': {'passed': 40, 'seconds_uncontrolled': .50, 'independent_model_ns': 35,
                'OPENBLAS_NUM_THREADS': 1, 'OMP_NUM_THREADS': 1,
                'CPU_affinity_enforced': False, 'address_space_cap_enforced': False,
                'observed_peak_memory': None},
    'independent_anchors': ['analytic independent-particle Rayleigh energy and gradient',
        'saved diagonal/full-SPD inverse action and projected residual',
        'accepted-point actual radial S-Gram', 'new objective explicit full CI/gradient reevaluation',
        'old/fresh 12-step trajectory and all diagnostics bitwise equality'],
    'root_separate_regressions': {'passed': 115, 'pytest_seconds_uncontrolled': 95.18,
                                'CPU_affinity': 1, 'numerical_threads': 1, 'timeout_s': 300,
                                'memory_cap_enforced': False, 'source_tests_unchanged': True,
                                'verification_sha256': sha(root_regression / 'verification.json')},
    'remaining_real_Yb_resume_run': True,
    'full_mixed_CI_energy_difference_implemented': False,
    'new_Yb_CI_or_gradient_evaluations': 0,
    'expanded_model_stationarity_certified': False,
    'experimental_or_highest_level_certified': False,
    'performance_claim': False})
checks = []


def check(name, success):
    checks.append({'name': name, 'passed': bool(success)})
    assert success, name


contract = json.loads((OWN / 'contract.json').read_text())
for relative, expected in contract['inputs_sha256'].items():
    check('unchanged_contract_input:' + relative, sha(ROOT / relative) == expected)
check('before_manifest_identity', sha(OWN / 'source_before_manifest.json') == contract['source_before_manifest_sha256'])
check('before_optimizer_identity', sha(OWN / 'source_before/orbital_optimization.py') == contract['source_before_optimizer_sha256'])
for relative, expected in manifest.items():
    check('current_source_identity:' + relative, sha(ROOT / relative) == expected)
check('only_owned_source_changed', changed == ['src/atomx/numerics/bspline/orbital_optimization.py'])
check('frozen_new_test_identity', sha(test) == 'e3a4cdb43f90d430d12d8a8f52229b14d725e18e192e57acb71a2107e7b83114')
check('root_tests_passed', root_check['passed'])
check('root_source_unchanged', root_check['source_unchanged_during_run'])
check('root_test_inputs_unchanged', root_check['original_and_new_test_inputs_unchanged'])
readme = (OWN / 'README.md').read_text()
save('verification.json', {'status': 'documentation_verification_in_progress'})
for _, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', readme):
    check('readme_link:' + target, (OWN / target.split('#')[0]).exists())
artifacts = {p.name: sha(p) for p in sorted(OWN.iterdir())
             if p.is_file() and p.name not in ('verification.json', 'trace.md')}
save('verification.json', {'all_passed': True, 'checks_count': len(checks), 'checks': checks,
                          'artifacts_sha256': artifacts, 'source_sha256': sha(source), 'test_sha256': sha(test),
                          'focused_test_controls_obtained': 40, 'root_regression_controls_obtained': 115,
                          'not_a_new_independent_numerical_replay': True,
                          'document_and_hash_verification_only': True,
                          'experimental_or_full_CV_certification': False})
(OWN / 'trace.md').write_text('EARS: validated immutable original-metric continuation\n\n'
    'Expectation: import exact ordered P/x/inverses with physical geometry binding, explicit new objective, no inherited CI/history/certification; preserve old default.\n'
    'Action: implement owned public numerical API and byte-sealed geometry, framing/positive-space/Gram guards, shared accepted-point snapshots and safe budget return; no full531 solve or mixed energy kernel.\n'
    'Results: 40 independent/negative focused controls passed0.50s; Ruff passed; root separate affected115 controls passed95.18s with unchanged source/tests. Three initial harness failures retained. No controlled timing or peakmemory proof.\n'
    'Significance: original residual yardstick can now be restored explicitly; scientific objective string remains caller declaration. Preserve old successful/failed source epochs and require real successor contract plus independent gradient to assess expanded-model stationarity.\n')
print(json.dumps({'checks': len(checks), 'source': sha(source), 'test': sha(test),
                  'result': sha(OWN / 'result.json'), 'verification': sha(OWN / 'verification.json'),
                  'README': sha(OWN / 'README.md')}, indent=2))
