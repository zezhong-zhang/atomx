"""Audit before/after observables, live RADIAL checks and test coverage."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT/relative).read_text())


def main():
    report = dict(baseline_commit='eb7bdae',
        scope='Analytic exponential moments; interpolation, physics and tolerances unchanged.',
        timing_scope='Warm single-thread native wall time; four fresh DF processes in '
                     'before/after/after/before order, six warm samples per element. '
                     'First solves excluded. Observable kernel timing is one process '
                     'per implementation; live RADIAL timings are not used for speed claims.',
        df={}, arrays={}, kernels={})
    for z in (1, 6, 10, 98):
        stages = {}
        for stage in ('before', 'after'):
            stages[stage] = [case for repeat in (0, 1)
                for case in read(f'df-{stage}-{repeat}/results.json')['cases']
                if case['z'] == z]
        before, after = stages['before'][0], stages['after'][0]
        for cases in stages.values():
            assert all(case['converged'] for case in cases)
            assert all(case['iterations'] == before['iterations'] for case in cases)
            assert all(case['energy_Ha'] == cases[0]['energy_Ha']
                       and case['orbital_energies_Ha'] == cases[0]['orbital_energies_Ha']
                       for case in cases)
        medians = {stage: float(np.median([case['seconds'] for case in cases
                                          if case['repeat'] > 0]))
                   for stage, cases in stages.items()}
        energy_difference = abs(after['energy_Ha']-before['energy_Ha'])
        orbital_difference = max(abs(after['orbital_energies_Ha'][key]-value)
                                 for key, value in before['orbital_energies_Ha'].items())
        assert energy_difference < 1e-9 and orbital_difference < 1e-9
        report['df'][str(z)] = dict(medians_s=medians,
            reduction=1-medians['after']/medians['before'],
            max_total_energy_difference_Ha=energy_difference,
            max_orbital_energy_difference_Ha=orbital_difference,
            iterations=before['iterations'], all_converged=True)
    for case in read('df-after-sbound/results.json')['cases']:
        previous = next(row for row in read('df-after-0/results.json')['cases']
                        if row['z'] == case['z'])
        assert all(case[key] == previous[key] for key in
                   ('energy_Ha', 'orbital_energies_Ha', 'iterations', 'converged'))
    report['df_after_schrodinger_fix'] = 'energies, orbitals, iterations unchanged'

    before = read('observables-before-supported/results.json')
    after = read('observables-after/results.json')
    assert before['settings'] == after['settings']
    report['observable_settings'] = after['settings']
    with (np.load(ROOT/'observables-before-supported/arrays.npz') as old,
          np.load(ROOT/'observables-after/arrays.npz') as new):
        assert set(old.files) == set(new.files)
        for key in old.files:
            np.testing.assert_allclose(old[key], new[key], rtol=2e-10, atol=2e-13)
            report['arrays'][key] = dict(
                max_absolute_difference=float(np.max(abs(new[key]-old[key]))),
                identical=bool(np.array_equal(old[key], new[key])))
    assert len(before['transitions']) == len(after['transitions'])
    line_difference = 0.
    for old, new in zip(before['transitions'], after['transitions'], strict=True):
        for key in ('initial_label', 'final_label', 'multipole', 'gauge'):
            assert old[key] == new[key]
        for key in ('energy_diff_hartree', 'line_strength', 'A_rate_per_s',
                    'A_length', 'A_velocity'):
            if old[key] is None:
                assert new[key] is None
            else:
                np.testing.assert_allclose(new[key], old[key], rtol=2e-9, atol=1e-13)
                if old[key]:
                    line_difference = max(line_difference, abs(new[key]/old[key]-1))
    report['transitions'] = dict(count=len(after['transitions']),
                                max_relative_difference=line_difference)
    for name in ('prefix_slope_0.0', 'prefix_slope_0.1', 'prefix_slope_2.0',
                 'prefix_slope_8.0', 'prefix_slope_20.0', 'prefix_slope_100.0', 'response'):
        medians = {stage: float(np.median(data['measurements'][name]['seconds']))
                   for stage, data in [('before', before), ('after', after)]}
        report['kernels'][name] = dict(medians_s=medians,
            speedup=medians['before']/medians['after'])
    bound = read('bound-radial/results.json')['cases']
    continuum = read('continuum-radial/results.json')['cases']
    assert len(bound) == 27 and len(continuum) == 36
    report['live_RADIAL'] = dict(bound_cases=len(bound), continuum_cases=len(continuum),
        max_energy_difference_Ha=max(abs(row['energy_difference_Ha']) for row in bound),
        max_bound_wave_error_over_peak=max(row['relative_wave_error'] for row in bound),
        max_continuum_wave_error_over_asymptotic_amplitude=max(row['amplitude_error']
                                                             for row in continuum),
        max_phase_difference_rad_mod_pi=max(row['phase_error'] for row in continuum))
    tree = ET.parse(ROOT/'full-tests.xml')
    cases = tree.findall('.//testcase')
    def outcome(case):
        for status in ('failure', 'error', 'skipped'):
            if case.find(status) is not None:
                if status == 'skipped' and case.find(status).get('type') == 'pytest.xfail':
                    return 'xfailed'
                return status
        return 'passed'
    baseline = {}
    for name in ('baseline-failures-final.xml', 'baseline-additional.xml',
                 'baseline-snapshots-properties.xml', 'baseline-reactions.xml',
                 'baseline-late.xml', 'baseline-last.xml'):
        for case in ET.parse(ROOT/name).findall('.//testcase'):
            baseline[(case.attrib['classname'], case.attrib['name'])] = outcome(case)
    failures = [case for case in cases if outcome(case) in ('failure', 'error')]
    current = ET.parse(ROOT/'current-numerics.xml').findall('.//testcase')
    safety = ET.parse(ROOT/'current-determinism-full.xml').findall('.//testcase')
    current += safety
    current_outcomes = {(case.attrib['classname'], case.attrib['name']): outcome(case)
                        for case in current}
    baseline_audit = []
    for case in failures:
        identity = (case.attrib['classname'], case.attrib['name'])
        baseline_audit.append(dict(test='::'.join(identity), full_audit=outcome(case),
            baseline=baseline.get(identity, 'not yet reproduced'),
            final_focused_status=current_outcomes.get(identity, 'not rerun')))
    assert all(row['baseline'] in ('failure', 'error') for row in baseline_audit)
    report['tests'] = dict(cases=len(cases),
        scope='Full audit after moments optimization, before Schrodinger repair; '
              'final focused suite verifies the subsequent repair.',
        passed=sum(outcome(case) == 'passed' for case in cases),
        failed=sum(case.find('failure') is not None for case in cases),
        errors=sum(case.find('error') is not None for case in cases),
        skipped=sum(outcome(case) == 'skipped' for case in cases),
        expected_failures=sum(outcome(case) == 'xfailed' for case in cases),
        baseline_audit=baseline_audit)
    assert all(outcome(case) == 'passed' for case in current)
    report['final_tests'] = dict(passed=len(current), numerics=len(current)-len(safety),
        reproducibility_and_bounds=len(safety),
        fixed_full_audit_failures=sum(row['final_focused_status'] == 'passed'
                                     for row in baseline_audit))
    schrodinger = read('sbound-diagnostic/after.json')['cases']
    assert len(schrodinger) == 12
    assert all(row['native_after_ier'] == row['radial_ier'] == 0 for row in schrodinger)
    report['schrodinger'] = dict(cases=len(schrodinger),
        failed_before=sum(row['native_ier'] != 0 for row in schrodinger), failed_after=0,
        max_energy_difference_Ha=max(abs(row['native_after_energy_Ha']-row['radial_energy_Ha'])
                                     for row in schrodinger),
        max_wave_difference=max(max(row['max_p_difference'], row['max_derivative_difference'])
                                for row in schrodinger),
        scope='Coulomb p/d roots at varied guesses, matching live RADIAL; '
              's-wave derivative/normalization separately verified analytically')
    (ROOT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in ROOT.rglob('*') if path.is_file() and path.name != 'manifest.json'}
    source_files = [Path('src/atomx/numerics/shooting')/name for name in
                   ('exponential_quadrature.py', 'schrodinger_bound.py', 'local_scf.py')]
    source_files += [Path('tests/test_exponential_quadrature.py'),
                    Path('tests/test_schrodinger_bound_shooting.py'),
                    Path('tests/test_determinism.py'),
                    Path('.github/workflows/test.yml'),
                    Path('.github/workflows/nightly.yml'),
                    Path('benchmarks/bench_shooting_observables.py')]
    manifest = dict(artifacts_sha256=manifest,
        final_source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in source_files})
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
