"""Rebuild the compact results from retained, ignored local run outputs."""
import hashlib
import json
import pstats
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def read(name):
    return json.loads((ROOT/'raw'/name/'results.json').read_text())


def main():
    summary = dict(baseline_commit='94d7592',
        timing_scope='Atom.solve only, no profiling; first solve of each element in each process excluded',
        process_order='before-0, after-0, after-1, before-1; microbenchmarks separate',
        threads=dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1),
        elements=[], kernels=[], observables={})
    panels = {stage: [read(f'df-{stage}-{i}') for i in range(2)]
              for stage in ('before', 'after')}
    assert all(not p['profiled'] for group in panels.values() for p in group)
    assert all(p['settings'] == panels['before'][0]['settings']
               for group in panels.values() for p in group)
    for group in panels.values():
        assert all(p['source_sha256'] == group[0]['source_sha256'] for p in group)
    before_source = panels['before'][0]['source_sha256']
    after_source = panels['after'][0]['source_sha256']
    assert set(before_source) == set(after_source)
    assert {name for name in before_source if before_source[name] != after_source[name]} == {
        'numerics/shooting/operators.py', 'numerics/shooting/dirac_green.py'}
    for z in (6, 10, 54, 98):
        cases = {stage: [r for p in group for r in p['cases'] if r['z'] == z]
                 for stage, group in panels.items()}
        reference = cases['before'][0]
        assert all(r['converged'] for group in cases.values() for r in group)
        assert all(r['iterations'] == reference['iterations']
                   for group in cases.values() for r in group)
        assert all(r['result'] == reference['result']
                   for group in cases.values() for r in group)
        row = dict(z=z, iterations=reference['iterations'],
                   result=reference['result'], energy_Ha=reference['energy_Ha'])
        for stage, group in cases.items():
            samples = [r['seconds'] for r in group if r['repeat'] > 0]
            assert len(samples) == 6
            row[stage+'_seconds'] = samples
            row[stage+'_median_seconds'] = statistics.median(samples)
        row['reduction_percent'] = 100*(1-row['after_median_seconds']/row['before_median_seconds'])
        row['max_energy_change_Ha'] = max(abs(r['energy_Ha']-reference['energy_Ha'])
                                         for r in cases['after'])
        row['max_orbital_change_Ha'] = max(
            abs(r['orbital_energies_Ha'][k]-value) for r in cases['after']
            for k, value in reference['orbital_energies_Ha'].items())
        summary['elements'].append(row)
    before, after = read('kernels-before'), read('kernels-after')
    with np.load(ROOT/'raw/kernels-before/outputs.npz') as a, np.load(ROOT/'raw/kernels-after/outputs.npz') as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            t0 = statistics.median(before['measurements'][name]['seconds'])
            t1 = statistics.median(after['measurements'][name]['seconds'])
            summary['kernels'].append(dict(name=name, before_seconds=t0, after_seconds=t1,
                speedup=t0/t1, max_absolute_change=float(np.max(abs(a[name]-b[name])))))
    before, after = read('observables-before'), read('observables-after')
    with np.load(ROOT/'raw/observables-before/arrays.npz') as a, np.load(ROOT/'raw/observables-after/arrays.npz') as b:
        assert set(a.files) == set(b.files)
        summary['observables']['arrays'] = {
            name: dict(bit_identical=bool(np.array_equal(a[name], b[name])),
                       max_absolute_change=float(np.max(abs(a[name]-b[name])))) for name in a.files}
    summary['observables']['transitions_bit_identical'] = before['transitions'] == after['transitions']
    summary['observables']['transition_count'] = len(before['transitions'])
    summary['observables']['bound_records_bit_identical'] = all(
        value == after['measurements'][name] for name, value in before['measurements'].items()
        if name.startswith('bound_'))
    summary['observables']['phases_bit_identical'] = all(
        value['phase_rad'] == after['measurements'][name]['phase_rad']
        for name, value in before['measurements'].items() if name.startswith('continuum_'))
    summary['tests'] = {}
    for name in ('numerics', 'determinism'):
        suites = list(ET.parse(ROOT/'raw'/f'{name}.xml').getroot().iter('testsuite'))
        summary['tests'][name] = {field: sum(int(s.get(field, 0)) for s in suites)
                                 for field in ('tests', 'failures', 'errors', 'skipped')}
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    stats = pstats.Stats(str(ROOT/'raw/profile-before/Z98-1.prof'))
    rows = [dict(file=name, line=line, function=function, calls=total,
                 self_seconds=self_time, cumulative_seconds=cumulative)
            for (name, line, function), (_, total, self_time, cumulative, _) in stats.stats.items()]
    profile = dict(scope='Warm Cf solve; instrumentation overhead; use cost distribution, not latency',
                   total_calls=stats.total_calls, seconds=stats.total_tt,
                   rows=sorted(rows, key=lambda r: r['self_seconds'], reverse=True)[:30])
    (ROOT/'hotspots.json').write_text(json.dumps(profile, indent=2)+'\n')
    sources = [REPO/p for p in ('src/atomx/numerics/shooting/operators.py',
        'src/atomx/numerics/shooting/dirac_green.py', 'tests/test_shooting_radial_operators.py',
        'benchmarks/bench_df_solver.py', 'benchmarks/bench_shooting_radial_kernels.py',
        'benchmarks/bench_shooting_observables.py')]+[Path(__file__).resolve()]
    manifest = dict(baseline_commit='94d7592',
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
        raw_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT/'raw').rglob('*')) if p.is_file()})
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
