"""Rebuild compact evidence from the retained, ignored raw run outputs."""
import hashlib
import json
import pstats
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
CHANGED = {'numerics/shooting/energy.py', 'numerics/shooting/hartree_fock.py',
           'physics/hartree_fock.py'}


def read(name):
    return json.loads((ROOT/'raw'/name/'results.json').read_text())


def identical_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    summary = dict(baseline_commit='af6128b',
        timing_scope='Atom.solve only; first solve of each element/process excluded',
        process_order='before-0, after-0, after-1, before-1; kernels separate',
        threads=dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1),
        elements=[], kernels=[], observables={})
    panels = {stage: [read(f'df-{stage}-{i}') for i in range(2)]
              for stage in ('before', 'after')}
    reference = panels['before'][0]
    for stage, group in panels.items():
        assert all(not report['profiled'] for report in group)
        assert all(report['settings'] == reference['settings'] for report in group)
        assert all(report['source_sha256'] == group[0]['source_sha256'] for report in group)
        if stage == 'after':
            assert {name for name in reference['source_sha256']
                    if reference['source_sha256'][name] != group[0]['source_sha256'][name]} == CHANGED
    for z in (6, 10, 54, 98):
        cases = {stage: [r for report in group for r in report['cases'] if r['z'] == z]
                 for stage, group in panels.items()}
        control = cases['before'][0]
        assert all(row['converged'] and row['iterations'] == control['iterations']
                   and row['result'] == control['result']
                   and row['energy_Ha'] == control['energy_Ha']
                   and row['orbital_energies_Ha'] == control['orbital_energies_Ha']
                   for group in cases.values() for row in group)
        row = dict(z=z, iterations=control['iterations'], result=control['result'],
                   energy_Ha=control['energy_Ha'], energies_bit_identical=True)
        for stage, group in cases.items():
            samples = [r['seconds'] for r in group if r['repeat'] > 0]
            assert len(samples) == 6
            row[stage+'_seconds'] = samples
            row[stage+'_median_seconds'] = statistics.median(samples)
        row['reduction_percent'] = 100*(1-row['after_median_seconds']/row['before_median_seconds'])
        summary['elements'].append(row)

    before, after = read('kernels-before'), read('kernels-after')
    assert before['cases'] == after['cases'] and before['settings'] == after['settings']
    summary['kernel_cases'] = before['cases']
    assert before['source_sha256'] == reference['source_sha256']
    assert after['source_sha256'] == panels['after'][0]['source_sha256']
    for name, value in before['measurements'].items():
        t0 = statistics.median(value['seconds'])
        t1 = statistics.median(after['measurements'][name]['seconds'])
        summary['kernels'].append(dict(name=name, before_seconds=value['seconds'],
            after_seconds=after['measurements'][name]['seconds'],
            before_median_seconds=t0, after_median_seconds=t1, speedup=t0/t1))
    with np.load(ROOT/'raw/kernels-before/outputs.npz') as a, np.load(ROOT/'raw/kernels-after/outputs.npz') as b:
        assert set(a.files) == set(b.files)
        assert all(identical_bits(a[name], b[name]) for name in a.files)
        summary['rotation_probes'] = dict(bit_identical=True, pairs=len(a.files),
                                          angles=sum(a[name].size for name in a.files))

    before, after = read('observables-before'), read('observables-after')
    assert before['settings'] == after['settings']
    assert before['source_sha256'] == reference['source_sha256']
    assert after['source_sha256'] == panels['after'][0]['source_sha256']
    with np.load(ROOT/'raw/observables-before/arrays.npz') as a, np.load(ROOT/'raw/observables-after/arrays.npz') as b:
        assert set(a.files) == set(b.files)
        summary['observables']['arrays'] = {
            name: dict(bit_identical=identical_bits(a[name], b[name]),
                       max_absolute_change=float(np.max(abs(a[name]-b[name])))) for name in a.files}
        assert all(row['bit_identical'] for row in summary['observables']['arrays'].values())
    summary['observables']['transitions_bit_identical'] = before['transitions'] == after['transitions']
    summary['observables']['transition_count'] = len(before['transitions'])
    summary['observables']['bound_records_bit_identical'] = all(
        value == after['measurements'][name] for name, value in before['measurements'].items()
        if name.startswith('bound_'))
    summary['observables']['phases_bit_identical'] = all(
        value['phase_rad'] == after['measurements'][name]['phase_rad']
        for name, value in before['measurements'].items() if name.startswith('continuum_'))
    assert all(summary['observables'][name] for name in (
        'transitions_bit_identical', 'bound_records_bit_identical', 'phases_bit_identical'))
    summary['tests'] = {}
    for name in ('numerics', 'determinism'):
        suites = list(ET.parse(ROOT/'raw'/f'{name}.xml').getroot().iter('testsuite'))
        summary['tests'][name] = {field: sum(int(s.get(field, 0)) for s in suites)
                                 for field in ('tests', 'failures', 'errors', 'skipped')}
        assert summary['tests'][name]['failures'] == summary['tests'][name]['errors'] == 0
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    hotspots = {}
    for stage in ('before', 'after'):
        stats = pstats.Stats(str(ROOT/'raw'/f'profile-{stage}/Z98-1.prof'))
        rows = [dict(file=name, line=line, function=function, calls=total,
                     self_seconds=self_time, cumulative_seconds=cumulative)
                for (name, line, function), (_, total, self_time, cumulative, _) in stats.stats.items()]
        hotspots[stage] = dict(scope='warm Cf; instrumentation overhead, not a latency claim',
            total_calls=stats.total_calls, seconds=stats.total_tt,
            rows=sorted(rows, key=lambda row: row['self_seconds'], reverse=True)[:30])
    (ROOT/'hotspots.json').write_text(json.dumps(hotspots, indent=2)+'\n')
    files = [REPO/'src/atomx'/name for name in sorted(CHANGED)] + [
        REPO/name for name in ('tests/test_shooting_rotation_energy.py',
            'tests/test_direct_hamiltonian_df.py', 'benchmarks/bench_df_solver.py',
            'benchmarks/bench_shooting_rotation_energy.py', 'benchmarks/bench_shooting_observables.py')]
    files.append(Path(__file__).resolve())
    manifest = dict(baseline_commit='af6128b',
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in files},
        raw_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT/'raw').rglob('*')) if p.is_file()})
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
