"""Audit retained measurements and record their limited comparison scope."""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bound_before=load(root/'bound-before/results.json')['cases']
bound_after=load(root/'bound-after-live/results.json')['cases']
assert len(bound_before)==len(bound_after)==27
for b,a in zip(bound_before,bound_after,strict=True):
    assert (a['z'],a['n'],a['kappa'])==(b['z'],b['n'],b['kappa'])
    assert a['energy_Ha']==b['energy_Ha'] and a['wave_sha256']==b['wave_sha256']
report=dict(baseline_commit='0a1685e',
    scope='Local screened/point DBOUND and DFREE fields; default Hamiltonian DF for Ne/Cf. Not full-table certification.',
    bound=dict(cases=27,energies_and_wave_hashes_identical=True,
        native_before_median_s=float(np.median([c['median_native_s'] for c in bound_before])),
        native_after_median_s=float(np.median([c['median_native_s'] for c in bound_after])),
        radial_cpu_median_s=float(np.median([c['radial_cpu_s'] for c in bound_after])),
        summed_case_time_reduction=1-sum(c['median_native_s'] for c in bound_after)/sum(c['median_native_s'] for c in bound_before),
        max_abs_energy_difference_Ha=max(abs(c['energy_difference_Ha']) for c in bound_after),
        max_wave_error_over_peak=max(c['relative_wave_error'] for c in bound_after)),df={})
for z in (10,98):
    times={v:[] for v in ('before','after')}
    identity=None
    for v in times:
        for p in sorted(root.glob(f'df-final-*-{v}/results.json')):
            for c in load(p)['cases']:
                if c['z']!=z:continue
                observed=(c['energy_Ha'],c['orbital_energies_Ha'],c['iterations'],c['converged'])
                if identity is None:identity=observed
                assert observed==identity and c['converged']
                if c['repeat']>0:times[v].append(c['seconds'])
    medians={v:float(np.median(t)) for v,t in times.items()}
    report['df'][str(z)]=dict(warm_times_s=times,medians_s=medians,
        reduction=1-medians['after']/medians['before'],
        energies_orbitals_iterations_identical=True,iterations=identity[2])
continuum_before=load(root/'continuum-before-r200/results.json')['cases']
continuum_after=load(root/'continuum-after-r200/results.json')['cases']
assert len(continuum_before)==len(continuum_after)==36
for b,a in zip(continuum_before,continuum_after,strict=True):
    assert all(a[k]==b[k] for k in ('z','energy','kappa','native_phase','amplitude_error','phase_error'))
report['continuum']=dict(cases=36,phase_and_errors_identical=True,
    max_wave_error_over_asymptotic_amplitude=max(c['amplitude_error'] for c in continuum_after),
    max_phase_error_mod_pi=max(c['phase_error'] for c in continuum_after),
    median_before_s=float(np.median([c['median_native_seconds'] for c in continuum_before])),
    median_after_s=float(np.median([c['median_native_seconds'] for c in continuum_after])),
    summed_case_time_reduction=1-sum(c['median_native_seconds'] for c in continuum_after)/sum(c['median_native_seconds'] for c in continuum_before))
report['timing_limits']='Warm native wall time; Fortran CPU time inside DBOUND/DFREE, process startup excluded. Fields and constants match. First calls and profiling excluded from DF final timings. Some other diagnostic processes ran concurrently; these are workload measurements.'
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
files=[Path('src/atomx/numerics/shooting')/p for p in ['dirac_power_series.py','dirac_integrator.py','dirac_green.py']]
files += [Path('src/atomx/corrections/breit.py'),Path('benchmarks/bench_radial_bound_kernels.py'),Path('benchmarks/reference_probes/radial_bound.f90'),Path('tests/test_dirac_dense_output.py')]
manifest=dict(final_source_sha256={str(p):sha(p) for p in files},
    artifact_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'})
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(report,indent=2))
