"""Combine immutable measurements, complete arrays and final replay proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import tarfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as inp:
        for block in iter(lambda:inp.read(1024**2),b''):
            result.update(block)
    return result.hexdigest()

raw=json.loads((HERE/'result.json').read_text())
corrected=json.loads((HERE/'accurate_scaling_result.json').read_text())
pre=json.loads((HERE/'pre_source_guard.json').read_text())
source_mismatches={name:dict(before=expected,after=digest(Path(name))) for name,expected in pre['files'].items() if digest(Path(name))!=expected}
angular=json.loads((HERE/'angular_inputs_identity.json').read_text())
angular_mismatches={name:expected['sha256'] for name,expected in angular['files'].items() if digest(Path(name))!=expected['sha256'] or digest(Path(expected['copied']))!=expected['sha256']}
identity=json.loads((HERE/'execution_identity.json').read_text())
runtime_mismatches={name:expected['sha256'] for name,expected in identity['actual_mapped_runtime'].items() if digest(Path(name))!=expected['sha256']}
assert not source_mismatches and not angular_mismatches and not runtime_mismatches
post=dict(source_files_checked=len(pre['files']),source_mismatches=source_mismatches,
          angular_files_checked=len(angular['files']),angular_mismatches=angular_mismatches,
          actual_runtime_files_checked=len(identity['actual_mapped_runtime']),runtime_mismatches=runtime_mismatches,
          strict_all_pass=True)
(HERE/'post_identity_guard.json').write_text(json.dumps(post,indent=2)+'\n')
with tarfile.open(HERE/'timing_refinement_sources.tar.gz','w:gz') as archive:
    for name in ('accurate_scaling.py','child_timer.cpp','child_timer','monitor_threads.py','test_benchmark.py','finalize.py'):
        archive.add(HERE/name,arcname=name,recursive=False)
    for path in (HERE/'angular_inputs').rglob('*'):
        if path.is_file():archive.add(path,arcname=str(path.relative_to(HERE)),recursive=False)
matched_pairs=[]
errors=[]
for pair in raw['matched']['pairs']:
    r=pair['round']; folder=HERE/f'matched_v002_round{r:02d}'
    arrays=np.load(folder/'complete_comparison.npz')
    atomx,ambit=arrays['atomx'],arrays['ambit']
    assert atomx.shape==ambit.shape==(27,6)
    assert np.array_equal(atomx[:,:3],ambit[:,:3])
    assert np.allclose(atomx[:,3:],ambit[:,3:],atol=1e-11,rtol=1e-10)
    assert (folder/'ambit.stderr').read_text().strip()==''
    errors.append(float(np.max(np.abs(atomx[:,3:]-ambit[:,3:]))))
    matched_pairs.append(dict(round=r,order=pair['order'],atomx_seconds=pair['atomx_seconds'],ambit_seconds=pair['ambit_seconds'],
                              workflow_ratio=pair['atomx_seconds']/pair['ambit_seconds'],scientific_gate=pair['scientific_gate'],
                              controlled_timing=pair['controlled_timing'],guards={name:value['guard'] for name,value in pair['executions'].items()},
                              peak_rss_bytes={name:value['peak_rss_bytes'] for name,value in pair['executions'].items()},
                              complete_array_file=str((folder/'complete_comparison.npz').relative_to(HERE)),
                              complete_array_sha256=digest(folder/'complete_comparison.npz')))
scaling={}
for workers,entry in corrected['results'].items():
    p=int(workers); pairs=[]
    for pair in entry['pairs']:
        samples={}
        for worker in (1,p):
            sample=pair['child_timings'][str(worker)]
            record=pair['executions'][str(worker)]
            folder=Path(record['cwd'])
            reported=int(re.search(r'AMBiT running with (\d+) OpenMP threads', (folder/'log.out').read_text()).group(1))
            assert reported==worker
            samples[str(worker)]=dict(child_wall_seconds=sample['seconds'],user_seconds=sample['user_seconds'],system_seconds=sample['system_seconds'],
                                      average_cpu_cores=(sample['user_seconds']+sample['system_seconds'])/sample['seconds'],peak_rss_bytes=sample['max_rss_bytes'],pid=sample['pid'],
                                      actual_threads=reported,actual_thread_scope='program omp_get_max_threads() with OMP_DYNAMIC=FALSE; OS active-thread samples unavailable',
                                      guard=record['guard'],directory=str(folder.relative_to(HERE)))
        s=pair['serial_seconds']/pair['parallel_seconds']
        passed=pair['scientific_gate'] and pair['controlled_timing']
        pairs.append(dict(round=pair['round'],order=pair['order'],samples=samples,
                          scientific_gate=pair['scientific_gate'],controlled_timing=pair['controlled_timing'],
                          speedup=s if passed else None,parallel_efficiency=s/p if passed else None,
                          observational_speedup=s,full_output_sha256=pair['complete_output_sha256']))
    assert len(pairs)==5
    scaling[workers]=dict(statistics=entry['statistics'],pairs=pairs,
                         median_peak_rss_bytes={str(w):statistics.median(pair['samples'][str(w)]['peak_rss_bytes'] for pair in pairs) for w in (1,p)},
                         median_user_seconds={str(w):statistics.median(pair['samples'][str(w)]['user_seconds'] for pair in pairs) for w in (1,p)},
                         median_system_seconds={str(w):statistics.median(pair['samples'][str(w)]['system_seconds'] for pair in pairs) for w in (1,p)})
checks=ET.parse(HERE/'final_test_checks.xml').getroot()
suites=list(checks.iter('testsuite'))
tests={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
assert tests==dict(tests=4,failures=0,errors=0,skipped=0)
report=dict(created_at=datetime.now(timezone.utc).isoformat(),status='complete bounded matched comparison and native scaling',
            matched=dict(orbitals=19,integrals=37350,signed_matrix_entries=27,components=3,
                         max_error_Ha=max(errors),precision_absolute_gate_Ha=1e-11,precision_relative_gate=1e-10,all_precision_gates_pass=True,
                         no_ambit_skipped_diagram_warning=True,same_Hamiltonian=True,independent_native_diagram_implementation=True,
                         shared_atomx_radial_inputs=True,experimental_accuracy_claim=False,
                         all_environment_gates_pass=all(pair['controlled_timing'] for pair in matched_pairs),
                         statistics=raw['matched']['statistics'],pairs=matched_pairs,
                         timing_scope=raw['matched']['timing_scope']),
            ambit_native_scaling=dict(corrected_child_wall=True,threads=[1,2,4,8],per_parallel_thread_pairs=5,
                                      fixed_radial_points=4000,all_full_signed_files_bitwise_equal=True,
                                      independent_atomx_matched=False,full_output_golden_sha256=corrected['reference_golden_complete_output_sha256'],
                                      input_sha256=corrected['input_sha256'],results=scaling,
                                      timing_scope=corrected['scopes']),
            identity=post,tests=dict(tests,xml='final_test_checks.xml'),
            evidence=dict(original_timeout_wait_observations='result.json',measurement_amendment='timer_amendment.json',
                          corrected_native_raw_records='accurate_scaling_result.json',actual_source_object_identity='execution_identity.json',
                          mapped_runtime_identity='execution_identity.json',pre_sources='pre_source_guard.json',post_sources='post_identity_guard.json',
                          original_source_object_archive='actual_sources_objects.tar.gz',original_archive_sha256=digest(HERE/'actual_sources_objects.tar.gz'),
                          refinement_source_archive='timing_refinement_sources.tar.gz',refinement_archive_sha256=digest(HERE/'timing_refinement_sources.tar.gz'),
                          complete_artifact_manifest='artifact_manifest.json'),
            full_runtime_copied=False,highest_level_reached=False,
            limitations=['Finite19orbitalSigma panel does not independently validate radial physics/continuum convergence/experimental spectra.',
                         'Fixed-energy workflows differ in reusable energy-dependence and binding work; warm scope ratio is not a whole-software ranking.',
                         'Six-level BeII is small and uses separate native AMBiT physics; no matched Atomx BeII precision ranking exists.',
                         'OpenMP thread maxima are confirmed by actual program log; active OS worker occupancy was not sampled.',
                         'Shared-machine load gates pass but no exclusive-core/frequency isolation is claimed.',
                         'Native caches/angular data are warm; each scientific workspace is fresh; original and corrected entire observations retained.'])
(HERE/'result.corrected.json').write_text(json.dumps(report,indent=2)+'\n')
files=sorted(path for path in HERE.rglob('*') if path.is_file() and '__pycache__' not in str(path) and path.name!='artifact_manifest.json')
manifest={str(path.relative_to(HERE)):dict(sha256=digest(path),bytes=path.stat().st_size) for path in files}
(HERE/'artifact_manifest.json').write_text(json.dumps(dict(file_count=len(files),files=manifest),indent=2)+'\n')
print(json.dumps(dict(final='result.corrected.json',precision=report['matched']['max_error_Ha'],matched=report['matched']['statistics'],
                      scaling={key:value['statistics'] for key,value in scaling.items()},identity=post,tests=tests,files=len(files)),indent=2))
