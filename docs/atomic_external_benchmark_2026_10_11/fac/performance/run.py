"""Bounded FAC/Atomx complete Ne-like frozen-configuration workflow scaling."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tarfile
import threading
import time

import numpy as np

from benchmarks.bench_reference_ci_transitions import fac_script, reference
from benchmarks.fac_runner import parse_hamiltonian_binary, parse_levels

ROOT = Path(__file__).resolve().parents[4]
OWNER = Path(__file__).resolve().parent
FAC = Path('/home/zzhang/code/fac').resolve()
SFAC = FAC/'sfac/sfac'
LIB = '/data/code/atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu'
ZVALUES = (18, 26, 54, 82)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


@contextmanager
def timing_lock():
    with (OWNER.parents[1]/'timing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def source_snapshot():
    paths = set((ROOT/'src/atomx').rglob('*.py'))
    paths.update((ROOT/'benchmarks').glob('*.py'))
    for suffix in ('*.c','*.h','*.f','*.F','*.o','*.a','Makefile*'):
        paths.update(FAC.rglob(suffix))
    paths.update((FAC/'config.log', SFAC, ROOT/'pyproject.toml'))
    paths.update(OWNER.glob('*.py')); paths.add(OWNER/'child_timer.cpp')
    paths.add(OWNER/'child_timer')
    return {str(p.resolve()): sha(p) for p in sorted(paths) if p.is_file()}


def archive_sources(manifest):
    with tarfile.open(OWNER/'actual_sources_build.tar.gz','w:gz') as archive:
        for name in manifest:
            archive.add(name, arcname=name.lstrip('/'), recursive=False)


def environment(workers):
    env = os.environ.copy()
    # Preserve channel binding; only scoped numerical thread/library variables.
    env.update(OMP_NUM_THREADS=str(workers), OMP_THREAD_LIMIT=str(workers),
               OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='false',
               OPENBLAS_NUM_THREADS=str(workers), MKL_NUM_THREADS=str(workers),
               BLIS_NUM_THREADS=str(workers), NUMBA_NUM_THREADS='1',
               PYTHONPATH=str(ROOT/'src')+':'+str(ROOT),
               LD_LIBRARY_PATH=LIB+':'+env.get('LD_LIBRARY_PATH',''))
    return env


def observe(pidfile, stop, samples, mapped, violations, budget, started):
    pid = None
    while not stop.wait(.005):
        if pid is None:
            try:
                pid = int(pidfile.read_text())
            except (OSError, ValueError):
                continue
        folder = Path('/proc')/str(pid)
        try:
            tasks = list((folder/'task').iterdir())
            running = 0; ticks = 0
            for task in tasks:
                try:
                    parts = (task/'stat').read_text().split(') ',1)[1].split()
                    running += parts[0] == 'R'
                    ticks += int(parts[11])+int(parts[12])
                except (OSError, ValueError):
                    pass
            samples.append({'elapsed': time.monotonic()-started,
                            'threads': len(tasks), 'running': running, 'cpu_ticks': ticks})
            for line in (folder/'maps').read_text().splitlines():
                items = line.split()
                if len(items)>5 and items[-1].startswith('/'):
                    name = items[-1]
                    if name not in mapped and Path(name).is_file():
                        mapped[name] = sha(name)
            reason = None
            if len(tasks)>budget:
                reason = f'actual_threads={len(tasks)} exceeds total cap={budget}'
            if time.monotonic()-started>600:
                reason = 'actual child wall exceeds 600 seconds'
            if reason and not violations:
                violations.append({'reason':reason,'elapsed':time.monotonic()-started,'pid':pid})
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def prepare_fac(z, folder, workers):
    original = fac_script(z,8000,4.)
    lines = [f'InitializeMPI({workers})', 'MPIRank()', "WallTime('initialized',0)",
             'LimitArray(0,0)']
    for line in original.splitlines():
        if line.startswith('Structure('):
            group = next(g for g in ('gnd','up3s','up3d') if f"'{g}'" in line)
            line = f"Structure('levels.b','{group}.ham',['{group}'])"
        lines.append(line)
        if line.startswith('OptimizeRadial('):
            lines.append("WallTime('ground_scf',0)")
        if line.startswith('Structure('):
            lines.append("WallTime('CI_"+group+"',0)")
        if line.startswith('PrintTable('):
            lines.append("WallTime('export_"+('levels' if 'levels' in line else 'lines')+"',0)")
    lines += ["WallTime('complete',0)"]
    (folder/'run.sf').write_text('\n'.join(lines)+'\n')


def timed_case(backend, workers, z, folder):
    folder.mkdir(parents=True, exist_ok=False)
    if backend=='FAC':
        prepare_fac(z,folder,workers)
        command = [str(SFAC),'run.sf']
    else:
        command = [str(ROOT/'.venv/bin/python'), str(OWNER/'atomx_worker.py'),
                   '--z',str(z),'--out',str(folder)]
    samples = []; mapped = {}; violations = []
    stopped = threading.Event(); start = time.monotonic()
    thread = threading.Thread(target=observe,args=(folder/'pid.txt',stopped,samples,mapped,
                              violations,8,start),daemon=True)
    thread.start()
    with (folder/'stdout.txt').open('w') as stdout, (folder/'stderr.txt').open('w') as stderr:
        proc = subprocess.run([str(OWNER/'child_timer'),str(folder/'timing.json'),
                               str(folder/'pid.txt'),*command],cwd=folder,
                              env=environment(workers),stdout=stdout,stderr=stderr,
                              check=False)
    stopped.set(); thread.join()
    dump(folder/'thread_samples.json',samples); dump(folder/'mapped_runtime.json',mapped)
    dump(folder/'resource_rejections.json',violations)
    record = json.loads((folder/'timing.json').read_text())
    record.update(backend=backend,workers=workers,z=z,folder=str(folder.relative_to(OWNER)),
                  returncode=proc.returncode, actual_threads_peak=max((x['threads'] for x in samples),default=None),
                  active_threads_peak=max((x['running'] for x in samples),default=None),
                  runtime_files=len(mapped),resource_ok=not violations and proc.returncode==0,
                  science_ok=False)
    if record['resource_ok']:
        if backend=='FAC':
            output = (folder/'stdout.txt').read_text()
            its = [line.split(':',1)[1].split() for line in output.splitlines() if line.startswith('optimize loop:')]
            if not its or max(int(v[0]) for v in its)>=150 or 'Maximum iteration reached' in output:
                raise RuntimeError('FAC optimizer gate failed')
            groups,modes = reference(folder)
            levels = parse_levels(folder/'levels.txt')
            arrays = {'levels':[r['ENERGY_eV'] for r in levels]}
            for mode,rows in modes.items():
                arrays[mode] = [[r['DELTA_E_eV'],r['reduced_matrix_element'],r['A_per_s']] for r in rows]
            blocks = {}
            for group in ('gnd','up3s','up3d'):
                result = parse_hamiltonian_binary(folder/f'{group}.ham')
                blocks[group] = [{k:v for k,v in block.items() if k not in ('hamiltonian','values')} for block in result['blocks']]
                for index,block in enumerate(result['blocks']):
                    arrays[f'{group}_H_{index}'] = block['hamiltonian']
            np.savez(folder/'complete_arrays.npz',**arrays)
            dump(folder/'science.json',{'z':z,'groups':groups,'modes':modes,'levels':levels,'blocks':blocks,
                                       'last_optimization':its[-1],
                                       'stage_timing_lines':[r for r in output.splitlines() if r.startswith('WallTime')]})
        else:
            science = json.loads((folder/'science.json').read_text())
            if not science['converged']:
                raise RuntimeError('Atomx optimizer gate failed')
            for mode,configs in science['modes'].items():
                if sum(len(v['upper_levels']) for v in configs.values())!=16 or sum(len(v['lines']) for v in configs.values())!=5:
                    raise RuntimeError(f'Atomx incomplete inventory {mode}')
        record['science_ok'] = True
    dump(folder/'record.json',record)
    return record


def run_group(backend,workers,path):
    load_before = os.getloadavg()[0]
    cases = [timed_case(backend,workers,z,path/f'Z{z:03d}') for z in ZVALUES]
    load_after = os.getloadavg()[0]
    record = {'backend':backend,'workers':workers,'cases':cases,
              'seconds':sum(r['seconds'] for r in cases),
              'load_before':load_before,'load_after':load_after,
              'load_ok':load_before+workers<=len(os.sched_getaffinity(0)) and abs(load_after-load_before)<=4.,
              'resource_ok':all(r['resource_ok'] for r in cases),
              'science_ok':all(r['science_ok'] for r in cases)}
    dump(path/'group.json',record)
    return record


def verify_against_reference(group, reference_group):
    differences = []
    for case,base in zip(group['cases'],reference_group['cases'],strict=True):
        if not case['science_ok'] or not base['science_ok']:
            continue
        with np.load(OWNER/case['folder']/'complete_arrays.npz') as current, np.load(OWNER/base['folder']/'complete_arrays.npz') as ref:
            if current.files != ref.files:
                raise RuntimeError('Complete array keys differ')
            for key in current.files:
                a = current[key]; b = ref[key]
                if a.shape != b.shape:
                    raise RuntimeError('Complete array shapes differ')
                absolute = float(np.max(np.abs(a-b))) if a.size else 0.
                relative = float(np.max(np.abs(a-b)/(np.maximum(np.abs(b),1e-30)))) if a.size else 0.
                # Eigenvector global phases are compared separately via projectors;
                # signed amplitudes remain preserved and also checked directly.
                passed = absolute<=1e-11 and (relative<=1e-10 or absolute==0.)
                differences.append({'z':case['z'],'array':key,'shape':list(a.shape),
                                    'maximum_absolute':absolute,'maximum_relative':relative,'passed':passed})
    group['full_array_correctness_ok'] = bool(differences) and all(r['passed'] for r in differences)
    group['full_array_comparisons'] = len(differences)
    dump(OWNER/group['cases'][0]['folder'].split('/Z')[0]/'complete_array_comparisons.json',differences)


def summarize(groups):
    rows = []
    for backend in ('FAC','Atomx'):
        subset = [g for g in groups if g['backend']==backend]
        for workers in (1,2,4,8):
            selected = [g for g in subset if g['workers']==workers]
            if not selected:
                continue
            times = [g['seconds'] for g in selected]
            controlled = len(selected)==3 and all(g['science_ok'] and g['resource_ok'] and g['load_ok'] and g.get('full_array_correctness_ok',False) for g in selected)
            ratios = []
            for group in selected:
                baseline = next(g for g in subset if g['round']==group['round'] and g['workers']==1)
                if not (baseline['science_ok'] and baseline['resource_ok'] and baseline['load_ok'] and baseline.get('full_array_correctness_ok',False)):
                    controlled = False
                ratios.append(baseline['seconds']/group['seconds'])
            rows.append({'workload':backend+' full four-ion frozen CI/E1 CLI workflow',
                         'workers':workers,'rounds':len(selected),'seconds_median':statistics.median(times),
                         'seconds_min':min(times),'seconds_max':max(times),
                         'speedup':statistics.median(ratios) if controlled else None,
                         'speedup_min':min(ratios) if controlled else None,
                         'speedup_max':max(ratios) if controlled else None,
                         'efficiency':statistics.median(ratios)/workers if controlled else None,
                         'efficiency_min':min(ratios)/workers if controlled else None,
                         'efficiency_max':max(ratios)/workers if controlled else None,
                         'actual_threads_peak':max((c['actual_threads_peak'] or 0 for g in selected for c in g['cases']),default=None),
                         'controlled':controlled,'observed_times_seconds':times,
                         'observed_paired_speedups':ratios,
                         'scope':'Sum of four freshly launched complete child lifetimes: imports/init, SCF, virtuals, separate frozen CI, long/finite photons, raw exports. OS/Numba cache warmed separately; no source build.'})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('warmup','measure','all'),default='all')
    args = parser.parse_args()
    before = source_snapshot(); dump(OWNER/'source_before.json',before)
    if not (OWNER/'actual_sources_build.tar.gz').exists():
        archive_sources(before)
    groups = []; warmups = []
    with timing_lock():
        if args.phase in ('warmup','all'):
            for backend in ('FAC','Atomx'):
                for workers in (1,2,4,8):
                    group = run_group(backend,workers,OWNER/'warmup'/backend/f'p{workers}')
                    warmups.append(group); dump(OWNER/'warmup.json',warmups)
                    print(json.dumps({'phase':'warmup','backend':backend,'workers':workers,
                                      'seconds':group['seconds'],'resource_ok':group['resource_ok'],
                                      'threads':[c['actual_threads_peak'] for c in group['cases']]}),flush=True)
        if args.phase in ('measure','all'):
            if not warmups:
                warmups=json.loads((OWNER/'warmup.json').read_text())
            permitted = {(g['backend'],g['workers']) for g in warmups if g['science_ok'] and g['resource_ok']}
            for round_index,order in enumerate(((1,2,4,8),(8,4,2,1),(1,2,4,8))):
                for backend in ('FAC','Atomx') if round_index%2==0 else ('Atomx','FAC'):
                    for workers in order:
                        if (backend,workers) not in permitted:
                            continue
                        group = run_group(backend,workers,OWNER/'measured'/backend/f'round{round_index:03d}'/f'p{workers}')
                        group['round']=round_index
                        groups.append(group); dump(OWNER/'groups.raw.json',groups)
                        print(json.dumps({'phase':'measured','round':round_index,'backend':backend,
                                          'workers':workers,'seconds':group['seconds'],
                                          'load_ok':group['load_ok'],'resource_ok':group['resource_ok']}),flush=True)
            for group in groups:
                baseline = next(g for g in groups if g['backend']==group['backend'] and g['round']==0 and g['workers']==1)
                verify_against_reference(group,baseline)
            dump(OWNER/'groups.verified.json',groups)
    after = source_snapshot(); dump(OWNER/'source_after.json',after)
    changed = {k:{'before':v,'after':after.get(k)} for k,v in before.items() if after.get(k)!=v}
    missing = sorted(set(before)-set(after)); added = sorted(set(after)-set(before))
    dump(OWNER/'source_guard.json',{'passed':not changed and not missing and not added,
                                   'changed':changed,'missing':missing,'added':added,'files':len(before)})
    result = {'scope':'Independent optimized local models, complete17 levels and10 E1 lines per ion. No equal-accuracy cross-code ranking.',
              'z_values':ZVALUES,'levels':68,'lines':40,'warmups':warmups,
              'cpu_rows':summarize(groups),'groups':groups,
              'source_guard_passed':not changed and not missing and not added,
              'highest_level_reached':False}
    dump(OWNER/'result.json',result)


if __name__ == '__main__':
    main()
