"""Current source, complete saved Yb term2 strong scaling; no production solve."""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import threading

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
from atomx.methods.core_sd import CoreSD
from atomx.potentials.nuclear import make_nuclear_potential

BASE = ROOT / 'docs/atomic_pci_sd_native_2026_10_09'
CONTRACT = json.loads((OWNER.parent / 'contract.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def memory():
    m = {r.split(':')[0]: int(r.split()[1]) * 1024
         for r in Path('/proc/meminfo').read_text().splitlines()}
    return m['MemAvailable']


def main():
    out = OWNER / 'yb_stage002'
    out.mkdir(exist_ok=False)
    cap = CONTRACT['timing_contract']['max_per_process_address_bytes']
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    wall_guard = threading.Timer(600, lambda: os._exit(124))
    wall_guard.daemon = True
    wall_guard.start()
    original = BASE / 'valence_term2_full003'
    parent = BASE / 'valence_term2_paired008'
    physical = BASE / 'yb_full_run008'
    kernel = ROOT / 'src/atomx/methods/_pci_valence_term2_streamed.f90'
    bound = json.loads((BASE / 'valence_term2_parallel_full006/resource.json').read_text())['bound_bytes']
    preflight = dict(bound_bytes=bound, address_cap_bytes=cap,
                     available_bytes=memory(), disk_bytes=shutil.disk_usage(out).free,
                     minimum_headroom_bytes=8 * 1024 ** 3, estimated_stage_seconds=510,
                     estimated_setup_seconds=60, process_workers=8, load=os.getloadavg(),
                     affinity=list(os.sched_getaffinity(0)))
    preflight['passed'] = (bound < cap and preflight['available_bytes'] >= cap + 8 * 1024 ** 3
                           and preflight['disk_bytes'] > 8 * 1024 ** 3)
    write(out / 'preflight.json', preflight)
    if not preflight['passed']:
        raise MemoryError('complete unchanged input rejected by fixed resource cap')
    local_contract = dict(hypothesis='Complete disjoint core-channel term2 action scales without signed-coordinate changes',
        fixed='Yb 68 core electrons / all 414 virtual radial orbitals / ns123 / Fermi Z70 / lmax3 Emax5; original numerator and supplied response, all 1076 channels, all allocated output coordinates',
        workers=[1, 2, 4, 8], response=True, rounds=3,
        orders=[[1, 2, 4, 8], [8, 4, 2, 1], [2, 8, 1, 4]],
        warmup_worker=8, precision_gate_Ha=1e-11, relative_gate=1e-10,
        timing_scope='warm complete term2 value and analytic response with native C input/output copies; no setup, integral preparation, full action or solve',
        reference_scope='preserved original pinned pCI diagram full output; not an independent pCI software production run',
        original_live_deferred='original whole Yb term2 measured 602.4896 seconds per call, exceeding current 600-second lane limit; original sealed full arrays remain comparison anchor',
        end_to_end_claim=False, kernel_sha256=digest(kernel))
    write(out / 'contract.json', local_contract)
    sources = [p for p in sorted((ROOT / 'src/atomx').rglob('*.py'))]
    sources += sorted((ROOT / 'src/atomx/methods').glob('_pci*.f90'))
    sources += [Path(__file__), ROOT / 'tools/build_pci_sd_backend.py', ROOT / 'tools/build_pci_valence_backend.py']
    identities = {}
    for p in sources:
        rel = p.relative_to(ROOT)
        dest = out / 'source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        identities[str(rel)] = digest(p)
    write(out / 'source_sha256.json', identities)
    inputs = [physical / 'yb_full_inputs.npz', physical / 'yb_full_plan.json',
              original / 'term2_6_-1.npz', original / 'result.json',
              parent / 'backend/manifest.json', BASE / 'valence_term2_parallel_full006/resource.json']
    input_manifest = {}
    for p in inputs:
        rel = p.relative_to(ROOT)
        dest = out / 'inputs' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        input_manifest[str(rel)] = digest(p)
    write(out / 'input_sha256.json', input_manifest)
    setup_start = time.perf_counter()
    backend = out / 'backend'
    shutil.copytree(parent / 'backend', backend)
    manifest = json.loads((backend / 'manifest.json').read_text())
    command = manifest['command'][:]
    command = [x for x in command if x not in ('valence_term2_streamed.o', 'valence_term2_one.o')]
    compile_commands = []
    objects = []
    for workers in (1, 2, 4, 8):
        name = f'benchmark_term2_{workers}'
        body = kernel.read_text().replace('CASE_THREADS', str(workers)).replace('valence_term2_streamed', name)
        body += f'''
subroutine atomx_benchmark_term2_{workers}(energy,d,dd,rd,pd) bind(C)
 use native_valence_bridge
 implicit none
 real(c_double),value,intent(in) :: energy
 real(c_double),intent(in) :: d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
 real(c_double),intent(out) :: rd(nxx,nxx,nvh),pd(nxx,nxx,nvh)
 xvi=d; dxvi=dd; xvo=0d0; dxvo=0d0
 call {name}(energy,.true.)
 rd=xvo; pd=dxvo
end subroutine
'''
        (backend / f'{name}.f90').write_text(body)
        invocation = command[:command.index('-shared')] + ['-fopenmp', '-c', f'{name}.f90']
        compile_commands.append(invocation)
        objects.append(f'{name}.o')
    command[command.index('-o'):command.index('-o')] = objects
    with (backend / 'benchmark_compiler.log').open('w') as log:
        for invocation in compile_commands + [command]:
            subprocess.run(invocation, cwd=backend, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    manifest['command'] = command
    manifest['compile_commands'] += compile_commands + [command]
    manifest['benchmark'] = local_contract
    manifest['files'] = {p.name:digest(p) for p in backend.iterdir() if p.is_file() and p.name != 'manifest.json'}
    write(backend / 'manifest.json', manifest)
    plan = json.loads((physical / 'yb_full_plan.json').read_text())
    occupied, virtual = (tuple(map(tuple, plan[k])) for k in ('occupied_keys', 'virtual_keys'))
    with np.load(physical / 'yb_full_inputs.npz', allow_pickle=False) as data:
        workspace = {k.removeprefix('workspace_'):data[k].copy() for k in data if k.startswith('workspace_')}
        workspace = {k:v.item() if v.ndim == 0 else v for k,v in workspace.items()}
        waves = {k:(data[f'P_{k[0]}_{k[1]}'].copy(), data[f'Q_{k[0]}_{k[1]}'].copy()) for k in occupied + virtual}
    core = CoreSD(waves, workspace, 70, 123, occupied_keys=occupied, virtual_keys=virtual,
        nuclear_potential=make_nuclear_potential(70, 'fermi'), backend_directory=backend,
        memory_bytes=cap, term21_threads=8, term21_mode='streamed', core_singles_blas=True,
        core_single_to_doubles_blas=True, radial_representation='polynomial')
    setup_seconds = time.perf_counter() - setup_start
    if core._blas_runtime['parallel'] != 2 or core._blas_runtime['threads'] != 1:
        raise ValueError('single inner thread OpenMP BLAS required')
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='F_CONTIGUOUS')
    init = core._backend.atomx_val_init
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ptr]
    init.restype = None
    slot = next(g.index((6, -1)) for g in core.plan['groups'] if (6, -1) in g)
    init(slot + 1, -1, core.plan['raw_amplitudes'], np.zeros(core.plan['raw_amplitudes']))
    with np.load(original / 'term2_6_-1.npz', allow_pickle=False) as d:
        numerator, prime, reference, reference_prime, serial, serial_prime = (np.array(d[k], order='F')
            for k in ('numerator','prime','original','original_prime','streamed','streamed_prime'))
    index = next(i for i,x in enumerate(core.integrals.labels) if x[:2] == (6, -1))
    energy = float(core.integrals.fock_Ha[index,index].real + 0.075)
    if energy != json.loads((original/'result.json').read_text())['rows'][0]['energy_Ha']:
        raise AssertionError('complete fixed denominator energy changed')
    functions = {n:getattr(core._backend, f'atomx_benchmark_term2_{n}') for n in (1,2,4,8)}
    for f in functions.values():
        f.argtypes = [ctypes.c_double] + [ptr] * 4
        f.restype = None
    actual, actual_prime = np.zeros_like(numerator), np.zeros_like(prime)
    observations = []
    sequence = [(None,8)] + [(round_id,n) for round_id,order in enumerate(local_contract['orders']) for n in order]
    for ordinal, (round_id, workers) in enumerate(sequence):
        with open(CONTRACT['timing_contract']['lock_file'], 'a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            before = os.getloadavg()
            started = time.perf_counter()
            functions[workers](energy, numerator, prime, actual, actual_prime)
            seconds = time.perf_counter() - started
            after = os.getloadavg()
        error = max(float(np.max(abs(a-b))) for a,b in [(actual,reference),(actual_prime,reference_prime)])
        relative = max(float(np.max(abs(a-b))) / max(float(np.max(abs(b))), 1e-300)
                       for a,b in [(actual,reference),(actual_prime,reference_prime)])
        bitwise = np.array_equal(actual, serial) and np.array_equal(actual_prime, serial_prime)
        guard = max(before+after) + workers <= len(os.sched_getaffinity(0)) and max(abs(a-b) for a,b in zip(before,after)) <= 4.0
        record = dict(ordinal=ordinal, round=round_id, warmup=round_id is None, workers=workers,
            seconds=seconds, load_before=before, load_after=after, timing_guard_pass=guard,
            maximum_signed_error_Ha=error, maximum_relative_error=relative, bitwise_equal_to_prior_serial=bool(bitwise),
            finite=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(actual_prime))))
        observations.append(record)
        np.savez_compressed(out/f'map_{ordinal}_{workers}.npz', signed=actual, response=actual_prime)
        write(out/'observations.json', observations)
        print(json.dumps(record), flush=True)
        if error > 1e-11 or relative > 1e-10 or not bitwise or not record['finite']:
            raise AssertionError('complete unchanged signed action failed original numerical gate')
    all_guard = all(r['timing_guard_pass'] for r in observations if not r['warmup'])
    scaling = []
    for workers in (1,2,4,8):
        own = [r for r in observations if r['workers']==workers and not r['warmup']]
        ratios = [next(r['seconds'] for r in observations if r['round']==x['round'] and r['workers']==1)/x['seconds'] for x in own]
        scaling.append(dict(workers=workers, seconds=[x['seconds'] for x in own], median_seconds=float(np.median([x['seconds'] for x in own])),
            observed_speedups=ratios, speedup=float(np.median(ratios)) if all_guard else None,
            efficiency=float(np.median(ratios))/workers if all_guard else None))
    runtime = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        name = line.split()[-1]
        if name.startswith('/') and any(x in name for x in ('libopenblas','libgomp','libgfortran','libquadmath')):
            p=Path(name)
            if not any(x['path']==str(p) for x in runtime): runtime.append(dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size))
    changed = {str(p.relative_to(ROOT)):digest(p) for p in sources if digest(p)!=identities[str(p.relative_to(ROOT))]}
    input_changed = {str(p.relative_to(ROOT)):digest(p) for p in inputs if digest(p)!=input_manifest[str(p.relative_to(ROOT))]}
    result=dict(status='complete signed control passed' if not changed and not input_changed else 'identity failure',
        precision_pass=True, maximum_signed_error_Ha=max(r['maximum_signed_error_Ha'] for r in observations),
        all_workers_bitwise_equal=True, timing_guard_pass=all_guard, scaling=scaling,
        setup_seconds=setup_seconds, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        shape=list(numerator.shape), output_elements_per_array=int(numerator.size), full_core_electrons=68,
        full_virtual_orbitals=len(virtual), energy_Ha=energy, full_response=True, runtime=runtime,
        current_source_changes=changed, original_input_changes=input_changed, core_provenance=dict(core.provenance),
        scope=local_contract['timing_scope'], independent_pci_software_claim=False, end_to_end_claim=False,
        original_live_deferred=local_contract['original_live_deferred'])
    write(out/'result.json', result)
    if changed or input_changed: raise AssertionError('sources or scientific inputs changed during experiment')

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        p=OWNER/'yb_stage002'
        if p.exists(): write(p/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise
