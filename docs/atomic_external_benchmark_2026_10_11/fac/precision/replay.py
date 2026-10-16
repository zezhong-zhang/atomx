"""Fresh complete FAC DC panels, strict gates, and complete signed artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import resource
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from atomx.angular.csf_hamiltonian import exact_block_matrix
from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
from atomx.physics.csf.matrix import (
    _group_csf_indices,
    _radial_evaluator,
    _reset_memo,
    build_ci_matrix,
)
from benchmarks import bench_reference_ci_operator as bench
from benchmarks.fac_runner import parse_hamiltonian_binary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FAC = Path('/home/zzhang/code/fac').resolve()
CONTRACT = HERE.parent / 'contract.json'
LIMIT_BYTES = 8 * 1024**3
RUNTIME_PATHS: set[Path] = set()
RUNTIME_FIRST: dict[str, str] = {}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def mapped_files(pid):
    try:
        rows = Path(f'/proc/{pid}/maps').read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return
    for row in rows:
        pieces = row.split(maxsplit=5)
        if len(pieces) == 6 and pieces[5].startswith('/'):
            path = Path(pieces[5].removesuffix(' (deleted)'))
            if path.is_file() and path not in RUNTIME_PATHS:
                RUNTIME_PATHS.add(path)
                RUNTIME_FIRST[str(path)] = digest(path)


def runtime_snapshot():
    mapped_files(os.getpid())
    return {str(p): digest(p) for p in sorted(RUNTIME_PATHS)}


def source_snapshot():
    paths = set((ROOT / 'src/atomx').rglob('*.py'))
    paths.update((ROOT / 'src/atomx').rglob('*.f90'))
    paths.update((ROOT / 'benchmarks').glob('*.py'))
    paths.update((ROOT / 'tests').glob('test_fac_ci_operator.py'))
    paths.update((ROOT / 'tests').glob('test_ci_analytic_slater_condon.py'))
    paths.update((ROOT / 'tests').glob('test_ci_grid_kernel.py'))
    for pattern in ('*.c', '*.h', '*.f', '*.inc', '*.a', '*.py', '*Makefile*'):
        paths.update(FAC.rglob(pattern))
    paths.update([HERE / 'replay.py', HERE / 'build_frontend.sh',
                  HERE / 'fac-ci-diagnostic', CONTRACT,
                  FAC / 'config.log', FAC / 'configure', FAC / 'configure.ac',
                  FAC / 'PFAC.egg-info/PKG-INFO',
                  ROOT / 'benchmarks/runs/2026-09-30-ci-common-integrals/sfac.c',
                  ROOT / 'benchmarks/runs/2026-09-30-ci-common-integrals/build-frontend.sh',
                  Path('/data/code/atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu/libgfortran.so.5.0.0')])
    return {str(p.resolve()): digest(p) for p in sorted(paths) if p.is_file()}


def run_reference(command, directory, *, stdin=None, timeout=180):
    """Blocking wait4 timing; watchdog and observer do not poll process.wait."""
    if stdin is not None:
        raise ValueError('This diagnostic contract uses script files, not stdin')
    write(directory / 'command.json', command)
    samples = []
    finished = threading.Event()
    timed_out = threading.Event()
    before = os.getloadavg()
    affinity = len(os.sched_getaffinity(0))
    if before[0] + 1 > affinity:
        raise RuntimeError('CPU load preflight failed')
    with (directory / 'stdout.txt').open('wb') as stdout, (directory / 'stderr.txt').open('wb') as stderr:
        start = time.monotonic_ns()
        child = subprocess.Popen(command, cwd=directory, stdout=stdout, stderr=stderr,
                                 start_new_session=True)

        def observe():
            while not finished.wait(0.01):
                mapped_files(child.pid)
                try:
                    tids = list(Path(f'/proc/{child.pid}/task').iterdir())
                    samples.append(dict(seconds=(time.monotonic_ns()-start)/1e9,
                                        threads=len(tids)))
                except FileNotFoundError:
                    pass

        def watchdog():
            if not finished.wait(600):
                timed_out.set()
                os.killpg(child.pid, signal.SIGKILL)

        observer = threading.Thread(target=observe, daemon=True)
        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        observer.start()
        watchdog_thread.start()
        _, status, usage = os.wait4(child.pid, 0)
        elapsed = (time.monotonic_ns() - start) / 1e9
        child.returncode = os.waitstatus_to_exitcode(status)
        finished.set()
        observer.join()
        watchdog_thread.join()
    after = os.getloadavg()
    peak = max((s['threads'] for s in samples), default=None)
    process = dict(status='timeout' if timed_out.is_set() else 'exited',
                   elapsed_s=elapsed, exit_code=child.returncode, pid=child.pid,
                   timer='CLOCK_MONOTONIC around Popen plus blocking os.wait4',
                   pure_backend_efficiency_claim=False, load_before=list(before),
                   load_after=list(after), affinity=affinity, peak_threads=peak,
                   thread_samples=samples, maxrss_bytes=usage.ru_maxrss*1024,
                   user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                   thread_cap_pass=peak is not None and peak <= 8,
                   load_gate_pass=before[0]+1 <= affinity and abs(after[0]-before[0]) <= 4,
                   wall_cap_seconds=600, address_cap_bytes=LIMIT_BYTES)
    write(directory / 'process.json', process)
    if child.returncode or timed_out.is_set() or not process['thread_cap_pass']:
        raise RuntimeError(f'FAC process/resource failure: {process}')
    return elapsed


def gate(values, reference):
    errors = np.abs(values-reference)
    relative = errors / np.maximum(np.abs(reference), np.finfo(float).tiny)
    return dict(max_absolute_Ha=float(errors.max(initial=0)),
                max_relative=float(relative.max(initial=0)),
                absolute_pass=bool(np.all(errors <= 1e-11)),
                relative_pass=bool(np.all(relative <= 1e-10)),
                strict_pass=bool(np.all(errors <= 1e-11) and np.all(relative <= 1e-10)),
                count=int(values.size))


def collect(case, folder):
    blocks, keys, orbitals = bench.angular_blocks(case['configurations'])
    radial = {keys[int(fields[0])]: float(fields[1])
              for line in (folder/'integrals.txt').read_text().splitlines()
              if (fields := line.split())}
    arrays = {'fac_radial_integrals_Ha': np.array([radial[k] for k in keys])}
    write(folder/'atomx_complete_csf_signatures.json', [
        dict(two_J=sym[0], parity=sym[1], signatures=[sig.key for sig in signatures])
        for sym, signatures in blocks.items()])
    fac_blocks = {(b['symmetry']//2, (-1)**(b['symmetry'] % 2)): b
                  for b in parse_hamiltonian_binary(folder/'hamilton.b')['blocks']}
    summary = {}
    all_eigenvalues = {}
    inventory = []
    for sym, signatures in blocks.items():
        name = f'J{sym[0]}_parity{sym[1]}'
        h_fac = fac_blocks[sym]['hamiltonian']
        h_atomx = exact_block_matrix(signatures, radial.__getitem__)
        arrays[name+'_fac_matrix_Ha'] = h_fac
        arrays[name+'_fac_basis'] = np.array(fac_blocks[sym]['basis'])
        arrays[name+'_atomx_common_integrals_matrix_Ha'] = h_atomx
        for label, h in [('fac', h_fac), ('common_integrals', h_atomx)]:
            arrays[name+'_'+label+'_eigenvalues_Ha'] = np.linalg.eigvalsh(h)
            all_eigenvalues.setdefault(label, []).append(arrays[name+'_'+label+'_eigenvalues_Ha'])
        inventory.append(dict(two_J=sym[0], parity=sym[1], dimension=len(signatures),
                              matrix_prefix=name))
    reference = np.concatenate(all_eigenvalues['fac'])
    summary['common_integrals'] = gate(np.concatenate(all_eigenvalues['common_integrals']), reference)
    # Shared reference zero also retained separately, never substituted for full-energy gate.
    sources = bench.read_spinors(folder, orbitals)
    for key, source in sources.items():
        arrays[f'fac_spinor_{key[0]}_{key[1]}'] = source
    csfs = enumerate_jj_csfs_many(case['configurations'])
    for points in (2000, 4000):
        waves, workspace = bench.common_grid(sources, case['z'], points, alpha=case['fac_alpha'])
        arrays[f'grid_{points}_r'] = workspace.r
        arrays[f'grid_{points}_difr'] = workspace.difr
        arrays[f'grid_{points}_rv_nucleus'] = workspace.rv_nuc
        for key, (p, q) in waves.items():
            arrays[f'grid_{points}_P_{key[0]}_{key[1]}'] = p
            arrays[f'grid_{points}_Q_{key[0]}_{key[1]}'] = q
        _reset_memo()
        evaluate = _radial_evaluator(waves, workspace, case['z'], workspace.n)
        raw = {key: evaluate(key) for key in keys}
        shifts = {key: bench.one_body_constant_shift(waves, workspace, key, 1/case['fac_alpha'])
                  for key in keys}
        aligned = {key: raw[key]+shifts[key] for key in keys}
        arrays[f'grid_{points}_raw_radial_Ha'] = np.array([raw[k] for k in keys])
        arrays[f'grid_{points}_c_alignment_Ha'] = np.array([shifts[k] for k in keys])
        arrays[f'grid_{points}_aligned_radial_Ha'] = np.array([aligned[k] for k in keys])
        public_h = build_ci_matrix(csfs, waves, get_shared_orbital_basis(csfs),
                                   workspace, case['z'], workspace.n)
        arrays[f'grid_{points}_public_complete_matrix_Ha'] = public_h
        pub_diff = []
        for sym, indices in _group_csf_indices(csfs):
            name = f'J{sym[0]}_parity{sym[1]}'
            raw_h = exact_block_matrix(blocks[sym], raw.__getitem__)
            aligned_h = exact_block_matrix(blocks[sym], aligned.__getitem__)
            pub_diff.append(np.max(np.abs(public_h[np.ix_(indices, indices)]-raw_h)))
            for label, h in [('raw', raw_h), ('aligned', aligned_h)]:
                prefix = f'grid_{points}_{name}_{label}'
                arrays[prefix+'_matrix_Ha'] = h
                arrays[prefix+'_eigenvalues_Ha'] = np.linalg.eigvalsh(h)
                all_eigenvalues.setdefault(f'{points}_{label}', []).append(arrays[prefix+'_eigenvalues_Ha'])
        summary[str(points)] = dict(
            raw_eigenvalues=gate(np.concatenate(all_eigenvalues[f'{points}_raw']), reference),
            c_aligned_eigenvalues=gate(np.concatenate(all_eigenvalues[f'{points}_aligned']), reference),
            raw_radial_integrals=gate(arrays[f'grid_{points}_raw_radial_Ha'], arrays['fac_radial_integrals_Ha']),
            c_aligned_radial_integrals=gate(arrays[f'grid_{points}_aligned_radial_Ha'], arrays['fac_radial_integrals_Ha']),
            public_complete_matrix_versus_exact_blocks_max_Ha=float(max(pub_diff)),
            public_complete_matrix_symmetric=bool(np.array_equal(public_h, public_h.T)),
            analytically_aligned_c=1/case['fac_alpha'])
    path = folder/'complete_signed_arrays.npz'
    np.savez_compressed(path, **arrays)
    # Read every coordinate back; do not verify only a sampled norm or eigenvalue slice.
    with np.load(path) as decoded:
        assert set(decoded.files) == set(arrays)
        for key, value in arrays.items():
            assert np.array_equal(value, decoded[key]), key
    write(folder/'array_inventory.json', {k: dict(shape=list(v.shape), dtype=str(v.dtype),
          nbytes=v.nbytes, logical_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in arrays.items()})
    write(folder/'symmetry_inventory.json', inventory)
    return dict(z=case['z'], sequence=case['sequence'], levels=case['n_levels'],
                integrals=case['n_integrals'], complete_symmetry_blocks=len(blocks),
                gates=summary, complete_arrays=len(arrays), complete_coordinates=sum(v.size for v in arrays.values()),
                archive_sha256=digest(path), all_arrays_read_back=True,
                process=json.loads((folder/'process.json').read_text()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT_BYTES, LIMIT_BYTES))
    if Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0].isdigit():
        available = int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024
        assert available >= 32*1024**3, '24 GiB reservation + 8 GiB headroom'
    before = source_snapshot()
    write(folder/'source_before.json', before)
    with tarfile.open(folder/'actual_sources_and_build_inputs.tar.gz', 'w:gz') as archive:
        for path in before:
            archive.add(path, arcname=path.lstrip('/'))
    mapped_files(os.getpid())
    write(folder/'runtime_before.json', runtime_snapshot())
    settings = SimpleNamespace(fac=HERE/'fac-ci-diagnostic', fac_points=8000,
                               fac_density=4., atomx_points=[2000, 4000], disable_fac_yk_cache=True)
    bench.run_reference = run_reference
    original_script = bench.script

    def explicit_serial_script(*args, **kwargs):
        return 'InitializeMPI(1)\n' + original_script(*args, **kwargs)

    bench.script = explicit_serial_script
    cases = []
    start = time.monotonic()
    with (HERE.parents[1]/'timing.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for sequence, z, expected in [('helium_like', 10, 98), ('carbon_like', 6, 20),
                                      ('neon_like', 26, 37), ('neon_like', 82, 37)]:
            if time.monotonic()-start >= 570:
                raise TimeoutError('Prospective complete-lane budget reached')
            label = f'{sequence}-Z{z:03d}'
            path = folder/label
            case = bench.run_case(z, sequence, path, settings)
            assert case['n_levels'] == expected
            write(path/'benchmark_case.json', case)
            summary = collect(case, path)
            cases.append(summary)
            write(folder/'cases.partial.json', cases)
            print(json.dumps(dict(case=label, common=summary['gates']['common_integrals'],
                                  aligned4000=summary['gates']['4000']['c_aligned_eigenvalues'])), flush=True)
    after = source_snapshot()
    write(folder/'source_after.json', after)
    assert after == before, 'Source epoch changed during science'
    runtime_after = runtime_snapshot()
    write(folder/'runtime_first_observed.json', RUNTIME_FIRST)
    write(folder/'runtime_after.json', runtime_after)
    assert runtime_after == RUNTIME_FIRST, 'Mapped dependency changed during science'
    result = dict(scope='Complete finite Dirac-Coulomb CI numerical comparison; not experimental precision or full solver speed ranking',
                  cases=cases, total_levels=sum(c['levels'] for c in cases),
                  total_integrals=sum(c['integrals'] for c in cases),
                  source_and_runtime_unchanged=True, source_files=len(before),
                  loaded_dependencies=len(runtime_after), elapsed_seconds=time.monotonic()-start,
                  parent_absolute_Ha=1e-11, parent_relative=1e-10,
                  highest_level_reached=False, experimental_accuracy_established=False,
                  fac_compressed_Yk=False, fac_points=8000, fac_density=4,
                  atomx_points=[2000, 4000], signed_matrices_preserved=True,
                  matrix_phase_equality_claim=False, performance_speedup_claim=False)
    write(folder/'result.json', result)


if __name__ == '__main__':
    main()
