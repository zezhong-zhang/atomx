"""Root-sealed two-branch foreground launch; stdlib accounting, no science."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

B = Path(__file__).resolve().parent
R = B.parents[2]
BRANCHES = ('history10', 'history40')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + '\n')


def main():
    expected_seals = {'history10': 'e222dae6e93548e91c8768a44885c880cbaafc7ded8a98169629b1bc47de6fee',
                      'history40': 'f696ff6c0a1d16e076950a4226a2e09f904953a5763ad472f66bd8b94d3e0f1c'}
    seals = {}
    for name in BRANCHES:
        d = B / name
        assert sha(d / 'execution_contract.json') == expected_seals[name]
        s = json.loads((d / 'execution_contract.json').read_text())
        assert sha(B / 'run.py') == s['driver_sha256']
        assert sha(d / 'contract.json') == s['contract_sha256']
        assert sha(d / 'preparation_verification.json') == s['preparation_sha256']
        assert not any((d / f).exists() for f in ('once_marker.json', 'root_execution.json', 'actual', 'root_numba_cache'))
        seals[name] = s
    host = {'load_average': list(os.getloadavg()), 'affinity': sorted(os.sched_getaffinity(0))}
    host['mem_available_KiB'] = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
    assert all(s['CPU'] in host['affinity'] for s in seals.values())
    assert host['mem_available_KiB'] >= int(7.9 * 2**20)
    assert host['load_average'][0] <= 32
    assert shutil.disk_usage(B).free >= 8 * 2**30
    with (B / 'pair_once_marker.json').open('x') as h:
        json.dump({'created_utc': now(), 'exactly_one_pair_launch': True,
                   'execution_seals_sha256': expected_seals, 'launcher_sha256': sha(__file__)}, h, indent=2)
        h.write('\n')
        h.flush()
        os.fsync(h.fileno())
    workers = []
    for name in BRANCHES:
        d, seal = B / name, seals[name]
        with (d / 'once_marker.json').open('x') as h:
            json.dump({'created_utc': now(), 'execution_contract_sha256': expected_seals[name], 'exactly_one_launch': True}, h, indent=2)
            h.write('\n')
            h.flush()
            os.fsync(h.fileno())
        env = os.environ.copy()
        thread_names = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS')
        for k in thread_names:
            env[k] = '1'
        cache = d / 'root_numba_cache'
        cache.mkdir(exist_ok=False)
        env['NUMBA_CACHE_DIR'] = str(cache)
        command = ['timeout', '--signal=TERM', '--kill-after=5s', '615s', '/data/code/atomx/.venv/bin/python',
                   str(B / 'run.py'), '--branch', name, '--execution-contract', str(d / 'execution_contract.json')]
        receipt = {'host_admission': host, 'status': 'foreground_started', 'started_utc': now(),
                   'command': command, 'cwd': str(R), 'launcher_sha256': sha(__file__),
                   'execution_contract_sha256': expected_seals[name], 'driver_sha256': sha(B / 'run.py'),
                   'contract_sha256': seal['contract_sha256'], 'preparation_sha256': seal['preparation_sha256'],
                   'thread_variables': {k: env[k] for k in thread_names}, 'CPU': seal['CPU'],
                   'aggregate_two_workers_address_space_GiB': 7.9,
                   'timing_scope': 'Uncontrolled concurrent same-model native MCDHF correctness comparison; no speedup claim'}
        write(d / 'root_execution.json', receipt)
        log = (d / 'root_execution.log').open('wb')
        begin = time.monotonic()
        process = subprocess.Popen(command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)
        workers.append((name, d, receipt, process, log, begin))
    def close_worker(worker):
        name, d, receipt, process, log, begin = worker
        code = process.wait()
        log.close()
        receipt.update(status='foreground_closed', exit_code=code, elapsed_uncontrolled_seconds=time.monotonic()-begin,
                       finished_utc=now(), log_sha256=sha(d / 'root_execution.log'))
        if (d / 'actual').exists():
            receipt['actual_files_sha256'] = {str(p.relative_to(d)): sha(p) for p in sorted((d / 'actual').rglob('*.json'))}
        write(d / 'root_execution.json', receipt)
        output = {k: receipt[k] for k in ('status', 'exit_code', 'elapsed_uncontrolled_seconds', 'log_sha256')}
        output['branch'] = name
        print(json.dumps(output), flush=True)
        return output

    outputs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(close_worker, worker) for worker in workers]):
            outputs.append(future.result())
    outputs.sort(key=lambda item: item['branch'])
    write(B / 'root_pair_execution.json', {'status': 'foreground_pair_closed', 'workers': outputs,
          'launcher_sha256': sha(__file__), 'pair_once_marker_sha256': sha(B / 'pair_once_marker.json'),
          'branch_receipts_sha256': {n: sha(B/n/'root_execution.json') for n in BRANCHES},
          'aggregate_address_space_GiB': 7.9, 'timing_claim_valid': False})
    print(json.dumps(outputs))
    return int(any(v['exit_code'] != 0 for v in outputs))


if __name__ == '__main__':
    raise SystemExit(main())
