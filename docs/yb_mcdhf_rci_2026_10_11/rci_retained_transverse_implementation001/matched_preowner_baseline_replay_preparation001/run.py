"""Serialized source-isolated owner control; controller imports stdlib only."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    contract = json.loads((D / 'executor_contract.json').read_text())
    for item in contract['prepared_files']:
        assert hashlib.sha256((D / item['path']).read_bytes()).hexdigest() == item['sha256']
    if not args.execute:
        print('Prepared source/input schema only; --execute requires root admission.')
        return
    out = D / 'actual'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    env = dict(os.environ)
    env.update({name: '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    results = []
    try:
        for epoch in ('current', 'preowner'):
            remaining = 120-(time.monotonic()-started)
            assert remaining > 0
            with (out / f'{epoch}.log').open('x') as log:
                child = subprocess.run([sys.executable, '-I', str(D / 'worker.py'), '--epoch', epoch],
                                       env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            assert child.returncode == 0, f'{epoch} worker exit{child.returncode}'
            results.append(json.loads((out / epoch / 'result.json').read_text()))
        for field in ('H_shape', 'gradient_shape', 'H_array_bytes_sha256', 'gradient_array_bytes_sha256', 'owner_header_sha256'):
            assert results[0][field] == results[1][field], f'matched baseline {field} changed'
        write(out / 'result.json', dict(status='matched_current_preowner_baseline_control_closed',
              same_epoch_H_and_signed_gradient_bitwise=True, DC_schema1_header_equal=True,
              workers=results, input_NPZ_loads=6, controller_NPZ_loads=0,
              H=2, gradient=2, CI=0, original_mixed_suite_passed=False,
              original_large_h_FD_qualified=False, optimizer_authorized=False,
              elapsed_uncontrolled_s=time.monotonic()-started))
        print('Two source-isolated native baseline controls closed; original failures retained.')
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), completed_workers=results,
                                       elapsed_uncontrolled_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
