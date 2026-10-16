"""Run an explicit, capped Sn12+ bare-DC CI baseline and retain its evidence."""
import argparse
import hashlib
import json
import os
import resource
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from atomx.workflows.tin_euv import plan_tin_n4_ci, run_tin_n4_ci


class _JsonArraySink:
    """Bounded line serialization; no complete line-list/string allocation."""

    def __init__(self, path):
        self.file = path.open('w')
        self.file.write('[\n')
        self.pending = []
        self.count = 0

    def __call__(self, line):
        self.pending.append(json.dumps(line, separators=(',', ':')))
        if len(self.pending) >= 1024:
            self.flush()

    def flush(self):
        if self.pending:
            if self.count:
                self.file.write(',\n')
            self.file.write(',\n'.join(self.pending))
            self.count += len(self.pending)
            self.pending.clear()

    def close(self):
        self.flush()
        self.file.write('\n]\n')
        self.file.close()


def _source_identity(root):
    paths = sorted((root/'src/atomx').rglob('*.py'))+[root/'tools/run_tin_euv.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-fold', type=int, default=1)
    parser.add_argument('--ns', type=int, default=90)
    parser.add_argument('--max-levels', type=int, default=2000)
    parser.add_argument('--max-determinants', type=int, default=2000)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--tensor-engine', choices=('pair', 'batched'), default='batched')
    parser.add_argument('--storage', choices=('global', 'blocked'), default='global')
    parser.add_argument('--emission-max-fold', type=int)
    parser.add_argument('--memory-gib', type=int, default=8)
    parser.add_argument('--ci-engine', choices=('streamed', 'spinor'), default='streamed')
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('positive timeout required')
    if not 1 <= args.memory_gib <= 64:
        parser.error('memory-gib must be in 1..64')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = _source_identity(root)
    plan = plan_tin_n4_ci(args.max_fold)
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    config = dict(Z=50, charge=12, nucleus='120Sn Fermi',
                  DHF='statistical VN [Kr]4d2', max_fold=args.max_fold,
                  ns=args.ns, ks=9, r_max_bohr=60.,
                  max_levels=args.max_levels, max_determinants=args.max_determinants,
                  timeout_seconds=args.timeout, memory_address_space_cap_GiB=args.memory_gib,
                  operator='Dirac-Coulomb', E1='length long_wavelength',
                  tensor_engine=args.tensor_engine,
                  storage=args.storage, emission_max_fold=args.emission_max_fold,
                  ci_engine=args.ci_engine,
                  thread_environment={name: os.environ.get(name) for name in
                      ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS',
                       'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')},
                  load_average_start=os.getloadavg(),
                  source_sha256=source,
                  acceptance=dict(CI_max_residual_Ha=1e-8, CI_Gram_defect=1e-10),
                  scope='complete declared finite CI baseline; no MBPT/Breit/QED or radial convergence claim')
    (out/'input.json').write_text(json.dumps(config, indent=2)+'\n')
    if os.getloadavg()[0] > (os.cpu_count() or 1):
        raise RuntimeError('load guard prevents a controlled baseline')
    cap = args.memory_gib*1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('declared baseline time cap')))
    signal.alarm(args.timeout)
    started = time.monotonic()

    def progress(*values):
        print('STAGE', *values, 'elapsed', round(time.monotonic()-started, 3), flush=True)

    sink = _JsonArraySink(out/'lines.json') if args.storage == 'blocked' else None
    try:
        result = run_tin_n4_ci(args.max_fold, ns=args.ns, max_levels=args.max_levels,
                              max_determinants=args.max_determinants,
                              tensor_engine=args.tensor_engine, progress=progress,
                              storage=args.storage, emission_max_fold=args.emission_max_fold,
                              memory_bytes=cap, line_sink=sink, ci_engine=args.ci_engine)
        if sink is not None:
            sink.close()
            sink = None
        signal.alarm(0)
        for name in ('levels', 'lines', 'summary', 'diagnostics'):
            if result[name] is not None:
                (out/f'{name}.json').write_text(json.dumps(result[name], indent=2)+'\n')
        arrays = dict(result['arrays'])
        arrays.update({f'orbital_n{n}_k{k}_{component}': pair[i]
                       for (n, k), pair in result['orbitals'].items()
                       for i, component in enumerate(('P', 'Q'))})
        np.savez_compressed(out/'native_arrays.npz', **arrays)
        # A same-source replay can also reconstruct the exact native workspace
        # from these grid controls and input.json; no external reference values
        # enter any Hamiltonian or line calculation.
        unchanged = _source_identity(root) == source
        if not unchanged:
            raise RuntimeError('source changed during calculation; do not certify this run')
        report = dict(status='DECLARED_BARE_DC_BASELINE_SOLVED', source_unchanged=True,
                      complete_paper_reproduction=False, core_MBPT_implemented=False,
                      wall_seconds=time.monotonic()-started,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      levels=len(result['levels']),
                      lines=result['line_count'] if result['lines'] is None else len(result['lines']),
                      summary=result['summary'], diagnostics=result['diagnostics'])
        display = {k: v for k, v in report.items() if k != 'diagnostics'}
        display['diagnostics'] = {k: report['diagnostics'][k] for k in
                                  ('CI_residual_Ha', 'CI_Gram_defect', 'timings')}
        print(json.dumps(display), flush=True)
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    except Exception as exc:
        report = dict(status='FAILED', error=repr(exc), wall_seconds=time.monotonic()-started,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      traceback=traceback.format_exc())
        (out/'failure.json').write_text(json.dumps(report, indent=2)+'\n')
        raise
    finally:
        signal.alarm(0)
        if sink is not None:
            sink.close()


if __name__ == '__main__':
    main()
