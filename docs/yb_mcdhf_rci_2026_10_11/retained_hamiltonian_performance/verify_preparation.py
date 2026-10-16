"""Read-only preparation and metadata regression controls; zero Atomx imports."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    started = time.perf_counter()
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 3:
        raise RuntimeError('Metadata validation requires explicit available CPU')
    os.sched_setaffinity(0, {available[-2]})
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    resource.setrlimit(resource.RLIMIT_AS, (int(1.0 * 2**30),) * 2)
    import numpy as np

    spec = importlib.util.spec_from_file_location('prepared_panel', HERE / 'run.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    contract = driver.load(HERE / 'contract.json')
    checks = driver.input_checks(contract)

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError('Preparation control failed: ' + name)

    def rejects(name, operation):
        try:
            operation()
        except (TypeError, ValueError):
            check(name, True)
        else:
            check(name, False)

    for key, row in contract['prerequisites'].items():
        check('frozen prerequisite:' + key, driver.sha(ROOT / row['path']) == row['expected_sha256'])
    qualified = driver.load(ROOT / contract['prerequisites']['qualification_result']['path'])
    proof = driver.load(ROOT / contract['prerequisites']['qualification_verification']['path'])
    check('actual live qualification1491 all pass', qualified['qualification_numeric_passed'] is True
          and qualified['checks_count'] == len(qualified['checks']) == 1491
          and all(r['passed'] is True for r in qualified['checks']))
    check('independent saved proof3217 all pass', proof['all_passed'] is True
          and proof['checks_count'] == len(proof['checks']) == 3217
          and all(r['passed'] is True for r in proof['checks']))
    for name, expected in proof['inputs_sha256'].items():
        check('independent proof input:' + name, driver.sha(ROOT / name) == expected)
    for key in ('qualification_result', 'qualification_contract', 'qualification_driver'):
        item = contract['prerequisites'][key]
        check('proof direct dependency:' + key, proof['inputs_sha256'][item['path']] == item['expected_sha256'])
    check('qualified exact bounded calls', qualified['actual_calls'] == {
        'started': dict(H_build=3, CI_solve=2, gradient=2), 'completed': dict(H_build=3, CI_solve=2, gradient=2)})
    check('same original point/metric and stationarity failure', qualified['point_arrays_sha256'] ==
          contract['fixed_conditions']['point_arrays_sha256'] and qualified['original_metric_sha256'] ==
          contract['fixed_conditions']['original_metric_sha256'] and qualified['original_stationarity_passed'] is False)

    # This is metadata only: no synthetic execution contract is written or used
    # to run a worker. Each candidate must fail before any scientific import.
    expected = dict(execution_authorized_by_root=True, paired_chain_performance_authorized=True,
        complete531_qualification_required=True, exclusive_resource_window_authorized=True,
        root_physics_diagnostic_completed=True, driver_sha256=driver.sha(HERE / 'run.py'),
        preparation_contract_sha256=driver.sha(HERE / 'contract.json'),
        candidate_source_map_sha256=contract['candidate_source_map_sha256'],
        max_public_H_builds=21, max_CI_solves=14, max_full_gradients=14, max_optimizer_calls=0)
    rejects('missing root seal rejects before science', lambda: driver.execution_admission(contract, {}))
    for key, value in expected.items():
        wrong = {**expected, key: False if type(value) is bool else 'wrong'}
        rejects('wrong exact execution field:' + key, lambda wrong=wrong: driver.execution_admission(contract, wrong))
    rejects('integer cannot substitute explicit boolean root authorization', lambda: driver.execution_admission(
        contract, {**expected, 'execution_authorized_by_root': 1}))

    scientific_metadata = {'negative_terms': np.int64(1219), 'energy': np.float64(-1.25),
        'pass': np.bool_(True), 'physical_targets': ((np.int64(3), -1, ((np.int64(0), np.float64(.2)),)),),
        'huge_integer': np.uint64(2**64 - 1)}
    native = driver.plain(scientific_metadata)
    check('nested signed count/scalar/tuple normalization', native == {
        'negative_terms': 1219, 'energy': -1.25, 'pass': True,
        'physical_targets': [[3, -1, [[0, .2]]]], 'huge_integer': 2**64 - 1})
    check('all JSON numbers have native Python type', type(native['negative_terms']) is int
          and type(native['energy']) is float and type(native['pass']) is bool)
    with tempfile.TemporaryDirectory(prefix='metadata_control_', dir=HERE) as temporary:
        target = Path(temporary) / 'roundtrip.json'
        driver.atomic_json(target, scientific_metadata)
        check('atomic fsync JSON roundtrip NumPy int64 failure fixed', driver.load(target) == native)
        rejects('nonfinite JSON metadata fails', lambda: driver.atomic_json(target, {'energy': np.float64(np.inf)}))
    rejects('non-string JSON key rejects', lambda: driver.plain({np.int64(1): 'bad'}))
    rejects('vector ndarray belongs to NPZ', lambda: driver.plain(np.array([1.])))
    rejects('zero-dimensional ndarray also belongs to NPZ', lambda: driver.plain(np.array(1.)))
    rejects('unsupported scalar rejects', lambda: driver.plain(np.complex128(1 + 2j)))

    qtree = ast.parse((ROOT / contract['prerequisites']['qualification_driver']['path']).read_text())
    helper = next(n for n in qtree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_point')
    check('bound point helper unchanged AST', hashlib.sha256(ast.dump(helper, include_attributes=False).encode()).hexdigest()
          == contract['qualification_prepare_point_AST_sha256'])
    for path in (HERE / 'run.py', HERE / 'verify_preparation.py'):
        tree = ast.parse(path.read_text())
        check('AST syntax:' + path.name, isinstance(tree, ast.Module))
    caps = contract['resources']['worker_caps']
    check('all worker budgets sum21H14CI14g', {name: sum(row[name] for row in caps.values())
          for name in ('H', 'CI', 'gradient')} == dict(H=21, CI=14, gradient=14))
    check('explicit cold2 warmup2 warm10 chains14', contract['resources']['cold_processes'] == 2
          and contract['resources']['explicit_warmup_chains'] == 2
          and contract['resources']['measured_warm_pairs'] == 5
          and contract['resources']['measured_warm_chains'] == 10
          and contract['resources']['total_chains'] == 14)
    check('no Atomx imported / no physics executed', not any(n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    check('readonly metadata bounded resources', time.perf_counter() - started < 30
          and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1024 * 1024
          and len(os.sched_getaffinity(0)) == 1)
    driver.require_checks(checks)
    result = dict(all_passed=True, preparation_only=True, check_count=len(checks), checks=checks,
        inputs_sha256=contract['inputs_sha256'], contract_sha256=driver.sha(HERE / 'contract.json'),
        driver_sha256=driver.sha(HERE / 'run.py'), verifier_sha256=driver.sha(__file__),
        source_map_sha256=contract['candidate_source_map_sha256'],
        qualification_result_sha256=contract['prerequisites']['qualification_result']['expected_sha256'],
        qualification_verification_sha256=contract['prerequisites']['qualification_verification']['expected_sha256'],
        H_builds=0, CI_solves=0, gradients=0, optimizer_calls=0, Atomx_imports=0,
        seconds=time.perf_counter() - started,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        scientific_performance_execution_authorized=False)
    driver.atomic_json(HERE / 'readonly_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'inputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
