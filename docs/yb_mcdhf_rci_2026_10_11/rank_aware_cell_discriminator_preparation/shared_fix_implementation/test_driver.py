"""Exactly three authorized shared positive-rank analytic tests, one run."""
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    with path.open('w') as f:
        f.write(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    begin = time.monotonic()
    seal = json.loads((HERE/'execution_contract.json').read_text())
    identity = json.loads((HERE/'source_identity.json').read_text())
    if not (seal['test_execution_authorized_by_root'] is True
            and seal['driver_sha256']==sha(Path(__file__))
            and seal['source_identity_sha256']==sha(HERE/'source_identity.json')
            and seal['source_map_sha256']==identity['source_map_sha256']
            and seal['CPU']==30 and seal['memory_GiB']==3.95
            and seal['inner_seconds']==60 and seal['outer_seconds']==65):
        raise ValueError('Exact authorized test seal required')
    for variable in seal['thread_variables']:
        if os.environ.get(variable)!='1':
            raise ValueError('Unpinned numerical thread')
    if os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD')!='1':
        raise ValueError('Plugin autoload forbidden')
    if os.environ.get('NUMBA_CACHE_DIR')!=str(HERE/'actual/numba_cache'):
        raise ValueError('Fresh scoped NumBa cache required')
    if 30 not in os.sched_getaffinity(0):
        raise ValueError('CPU30 unavailable')
    resource.setrlimit(resource.RLIMIT_AS,(int(3.95*2**30),)*2)
    os.sched_setaffinity(0,{30})
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s test cap')))
    signal.alarm(60)
    output=HERE/'actual'
    started,completed,reports={}, {}, []
    def source_guard():
        for name,digest in identity['all_atomx_python_sha256'].items():
            if sha(REPO/name)!=digest:
                raise ValueError('Changed shared source')
        if sha(REPO/'tests/test_convol.py')!=seal['test_file_sha256']:
            raise ValueError('Changed selected test')
    def flush():
        write(output/'progress.json',{'started':started,'completed':completed,
              'test_reports':reports,'elapsed_uncontrolled_s':time.monotonic()-begin})
    source_guard()
    sys.path.insert(0,str(REPO/'src'))
    import numpy as np
    import pytest
    import scipy.linalg

    import atomx.methods.ci as ci
    import atomx.methods.csf.variational as variational
    import atomx.numerics.bspline.basis as basis
    import atomx.numerics.bspline.convol as convol
    import atomx.numerics.bspline.dirac_assembly as dirac
    import atomx.numerics.bspline.orbital_optimization as optimizer
    import atomx.physics.csf.matrix as matrix

    def wrap(name,function):
        def call(*args,**kwargs):
            started[name]=started.get(name,0)+1
            if started[name]>seal['exact_calls'][name]:
                flush()
                raise RuntimeError('Budget '+name)
            value=function(*args,**kwargs)
            completed[name]=completed.get(name,0)+1
            return value
        return call
    basis.prepare_bspline_data=wrap('basis_setup',basis.prepare_bspline_data)
    np.linalg.eigvalsh=wrap('Gauss_setup_eigvalsh',np.linalg.eigvalsh)
    np.linalg.eigh=wrap('numpy_eigh',np.linalg.eigh)
    scipy.linalg.eigh=wrap('scipy_eigh',scipy.linalg.eigh)
    for name,label in (('_density_sym','density_band'),('_convol_case1_t','full_convolutions'),
                       ('_convol_case1_semiseparable_t','factor_convolutions'),('_sum_amb_banded','band_scalars')):
        setattr(convol,name,wrap(label,getattr(convol,name)))
    convol.ConvolCache.get_rkb_t=wrap('bank_get',convol.ConvolCache.get_rkb_t)
    convol.ConvolCache.get_coulomb_factors=wrap('factor_get',convol.ConvolCache.get_coulomb_factors)
    dirac.build_dirac_matrices=wrap('hS',dirac.build_dirac_matrices)
    matrix.build_dirac_matrices=dirac.build_dirac_matrices
    variational.build_dirac_matrices=dirac.build_dirac_matrices
    dirac.solve_dirac=wrap('Dirac_solve',dirac.solve_dirac)
    matrix.build_ci_matrix=wrap('H',matrix.build_ci_matrix)
    ci.ci_solve=wrap('CI',ci.ci_solve)
    variational.ci_solve=ci.ci_solve
    variational.CSFOrbitalFunctional.gradient_from_weights=wrap(
        'gradient',variational.CSFOrbitalFunctional.gradient_from_weights)
    optimizer.optimize_orbitals=wrap('optimizer',optimizer.optimize_orbitals)

    class Record:
        def pytest_collection_modifyitems(self,items):
            if len(items)!=3 or any(not i.name.startswith('test_positive_rank_origin_cell_signed_components') for i in items):
                raise ValueError('Exactly authorized three cases required')
        def pytest_runtest_logreport(self,report):
            if report.when=='call':
                reports.append({'nodeid':report.nodeid,'outcome':report.outcome})
                flush()
    try:
        exit_code=int(pytest.main(['-q','-x','tests/test_convol.py::test_positive_rank_origin_cell_signed_components'],plugins=[Record()]))
        counts_ok=all(started.get(k,0)==completed.get(k,0)==n for k,n in seal['exact_calls'].items())
        source_guard()
        tests_ok=exit_code==0 and len(reports)==3 and all(x['outcome']=='passed' for x in reports)
        write(output/'result.json',{'tests_passed':tests_ok,'exact_call_budget_passed':counts_ok,
              'started':started,'completed':completed,'reports':reports,'pytest_exit_code':exit_code,
              'source_map_sha256':identity['source_map_sha256'],'driver_sha256':sha(Path(__file__)),
              'test_file_sha256':seal['test_file_sha256'],'GPU_calls':0,'H_CI_gradient_Dirac_optimizer':0,
              'elapsed_uncontrolled_seconds':time.monotonic()-begin,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        flush()
        if not (tests_ok and counts_ok):
            raise RuntimeError('Closed test or budget failure')
    except BaseException as exc:
        write(output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'started':started,'completed':completed,'reports':reports,'no_retry':True})
        flush()
        raise


if __name__=='__main__':
    main()
