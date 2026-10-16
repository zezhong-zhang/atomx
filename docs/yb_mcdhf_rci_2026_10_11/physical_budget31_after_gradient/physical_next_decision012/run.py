"""Independent analytic CI-response curvature on the complete original Yb A/B.

No displaced H or step scan. A fresh native H(0), H'(0), H''(0) are built
on the same mathematical polar path; its rounded zero-point difference from
the literal archived R0 is measured explicitly. No optimizer admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'pyproject.toml').is_file())
BASE = HERE.parent
OLD = BASE/'physical_next_decision005/execution001'
THREADS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
           'NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            h.update(b)
    return h.hexdigest()


def plain(x):
    if isinstance(x,dict):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [plain(v) for v in x]
    return plain(x.tolist()) if hasattr(x,'tolist') else x


def save(path,value):
    Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def deep(x):
    return tuple(map(deep,x)) if isinstance(x,list) else x


def source_map():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/atomx').rglob('*.py'))}


def prepare():
    original = read(BASE/'physical_next_decision006/execution001/contract.json')
    inputs = {v['path']:v['sha256'] for v in original['inputs'].values()}
    paths = [HERE/'run.py',HERE/'test_receipt.json',HERE/'grasp_source_reference.json',
             BASE/'physical_next_decision011/actual001/result.json']
    for arm in ('A','B'):
        paths.extend(OLD/'actual'/(arm+'_R0'+suffix) for suffix in ('.json','.npz'))
        paths.append(OLD/'actual'/(arm+'_R0_state_gradients.json'))
    inputs.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    mapping = source_map()
    identity = hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(HERE/'source_current.json',dict(all_atomx_python_sha256=mapping,source_map_sha256=identity))
    contract = dict(hypothesis='The measured O(h^2) finite CI-response component follows the independent analytic 2n+1 coefficient, rather than a missing first-derivative factor.',
        formula='Q(H0-E0)Q c1=-Q H1 c0, c0.T c1=0; B_CI=c1.T H2 c0+c1.T(H1-E1)c1; finite coefficient-response FD=h^2 B_CI+O(h^4). H1/H2 are derivatives, not Taylor coefficients.',
        fixed=original['fixed'],original=original,source_map_sha256=identity,inputs=inputs,
        scope='Complete original A1595/B2574, five sectors each, all31 waves and3689 original direction coordinates. No full4f12 trial, new displaced H, optimizer, NIST fit or original gate supersession.',
        zero_point='Differentiate the actual mathematical polar path at t=0 including its finite input Gram and nonzero tangency defect. Use its polar0 bank in fresh H0/H1/H2; measure its rounded difference from literal accepted13 separately, never silently equate the two.',
        acceptance={'polar0_minus_saved_wave':1e-12,'polar0_native_map':1e-12,
                    'radial_constraint_derivatives':1e-10,'fresh_H0_minus_saved_H0_Ha':1e-7,
                    'complete_eigensystem_residual_Ha':1e-8,'complete_eigensystem_Gram':1e-10,
                    'target_overlap_squared_min':.999999,'first_action_minus_original_Ha':1e-9,
                    'curvature_prediction_abs_Ha':2e-12,'curvature_prediction_relative':.005,
                    'ordinary_FD_gate':'Original two failed large-step gates remain failed; curvature comparison cannot replace them.'},
        stopping='First identity, scientific-control or resource failure stops, preserving result. No automatic retry or change of h, weights, model, gate or resource cap.',
        limits={'CPU':27,'threads':1,'AS_bytes':8*1024**3,'seconds':600,'load_max':24,
                'available_min_bytes':16*1024**3,'radial_keys':70971,'raw_determinants':2659,
                'exact_block_matrix_calls':30,'complete_eigh_calls':10,'response_calls':10,
                'decoded_input_bytes':256*1024**2,'array_output_bytes':128*1024**2})
    save(HERE/'contract.json',contract)
    print(json.dumps(dict(status='PREPARED_SOURCE_ONLY',source=identity,inputs=len(inputs))))


def execute():
    contract = read(HERE/'contract.json')
    identity = read(HERE/'source_current.json')
    if source_map()!=identity['all_atomx_python_sha256']:
        raise ValueError('current actual source changed')
    for path,h in contract['inputs'].items():
        if sha(ROOT/path)!=h:
            raise ValueError('bound input changed '+path)
    cap = contract['limits']
    available = next(int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines()
                     if r.startswith('MemAvailable:'))
    if (str(Path(sys.executable).absolute())!=str(ROOT/'.venv/bin/python')
            or os.sched_getaffinity(0)!={cap['CPU']} or any(os.environ.get(k)!='1' for k in THREADS)
            or available<cap['available_min_bytes'] or os.getloadavg()[0]>cap['load_max']):
        raise ValueError('resource admission')
    out = HERE/'actual001'
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS,(cap['AS_bytes'],cap['AS_bytes']))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s stop')))
    signal.alarm(cap['seconds'])
    started = time.monotonic()
    checks,states,predictions,counts = [],{},{},{}
    report = dict(status='RUNNING',checks=checks,states=states,predictions=predictions,counts=counts,
                  source_map_sha256=contract['source_map_sha256'],ordinary_FD005_superseded=False,
                  optimizer_admitted=False,experimental_accuracy_qualified=False,host_load=os.getloadavg())
    acceptance = contract['acceptance']

    def check(name,value,limit=None):
        ok = bool(value) if limit is None else bool(value<=limit)
        checks.append(dict(name=name,observed=plain(value),limit=limit,PASS=ok))
        if not ok:
            raise ValueError('gate '+name)

    def count(key,maximum):
        counts[key] = counts.get(key,0)+1
        if counts[key]>maximum:
            raise ValueError('work cap '+key)

    try:
        import numpy as np

        from atomx.angular import csf_hamiltonian as angular
        from atomx.methods.ci import _centered_dense_eigh
        from atomx.numerics.bspline import orbital_optimization as opt
        from atomx.numerics.variational_response import (
            polar_path_derivatives,
            real_eigenstate_response,
        )
        from atomx.physics.csf import matrix
        from atomx.physics.csf.effective_one_body import CIReferenceBinding
        from atomx.physics.csf.radial_increment import bspline_dc_radial_derivatives
        from atomx.potentials.nuclear import make_nuclear_potential

        original = contract['original']
        decoded = 0

        def load(path,fields):
            nonlocal decoded
            with np.load(path,allow_pickle=False) as z:
                arrays = {k:z[k] for k in fields}
            decoded += sum(v.nbytes for v in arrays.values())
            check('decoded input budget '+str(decoded),decoded<=cap['decoded_input_bytes'])
            check('finite arrays '+str(decoded),all(np.isfinite(v).all() for v in arrays.values()))
            return arrays

        def archive(role):
            return load(ROOT/original['inputs'][role]['path'],original['arrays'][role]['fields'])

        geometry,cp,sm = archive('workspace'),archive('checkpoint_arrays'),archive('saved_S')
        data = {k[6:]:v.item() if v.ndim==0 else v for k,v in geometry.items() if k.startswith('basis_')}
        old_meta = read(OLD/'actual/A_R0.json')
        banks = load(OLD/'actual/A_R0.npz',('keys','P','Q'))
        keys = tuple(map(tuple,banks['keys'].tolist()))
        waves = {k:(p,q) for k,p,q in zip(keys,banks['P'],banks['Q'],strict=True)}
        nucleus = make_nuclear_potential(70,'fermi',A=175)
        binding = plain(asdict(CIReferenceBinding.capture([(*waves[k],k[1]) for k in keys],{},data,70,123,nucleus)))
        check('original literal radial/nuclear binding',binding==old_meta['owner']['radial_binding'])
        layout = tuple((tuple(map(tuple,r['keys'])),cp['projector_'+str(i)],tuple(r['shape']),slice(*r['slice']))
                       for i,r in enumerate(read(ROOT/original['inputs']['layout']['path'])))
        inverse = tuple(tuple(cp['inverse_blocks_'+str(i)]) for i in range(7))
        check('original residual metric',opt._residual_metric_sha256(layout,cp['inverse_diagonal'],inverse)
              ==original['fixed']['metric_sha256'])
        direction = -cp['full_normalized_pg']
        check('complete direction',direction.shape==(3689,) and len(keys)==31)
        jets = [{},{},{}]
        for group,projector,shape,where in layout:
            y,d = cp['x'][where].reshape(shape),direction[where].reshape(shape)
            polar = polar_path_derivatives(y,d)
            check('native polar0 '+str(group[0][1]),float(np.max(abs(polar[0]-opt._polar_retraction(y)))),acceptance['polar0_native_map'])
            p,p1,p2 = (projector@a for a in polar)
            metric = sm['s_'+str(group[0][1])]
            check('first constraint '+str(group[0][1]),float(np.max(abs(p.T@metric@p1+p1.T@metric@p))),acceptance['radial_constraint_derivatives'])
            check('second constraint '+str(group[0][1]),float(np.max(abs(p.T@metric@p2+p2.T@metric@p+2*p1.T@metric@p1))),acceptance['radial_constraint_derivatives'])
            for order,array in enumerate((p,p1,p2)):
                for j,k in enumerate(group):
                    jets[order][k] = array[:123,j].copy(),array[123:,j].copy()
        discrepancy = max(float(np.max(abs(jets[0][k][s]-waves[k][s]))) for k in keys for s in (0,1))
        check('polar0 minus literal saved bank',discrepancy,acceptance['polar0_minus_saved_wave'])
        report['polar0_minus_saved_wave_max'] = discrepancy
        derivative = bspline_dc_radial_derivatives(*jets,data,70,123,nuclear_potential=nucleus)
        value = matrix._radial_evaluator(jets[0],data,70,123,nuclear_potential=nucleus)

        @lru_cache(None)
        def values(key):
            count('radial_keys',cap['radial_keys'])
            return float(value(key)),*derivative.components(key)

        comparison = read(BASE/'physical_next_decision011/actual001/result.json')['comparisons']
        output_bytes = 0
        for arm in ('A','B'):
            meta = read(OLD/'actual'/(arm+'_R0.json'))
            check('same complete basis shape '+arm,len(meta['owner']['CSFs'])==(1595 if arm=='A' else 2574))
            old = load(OLD/'actual'/(arm+'_R0.npz'),('H','mixing','energies'))
            gradients = read(OLD/'actual'/(arm+'_R0_state_gradients.json'))['states']
            states[arm] = {}
            for row in meta['rows']:
                uid,label = row['UID'],row['label']
                indices = np.array([i for i,s in enumerate(meta['owner']['CSFs']) if (s[2][1],s[1])==tuple(uid[:2])])
                signatures = [deep(meta['owner']['CSFs'][i][2]) for i in indices]
                operators = []
                for order in range(3):
                    count('exact_block_matrix_calls',cap['exact_block_matrix_calls'])
                    operators.append(angular.exact_block_matrix(signatures,lambda key,order=order:values(key)[order],max_determinants=2000 if arm=='A' else 2659))
                h,h1,h2 = operators
                check('fresh polar0 H matches literal R0 '+arm+label,float(np.max(abs(h-old['H'][np.ix_(indices,indices)]))),acceptance['fresh_H0_minus_saved_H0_Ha'])
                count('complete_eigh_calls',cap['complete_eigh_calls'])
                e,v,origin = _centered_dense_eigh(h,driver='evd')
                check('complete residual '+arm+label,float(np.max(abs(h@v-v*e))),1e-8)
                check('complete Gram '+arm+label,float(np.max(abs(v.T@v-np.eye(len(e))))),1e-10)
                overlaps = abs(v.T@old['mixing'][indices,row['column']])**2
                root = int(np.argmax(overlaps))
                check('literal root and overlap '+arm+label,root==row['selected_level'][2] and overlaps[root]>.999999)
                count('response_calls',cap['response_calls'])
                response = real_eigenstate_response(h,h1,h2,e,v,root)
                check('first action '+arm+label,abs(response.first-gradients[label]['action_Ha']),1e-9)
                states[arm][label] = dict(UID=uid,root=root,dimension=len(e),origin_Ha=origin,
                    first_Ha=response.first,second_Ha=response.second,
                    CI_third_coefficient_Ha=response.ci_third_coefficient,
                    nearest_gap_Ha=response.nearest_gap,response_residual_Ha=response.response_residual,
                    gauge_defect=response.gauge_defect,overlap_squared=float(overlaps[root]))
                arrays = dict(H0=h,H1=h1,H2=h2,energies=e,vectors=v,
                              c1=response.coefficient_derivative,rows=indices)
                output_bytes += sum(x.nbytes for x in arrays.values())
                check('array output budget',output_bytes<=cap['array_output_bytes'])
                target = out/(arm+'_'+label.replace('/','_')+'.npz')
                np.savez_compressed(target,**arrays)
                states[arm][label]['arrays_sha256'] = sha(target)
                save(out/'partial.json',report)
            for label,uid in original['UIDs'][1:]:
                coefficient = states[arm][label]['CI_third_coefficient_Ha']-states[arm][original['UIDs'][0][0]]['CI_third_coefficient_Ha']
                for step in (.002,.001):
                    measured = next(r['finite_CI_response_Ha'] for r in comparison if r['arm']==arm and r['label']==label and r['kind']=='line' and r['step']==step)
                    predicted = step**2*coefficient
                    error = abs(predicted-measured)
                    limit = max(acceptance['curvature_prediction_abs_Ha'],acceptance['curvature_prediction_relative']*abs(measured))
                    check('analytic response curvature '+arm+label+str(step),error,limit)
                    predictions[arm+'/'+label+'/'+str(step)] = dict(UID=uid,step=step,coefficient_Ha=coefficient,
                        predicted_Ha=predicted,measured_Ha=measured,error_Ha=error,limit_Ha=limit)
            del old
        check('exact complete work',all(counts[k]==cap[k] for k in ('exact_block_matrix_calls','complete_eigh_calls','response_calls')))
        check('unchanged source after',source_map()==identity['all_atomx_python_sha256'])
        for path,h in contract['inputs'].items():
            check('unchanged input '+path,sha(ROOT/path)==h)
        report.update(status='PASS_analytic_CI_response_curvature_original_FD_failures_preserved',
                      decoded_input_bytes=decoded,array_output_bytes=output_bytes)
    except BaseException as exc:
        report.update(status='FAILED_ANALYTIC_CURVATURE_PRESERVED',error=repr(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save(out/'result.json',report)
        print(json.dumps({k:report[k] for k in ('status','counts','elapsed_seconds','peak_RSS_MiB')}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if args.prepare==args.execute:
        raise SystemExit('choose exactly one mode')
    prepare() if args.prepare else execute()
