"""Qualify the existing native centered dense helper on literal saved005 H."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())
spec = importlib.util.spec_from_file_location('exact_dyadic_011', HERE/'run.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def prepare():
    parent = audit.read(HERE/'contract.json')
    inputs = dict(parent['inputs'])
    for name in ('run.py','contract.json','actual001/result.json','test_receipt.json'):
        inputs[str((HERE/name).relative_to(ROOT))] = audit.sha(HERE/name)
    audit.save(HERE/'centered_contract.json',dict(
        hypothesis='Existing current native integer-centered full dense solver removes the large-origin root error at the literal005 H; physical finite-step CI response remains.',
        fixed='All ten saved005 complete H, all five original full symmetry sectors and target UIDs, original steps/direction/analytic gates; no new radial/H/gradient or optimizer.',
        representation='Current atomx.methods.ci._centered_dense_eigh with complete driver=evd, exactly as current ci_solve; no physical diagonal shift. Never construct CIResult from archived arrays.',
        acceptance={'full_residual_abs_Ha':1e-8,'full_Gram_abs':1e-10,
                    'matching':'same original local root, maximum saved-column overlap squared>0.999999; no energy/NIST selection',
                    'immutable':'matrix owner frame hash before/after',
                    'exact_Rayleigh_check':'unmodified literal H and new signed vectors, dyadic norm/quadratic',
                    'ordinary_FD':'retain all original state/line/EOL gates, outcomes may FAIL'},
        stopping='First identity/full-spectrum/residual/matching/resource failure stops; no automatic retry or gate/step/model changes.',
        limits={'CPU':27,'AS_bytes':4*1024**3,'seconds':180,'NPZ_loads':10,
                'eigh_calls':50,'quadratic_calls':50,'output_array_bytes_max':128*1024**2,
                'load_max':24,'available_min_bytes':8*1024**3},
        inputs=inputs,driver_sha256=audit.sha(__file__), original005_superseded=False))
    print('PREPARED native centered replay; no numerical imports or solves')


def execute():
    contract = audit.read(HERE/'centered_contract.json')
    if audit.sha(__file__) != contract['driver_sha256']:
        raise ValueError('changed centered driver')
    for path,digest in contract['inputs'].items():
        if audit.sha(ROOT/path) != digest:
            raise ValueError('changed input '+path)
    current = audit.read(audit.LATEST/'current_source_identity.json')
    for path,digest in current['all_atomx_python_sha256'].items():
        if audit.sha(ROOT/path) != digest:
            raise ValueError('changed current source '+path)
    cap = contract['limits']
    available = next(int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines()
                     if r.startswith('MemAvailable:'))
    if (str(Path(sys.executable).absolute()) != str(ROOT/'.venv/bin/python')
            or os.sched_getaffinity(0) != {cap['CPU']}
            or any(os.environ.get(k) != '1' for k in audit.THREADS)
            or available < cap['available_min_bytes'] or os.getloadavg()[0] > cap['load_max']):
        raise ValueError('resource admission')
    out = HERE/'centered001'
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS,(cap['AS_bytes'],cap['AS_bytes']))
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(TimeoutError('180s centered stop')))
    signal.alarm(cap['seconds'])
    started = time.monotonic()
    checks, points, outcomes = [], {}, []
    counts = collections.Counter()
    report = dict(status='RUNNING',checks=checks,points=points,outcomes=outcomes,
                  source_map_sha256=current['source_map_sha256'],original005_superseded=False,
                  optimizer_admitted=False,new_radial_H_gradient_calls=0,host_load=os.getloadavg())

    def check(name,value):
        checks.append(dict(name=name,PASS=bool(value)))
        if not value:
            raise ValueError('gate '+name)

    try:
        import numpy as np

        from atomx.methods import ci

        check('actual native CI source',Path(ci.__file__).resolve()==ROOT/'src/atomx/methods/ci.py')
        original_eigh = ci.linalg.eigh

        def observed_eigh(*args,**kw):
            counts['eigh_calls'] += 1
            check('eigh budget',counts['eigh_calls'] <= cap['eigh_calls'])
            check('complete evd',kw.get('driver')=='evd' and kw.get('subset_by_index') is None)
            return original_eigh(*args,**kw)

        ci.linalg.eigh = observed_eigh
        old_contract = audit.read(audit.OLD/'contract.json')
        output_bytes = 0
        exact_values = {}
        for arm in ('A','B'):
            exact_values[arm] = {}
            for tag in ('R0','plus_h','minus_h','plus_half','minus_half'):
                name = arm+'_'+tag
                meta = audit.read(audit.OLD/'actual'/(name+'.json'))
                n = 1595 if arm=='A' else 2574
                with np.load(audit.OLD/'actual'/(name+'.npz'),allow_pickle=False) as z:
                    h, old_vectors = z['H'],z['mixing']
                counts['NPZ_loads'] += 1
                check('literal complete arrays '+name,h.shape==old_vectors.shape==(n,n)
                      and h.dtype==old_vectors.dtype==np.float64 and np.isfinite(h).all()
                      and np.isfinite(old_vectors).all() and np.array_equal(h,h.T))
                frame = repr((h.shape,h.dtype.str)).encode()+h.tobytes(order='C')
                check('same full H owner '+name,hashlib.sha256(frame).hexdigest()==meta['owner_matrix_sha256'])
                h.flags.writeable = False
                labels = [(s[2][1],s[1]) for s in meta['owner']['CSFs']]
                sectors = list(dict.fromkeys(labels))
                selected = {tuple(row['UID'][:2]):row for row in meta['rows']}
                check('all five sectors '+name,len(sectors)==5 and set(sectors)==set(selected))
                saved_arrays, block_reports, roots, all_values, records = {}, [], {}, [], []
                exact_values[arm][tag] = {}
                for index,sector in enumerate(sectors):
                    ix = np.array([i for i,s in enumerate(labels) if s==sector])
                    block = h[np.ix_(ix,ix)]
                    before = hashlib.sha256(block.tobytes()).hexdigest()
                    values,vectors,origin = ci._centered_dense_eigh(block,driver='evd')
                    check('unchanged block '+name+str(sector),hashlib.sha256(block.tobytes()).hexdigest()==before)
                    residual = float(np.max(abs(block@vectors-vectors*values)))
                    gram = float(np.max(abs(vectors.T@vectors-np.eye(len(ix)))))
                    check('full residual '+name+str(sector),residual<=1e-8)
                    check('full Gram '+name+str(sector),gram<=1e-10)
                    row = selected[sector]
                    previous = old_vectors[ix,row['column']]
                    overlaps = abs(vectors.T@previous)**2
                    local = int(np.argmax(overlaps))
                    check('literal root and signed projector '+name+str(sector),
                          local==row['selected_level'][2] and overlaps[local]>.999999)
                    q,norm = audit.exact_quadratic(block,vectors[:,local],Fraction())
                    counts['quadratic_calls'] += 1
                    label = row['label']
                    exact_values[arm][tag][label] = q
                    roots[label] = dict(UID=row['UID'],local_root=local,energy_Ha=float(values[local]),
                        exact_Rayleigh_Ha=float(q),reported_minus_Rayleigh_Ha=float(Fraction.from_float(float(values[local]))-q),
                        overlap_squared=float(overlaps[local]),actual_norm=float(norm),origin_Ha=origin)
                    block_reports.append(dict(sector=list(sector),dimension=len(ix),residual_max_Ha=residual,Gram_max=gram))
                    for j,value in enumerate(values):
                        records.append((label if j==local else None,len(all_values)))
                        all_values.append(float(value))
                    saved_arrays['rows_'+str(index)] = ix
                    saved_arrays['values_'+str(index)] = values
                    saved_arrays['vectors_'+str(index)] = vectors
                order = np.argsort(all_values,kind='stable')
                inverse_order = np.empty(n,dtype=int)
                inverse_order[order] = np.arange(n)
                weights = np.zeros(n)
                for label,column in records:
                    if label is not None:
                        weights[inverse_order[column]] = .2
                callback = float(weights@np.asarray(all_values)[order])
                output_bytes += sum(a.nbytes for a in saved_arrays.values())
                check('output size '+name,output_bytes<=cap['output_array_bytes_max'])
                target = out/(name+'.npz')
                np.savez_compressed(target,**saved_arrays)
                points[name] = dict(states=roots,EOL_Ha=callback,blocks=block_reports,
                                    full_eigenblock_arrays_sha256=audit.sha(target))
                check('whole H unchanged '+name,hashlib.sha256(repr((h.shape,h.dtype.str)).encode()+h.tobytes(order='C')).hexdigest()==meta['owner_matrix_sha256'])
                del h,old_vectors,saved_arrays
            gradients = audit.read(audit.OLD/'actual'/(arm+'_R0_state_gradients.json'))
            for plus,minus,step in (('plus_h','minus_h',.002),('plus_half','minus_half',.001)):
                slopes = {}
                for label,_uid in old_contract['UIDs']:
                    ep = Fraction.from_float(points[arm+'_'+plus]['states'][label]['energy_Ha'])
                    em = Fraction.from_float(points[arm+'_'+minus]['states'][label]['energy_Ha'])
                    slopes[label] = (ep-em)/(2*Fraction.from_float(step))
                ground = old_contract['UIDs'][0][0]
                for kind in ('state','line'):
                    for label,uid in old_contract['UIDs']:
                        if kind=='line' and label==ground:
                            continue
                        analytic = gradients['states'][label]['action_Ha']
                        slope = slopes[label]
                        if kind=='line':
                            analytic -= gradients['states'][ground]['action_Ha']
                            slope -= slopes[ground]
                        tolerance = max(1e-8,1e-5*abs(analytic))
                        error = float(slope-Fraction.from_float(analytic))
                        outcomes.append(dict(arm=arm,step=step,kind=kind,label=label,UID=uid,
                                             analytic_Ha=analytic,central_Ha=float(slope),signed_error_Ha=error,
                                             tolerance_Ha=tolerance,PASS=abs(error)<=tolerance))
                slope = (Fraction.from_float(points[arm+'_'+plus]['EOL_Ha'])-
                         Fraction.from_float(points[arm+'_'+minus]['EOL_Ha']))/(2*Fraction.from_float(step))
                analytic = gradients['EOL_action_Ha']
                tolerance = max(1e-8,1e-5*abs(analytic))
                error = float(slope-Fraction.from_float(analytic))
                outcomes.append(dict(arm=arm,step=step,kind='EOL',label='EOL',analytic_Ha=analytic,
                                     central_Ha=float(slope),signed_error_Ha=error,tolerance_Ha=tolerance,
                                     PASS=abs(error)<=tolerance))
        check('complete work',all(counts[k]==cap[k] for k in ('NPZ_loads','eigh_calls','quadratic_calls')))
        for path,digest in contract['inputs'].items():
            check('unchanged input '+path,audit.sha(ROOT/path)==digest)
        for path,digest in current['all_atomx_python_sha256'].items():
            if audit.sha(ROOT/path)!=digest:
                raise ValueError('source changed during replay')
        report['status'] = 'COMPLETE_current_native_centered_replay_original005_preserved'
        report['ordinary_FD_passed'] = all(r['PASS'] for r in outcomes)
        report['output_array_bytes'] = output_bytes
    except BaseException as exc:
        report.update(status='FAILED_CENTERED_REPLAY_PRESERVED',error=repr(exc))
        raise
    finally:
        report.update(calls=dict(counts),elapsed_seconds=time.monotonic()-started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        audit.save(out/'result.json',report)
        print(json.dumps({k:report[k] for k in ('status','calls','elapsed_seconds','peak_RSS_MiB')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if args.prepare == args.execute:
        raise SystemExit('choose exactly one mode')
    prepare() if args.prepare else execute()
