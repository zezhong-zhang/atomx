"""Read back all retained native pencils/states, with no atomic re-solves."""
import hashlib
import json
import os
import tarfile
from pathlib import Path


def main():
    os.sched_setaffinity(0, {27})
    import numpy as np
    from scipy import linalg

    here = Path(__file__).resolve().parent
    root = next(p for p in here.parents if (p/'pyproject.toml').is_file())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(status='RUNNING', controls=[], roots=[])
    def gate(name, value, cap=None):
        passed = bool(value) if cap is None else bool(value <= cap)
        report['controls'].append(dict(name=name, measured=value, cap=cap, passed=passed))
        if not passed:
            raise AssertionError(name+': '+str(value))
    try:
        source = json.loads((here/'source003.json').read_text())['all_atomx_python_sha256']
        gate('actual359_source_files', len(source)==359)
        gate('source_bytes_unchanged', all(sha(root/p)==h for p,h in source.items()))
        with tarfile.open(here/'source003.tar.gz') as archive:
            gate('archive_whole_census', sorted(archive.getnames())==sorted(source))
            gate('archive_all_source_bytes', all(hashlib.sha256(archive.extractfile(p).read()).hexdigest()==h for p,h in source.items()))
        contract = json.loads((here/'contract003.json').read_text())
        gate('all_frozen_input_hashes', all(sha(root/p)==h for p,h in contract['inputs'].items()))
        actual = json.loads((here/'actual003/result.json').read_text())
        gate('actual_all120_live_checks', actual['status']=='PASS_NATIVE_DHF_BASELINE_ONLY' and len(actual['controls'])==120 and all(x['passed'] for x in actual['controls']))
        wide = np.longdouble
        for ns in (180,220):
            with np.load(here/f'actual003/blocks_ns{ns}.npz',allow_pickle=False) as a:
                gate(f'{ns}:no_object_fields', all(a[k].dtype.kind!='O' for k in a))
                keep = a['active_coordinates']
                for k in (-4,-3,-2,-1,1,2,3):
                    h,s,c = [a[f'{name}_k{k}'] for name in ('H','S','occupied')]
                    gate(f'{ns}:core_census:{k}', c.shape[1]=={-4:1,-3:2,-2:4,-1:5,1:4,2:2,3:1}[k])
                    gram = c.astype(wide).T@s.astype(wide)@c.astype(wide)
                    gate(f'{ns}:core_metric:{k}',float(np.max(abs(gram-np.eye(c.shape[1])))),1e-10)
                    if k not in (-3,-1,2):
                        continue
                    b = a[f'complement_k{k}']
                    e = a[f'positive_E_k{k}']
                    v = a[f'positive_C_k{k}'][keep]
                    gate(f'{ns}:all_columns_retained:{k}',len(e)==len(a[f'raw_positive_E_k{k}']))
                    gram = v.astype(wide).T@s.astype(wide)@v.astype(wide)
                    gate(f'{ns}:all_physical_column_Gram:{k}',float(np.max(abs(gram-np.eye(len(e))))),1e-10)
                    gate(f'{ns}:occupied_orthogonality:{k}',float(np.max(abs(c.astype(wide).T@s.astype(wide)@v.astype(wide)))),1e-10)
                    sp = b.T@s@b
                    sp = np.tril(sp)+np.tril(sp,-1).T
                    r = np.asarray(b.astype(wide).T@(h.astype(wide)@v[:,0].astype(wide)
                        -wide(e[0])*(s.astype(wide)@v[:,0].astype(wide))),dtype=float)
                    dual = float(np.linalg.norm(linalg.solve_triangular(linalg.cholesky(sp,lower=True),r,lower=True)))
                    report['roots'].append(dict(ns=ns,kappa=k,original_saved_pencil_dual_Ha=dual))
                    gate(f'{ns}:original_saved_pencil_root:{k}',dual,1e-8)
                for n in (68,69):
                    h,c,e=[a[f'CI_{name}_N{n}'] for name in ('H','C','E')]
                    gate(f'{ns}:CI_residual:{n}',float(np.linalg.norm(h@c-c*e,axis=0).max()),1e-8)
                    gate(f'{ns}:CI_Gram:{n}',float(np.max(abs(c.T@c-np.eye(len(e))))),1e-10)
        moments=json.loads((here/'static_moments002/result.json').read_text())
        gate('all30_moment_controls',moments['status']=='PASS_STATIC_DHF_MOMENTS_ONLY' and len(moments['controls'])==30 and all(x['passed'] for x in moments['controls']))
        report['status']='PASS_SAVED_DATA_REVIEW'
    except Exception as error:
        report.update(status='FAILED',exception=repr(error))
        raise
    finally:
        (here/'saved_data_review.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(status=report['status'],controls=len(report['controls']),roots=report['roots']),indent=2))


if __name__ == '__main__':
    main()
