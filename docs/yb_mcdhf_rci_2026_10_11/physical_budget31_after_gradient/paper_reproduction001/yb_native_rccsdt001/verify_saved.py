"""Independently read the complete saved equations, states and phase spans."""
import hashlib
import itertools
import json
import os
from pathlib import Path


def main():
    os.sched_setaffinity(0,{27})
    import numpy as np

    here=Path(__file__).resolve().parent
    root=next(p for p in here.parents if (p/'pyproject.toml').is_file())
    out=here/'actual001'
    result=json.loads((out/'result.json').read_text())
    operator=json.loads((out/'operator.json').read_text())
    report=dict(status='RUNNING',controls=[],target_errors=[])
    def gate(name,value,cap=None):
        passed=bool(value) if cap is None else bool(value<=cap)
        report['controls'].append(dict(name=name,measured=value,cap=cap,passed=passed))
        if not passed:
            raise AssertionError(name+': '+str(value))
    try:
        def sha(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()
        source=json.loads((here/'source.json').read_text())['files']
        gate('actual_source_bytes',all(sha(root/p)==h for p,h in source.items()))
        contract=json.loads((here/'contract.json').read_text())
        gate('all_contract_input_bytes',all(sha(root/p)==h for p,h in contract['inputs'].items()))
        gate('all44_live_controls',result['status']=='PASS_BOUNDED_NATIVE_YB_CCSDT_PILOT' and len(result['controls'])==44 and all(c['passed'] for c in result['controls']))
        with np.load(out/'integrals.npz',allow_pickle=False) as z:
            gate('operator_exactly_real',all(not np.any(z[k].imag) for k in z))
        with np.load(out/'independent_FCI.npz',allow_pickle=False) as z:
            hc,ha,ec,vc,ev,vv,jc,ja=[z[k] for k in ('core_H','attached_H','core_E','core_C','attached_E','attached_C','core_J2','attached_J2')]
            core_masks=tuple(map(int,z['core_masks']))
            attached_masks=tuple(map(int,z['attached_masks']))
        gate('complete210_core_masks',set(core_masks)=={sum(1<<i for i in c) for c in itertools.combinations(range(10),4)})
        gate('complete252_attached_masks',set(attached_masks)=={sum(1<<i for i in c) for c in itertools.combinations(range(10),5)})
        for label,h,e,c in [('core',hc,ec,vc),('attached',ha,ev,vv)]:
            gate(label+':FCI_residual',float(np.linalg.norm(h@c-c*e,axis=0).max()),1e-10)
            gate(label+':FCI_Gram',float(np.max(abs(c.T@c-np.eye(len(e))))),1e-10)
        lookup={mask:i for i,mask in enumerate(attached_masks)}
        origin=operator['absolute_energy_origin_Ha']
        for arm in result['arms']:
            label=arm['label']
            with np.load(out/(label+'.npz'),allow_pickle=False) as z:
                e,c,h,v,core_c,full=[z[k] for k in ('EA','physical_C','projected_H','projected_C','core_C','full_residuals')]
                rows=[lookup[int(mask)] for mask in z['projected_masks']]
            gate(label+':projected_right_equations',float(np.max(abs(h@v-v*e))),1e-10)
            gate(label+':all_physical_norms',float(np.max(abs(np.linalg.norm(c,axis=0)-1))),1e-10)
            gate(label+':core_J0',float(np.linalg.norm(jc@core_c)),1e-7)
            actual=np.max(abs(ha@c-c*(arm['core_energy_relative_Ha']+e)),axis=0)
            gate(label+':full_residual_archive',float(np.max(abs(actual-full))),1e-10)
            gate(label+':complete_projection_count',len(rows)==(66 if label.endswith('R2') else 186))
            for name,target in arm['levels'].items():
                j=.5 if name=='S' else 1.5
                roots=target['roots']
                gate(label+':complete_J_multiplet:'+name,len(roots)==(2 if name=='S' else 4))
                gate(label+':J2:'+name,float(np.linalg.norm(ja@c[:,roots]-j*(j+1)*c[:,roots],axis=0).max()),1e-7)
                exact=result['FCI']['levels'][name]['mean_energy_Ha']
                total=arm['core_energy_relative_Ha']+target['mean_energy_Ha']
                report['target_errors'].append(dict(label=label,target=name,
                    total69_relative_Ha=total,total69_absolute_Ha=origin+total,
                    total69_minus_FCI_Ha=total-exact,
                    EA_minus_FCI_Ha=target['mean_energy_Ha']-(exact-ec[0]),
                    maximum_target_full_residual_Ha=float(actual[roots].max())))
        with np.load(out/'CCSDT_R3.npz',allow_pickle=False) as z:
            base=z['physical_C']
        with np.load(out/'signed_phase.npz',allow_pickle=False) as z:
            signed=z['physical_C']
            signed_e=z['EA']
        from independent import levels

        matched=levels(signed_e,signed,attached_masks,[tuple(x) for x in operator['labels']],ja)
        phases=np.array([np.prod([-1 if operator['labels'][i][:2]==[5,1] else 1 for i in range(10) if mask&(1<<i)]) for mask in attached_masks])
        base_arm=next(a for a in result['arms'] if a['label']=='CCSDT_R3')
        for name in ('S','D3'):
            left=phases[:,None]*base[:,base_arm['levels'][name]['roots']]
            right=signed[:,matched[name]['roots']]
            def projector(c):
                return c@np.linalg.solve(c.conj().T@c,c.conj().T)
            error=float(np.max(abs(projector(left)-projector(right))))
            gate('signed_complete_physical_projector:'+name,error,1e-9)
        report['status']='PASS_SAVED_NATIVE_CCSDT_REVIEW'
    except Exception as error:
        report.update(status='FAILED',exception=repr(error))
        raise
    finally:
        (here/'saved_data_review.json').write_text(json.dumps(report,indent=2)+'\n')
        print(report['status'],len(report['controls']))


if __name__=='__main__':
    main()
