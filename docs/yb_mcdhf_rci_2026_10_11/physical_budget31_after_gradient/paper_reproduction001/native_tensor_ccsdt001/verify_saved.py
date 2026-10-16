"""Saved-array closure; no atomic integral construction or new root solve."""
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'pyproject.toml').is_file())
OUT = HERE/'actual001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from atomx.constants import HARTREE_CM1
    report = json.loads((OUT/'result.json').read_text())
    source = json.loads((HERE/'source.json').read_text())
    contract = json.loads((HERE/'contract.json').read_text())
    checks = []
    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        checks.append(dict(name=name, measured=measured, cap=cap, passed=bool(passed)))
        assert passed, checks[-1]
    mapping = {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/atomx').rglob('*.py'))}
    gate('all_current_source_files',mapping==source['files'])
    gate('contract_source_identity',contract['source_map_sha256']==source['source_map_sha256'])
    gate('result_contract_identity',report['contract_sha256']==sha(HERE/'contract.json'))
    gate('all_input_bindings',all(sha(ROOT/n)==h for n,h in contract['inputs'].items()))
    with tarfile.open(HERE/'source.tar.gz') as archive:
        gate('complete_archive_members',set(archive.getnames())==set(mapping))
        gate('all_archive_content_hashes',all(hashlib.sha256(archive.extractfile(n).read()).hexdigest()==h for n,h in mapping.items()))
    gate('all196_live_controls',len(report['controls'])==196 and all(c['passed'] for c in report['controls']))
    gate('live_completed',report['status']=='PASS_NATIVE_TENSOR_YB_5D_PAIR_PILOT')
    with np.load(OUT/'prior_replay.npz') as z:
        gate('all_prior_directed_entries',np.max(abs(z['tensor_H']-z['finite_H_in_tensor_coordinates'])),1e-9)
    with np.load(OUT/'core_FCI.npz') as z:
        gate('full_core_FCI_eigen_equations',np.max(abs(z['H']@z['C']-z['C']*z['E'])),1e-10)
        gate('full_core_FCI_Gram',np.max(abs(z['C'].T@z['C']-np.eye(len(z['E'])))),1e-10)
        for name in ('CCSD','CCSDT'):
            with np.load(OUT/(name+'_core.npz')) as c:
                wave=c['physical_C']
                energy=report['cores'][name]['energy_Ha']
                gate(name+':physical_norm',abs(np.linalg.norm(wave)-1),1e-12)
                gate(name+':J0',np.linalg.norm(z['J2']@wave),1e-7)
                residual=np.linalg.norm(z['H']@wave-energy*wave)
                gate(name+':preserved_nonzero_full_H',residual>1e-6)
                gate(name+':saved_full_H_residual',abs(residual-report['cores'][name]['full_H_residual_Ha']),1e-12)
    energies={name:{target:[] for target in ('S','D3','D5')} for name in report['targets']}
    target_errors=[]
    for sector in report['sectors']:
        m2=sector['m2']
        with np.load(OUT/f'M{m2}_FCI.npz') as f:
            gate(f'M{m2}:complete_FCI_dimension',len(f['E'])==sector['full_FCI_dimension'])
            gate(f'M{m2}:FCI_all_equations',np.max(abs(f['H']@f['C']-f['C']*f['E'])),1e-10)
            gate(f'M{m2}:FCI_full_Gram',np.max(abs(f['C'].T@f['C']-np.eye(len(f['E'])))),1e-10)
            for target in report['targets']['FCI']:
                if target['m2']==m2:
                    k=target['root']
                    wave=f['C'][:,k]
                    angular={'S':.5,'D3':1.5,'D5':2.5}[target['target']]
                    gate(f'M{m2}:FCI_J2:{target["target"]}',np.linalg.norm(f['J2']@wave-angular*(angular+1)*wave),1e-7)
                    gate(f'M{m2}:FCI_saved_energy:{target["target"]}',abs(f['E'][k]-target['energy_Ha']),1e-12)
                    energies['FCI'][target['target']].append(float(f['E'][k]))
            for name in ('CCSD','CCSDT'):
                with np.load(OUT/f'M{m2}_{name}_R3.npz') as upper:
                    for rank in (2,3):
                        label=name+f'_R{rank}'
                        with np.load(OUT/f'M{m2}_{label}.npz') as a:
                            gate(f'M{m2}:{label}:all_projected_equations',np.max(abs(a['projected_H']@a['C']-a['C']*a['E'])),1e-10)
                            gate(f'M{m2}:{label}:all_physical_norms',np.max(abs(np.linalg.norm(a['physical_C'],axis=0)-1)),1e-12)
                            if rank==2:
                                lookup={int(index):k for k,index in enumerate(upper['indices'])}
                                selected=[lookup[int(index)] for index in a['indices']]
                                gate(f'M{m2}:{label}:principal_restriction',np.max(abs(a['projected_H']-upper['projected_H'][np.ix_(selected,selected)])),1e-12)
                            for target in report['targets'][label]:
                                if target['m2']!=m2:
                                    continue
                                k=target['root']
                                wave=a['physical_C'][:,k]
                                energy=float((a['E'][k]+report['cores'][name]['energy_Ha']).real)
                                angular={'S':.5,'D3':1.5,'D5':2.5}[target['target']]
                                gate(f'M{m2}:{label}:J2:{target["target"]}',np.linalg.norm(f['J2']@wave-angular*(angular+1)*wave),1e-7)
                                gate(f'M{m2}:{label}:target_energy:{target["target"]}',abs(energy-target['energy_Ha']),1e-12)
                                residual=float(np.linalg.norm(f['H']@wave-energy*wave))
                                gate(f'M{m2}:{label}:physical_defect_saved:{target["target"]}',abs(residual-target['full_H_residual_Ha']),1e-12)
                                gate(f'M{m2}:{label}:omitted_defect_nonzero:{target["target"]}',residual>1e-6)
                                energies[label][target['target']].append(energy)
                                ref=next(x for x in report['targets']['FCI'] if x['m2']==m2 and x['target']==target['target'])
                                target_errors.append(dict(method=label,target=target['target'],m2=m2,total_minus_FCI_cm1=(energy-ref['energy_Ha'])*HARTREE_CM1,full_H_residual_Ha=residual))
    lines={}
    for name,targets in energies.items():
        means={}
        for target,count in [('S',2),('D3',4),('D5',6)]:
            values=targets[target]
            gate(name+':all_partners:'+target,len(values)==count)
            gate(name+':multiplet_spread:'+target,np.ptp(values),1e-8)
            means[target]=float(np.mean(values))
        lines[name]={target:(means[target]-means['S'])*HARTREE_CM1 for target in ('D3','D5')}
        for target,value in lines[name].items():
            gate(name+':line_ledger:'+target,abs(value-report['lines_cm1'][name][target]),1e-8)
    output=dict(status='PASS_SAVED_NATIVE_TENSOR_REVIEW',controls=checks,source_files=len(mapping),lines_cm1=lines,target_errors=target_errors,scope='Saved equations, complete target-multiplet identities, physical norms and full-H defects. No new atomic integrals or root solves; other M/parity sectors remain uncomputed.')
    (HERE/'saved_data_review.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],controls=len(checks),source_files=len(mapping))))


if __name__=='__main__':
    main()
