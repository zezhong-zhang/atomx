"""Complete declared one-4f-hole model; independently gates every sector."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.ci_effective import build_finite_ci_effective_hamiltonian
from atomx.constants import HARTREE_CM1

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'yb_diagnosis_repair_2026_10_08'))
from common import CORE, EXPERIMENT, load_reference

SECTORS={'6s_1/2':(1,1),'5d_3/2':(3,1),'5d_5/2':(5,1),
         '6p_1/2':(1,-1),'6p_3/2':(3,-1)}

def spaces():
    pp=[CORE+' '+s+'1' for s in ('6s','5d','6p')]
    deep=CORE.replace('4f14','4f13')
    outer=('6s','5d','6p')
    qq=[deep+' '+(a+'2' if a==b else a+'1 '+b+'1')
        for i,a in enumerate(outer) for b in outer[i:]]
    p_csfs=enumerate_jj_csfs_many(pp)
    p_signatures={csf_signature(c).key for c in p_csfs}
    all_csfs=enumerate_jj_csfs_many(pp+qq)
    result={}
    records={}
    for label,sector in SECTORS.items():
        csfs=[c for c in all_csfs if (c.total_J2,c.parity)==sector]
        p=tuple(i for i,c in enumerate(csfs) if csf_signature(c).key in p_signatures)
        assert len(p)==1
        patterns={csf_signature(c).pattern for c in csfs}
        determinants=sum(dict(pattern_m_counts(pattern)).get(sector[0],0) for pattern in patterns)
        if len(csfs)>256 or determinants>2000: raise RuntimeError('declared space exceeds resource cap')
        result[label]=(csfs,p)
        records[label]=dict(sector=sector,CSFs=len(csfs),P=len(p),Q=len(csfs)-len(p),
                            determinants=determinants,
                            identities=[csf_signature(c).key for c in csfs])
    return result,records

def main():
    resource.setrlimit(resource.RLIMIT_AS,(4096*1024**2,4096*1024**2))
    declared,counts=spaces()
    (HERE/'yb_preflight.json').write_text(json.dumps(dict(configurations='all 4f14 v1 and 4f13 v2 in declared valence pool',sectors=counts),indent=2)+'\n')
    if len(sys.argv)==1 or sys.argv[1]=='--preflight': return
    label=sys.argv[1]
    csfs,p=declared[label]
    core,result,levels,identity=load_reference('v68_default')
    started=time.monotonic()
    model=build_finite_ci_effective_hamiltonian(csfs,p,result.orbitals_dict,core._bsp_data,70,core._bsp_data['ns'],
        reference_identity='Yb:finite-one-f-hole:'+label+':'+identity['orbitals_workspace_sha256'],
        nuclear_potential=core._nuclear_potential,dimension_cap=256,max_determinants=2000)
    e0=float(model.self_energy.reference_energies_Ha[0])
    poles=model.self_energy.poles_Ha
    # Bound by the nearest coupled Q poles. The bracket is selected from
    # the Hamiltonian, independently of experimental transition energies.
    below=poles[poles<e0]; above=poles[poles>e0]
    lo=max(e0-1.,float(np.max(below))+1e-6 if len(below) else -np.inf)
    hi=min(e0+.01,float(np.min(above))-1e-6 if len(above) else np.inf)
    root=model.solve((lo,hi))
    en,vec=np.linalg.eigh(model.hamiltonian_Ha)
    column=int(np.argmin(abs(en-root.energy_Ha)))
    energy_error=float(abs(en[column]-root.energy_Ha))
    projector_error=float(np.max(abs(np.outer(vec[:,column],vec[:,column])-np.outer(root.coefficients,root.coefficients))))
    if energy_error>1e-10 or projector_error>1e-8 or root.p_weight<.5:
        raise RuntimeError('root identity or dense reference gate failed')
    diagonal_control=float(model.diagonal_q_control(e0)[0,0])
    full_fixed=float(model.effective_hamiltonian(e0)[0,0])
    np.savez_compressed(HERE/(label.replace('/','_')+'_qspace.npz'),hamiltonian=model.hamiltonian_Ha,
                        p_indices=p,full_vector=root.coefficients,q_poles=model.q_poles_Ha)
    report=dict(label=label,reference_identity=identity,bare_total_Ha=e0,full_Q_total_Ha=root.energy_Ha,
        shift_Ha=root.shift_Ha,shift_cm1=root.shift_Ha*HARTREE_CM1,P_weight=root.p_weight,
        full_residual_Ha=root.residual_Ha,dense_energy_error_Ha=energy_error,
        dense_projector_error=projector_error,diagonal_Q_fixed_energy_Ha=diagonal_control,
        full_Q_fixed_energy_Ha=full_fixed,Q_mixing_at_fixed_energy_cm1=(full_fixed-diagonal_control)*HARTREE_CM1,
        bracket_Ha=[lo,hi],dimension=len(csfs),seconds=time.monotonic()-started,
        max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,pass_=True)
    (HERE/(label.replace('/','_')+'_qspace.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='reference_identity'}),flush=True)

if __name__=='__main__': main()
