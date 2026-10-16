"""Matched H0 controls and common-ground five-state responses."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
from atomx.constants import HARTREE_CM1
from atomx.methods.mbpt import _build_closed_shell_fock
from atomx.numerics.bspline.convol import ConvolCache
from atomx.physics.csf.matrix import build_ci_matrix

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from probe_yb_qspace import spaces, SECTORS, CORE, EXPERIMENT, load_reference

def main():
    resource.setrlimit(resource.RLIMIT_AS,(4096*1024**2,4096*1024**2))
    start=time.monotonic()
    core,ref,_,identity=load_reference('v68_default')
    data=core._bsp_data
    od=ref.orbitals_dict
    csf=enumerate_jj_csfs(CORE)
    ecore=float(build_ci_matrix(csf,od,get_shared_orbital_basis(csf),data,70,data['ns'],
                               nuclear_potential=core._nuclear_potential)[0,0])
    valence={'6s_1/2':(6,-1),'5d_3/2':(5,2),'5d_5/2':(5,-3),'6p_1/2':(6,1),'6p_3/2':(6,-2)}
    active=list(valence.values())+[(4,3),(4,-4)]
    eps={}; cache=ConvolCache(data)
    for n,k in active:
        h,s=_build_closed_shell_fock(core,k,cache=cache,potential_mode='VN')
        x=np.concatenate(od[n,k])
        eps[n,k]=float(x@h@x)
        if abs(float(x@s@x)-1)>1e-7: raise RuntimeError('reference Fock metric gate')
    declared,_=spaces()
    rows={}; max_zero=0.
    for label in SECTORS:
        r=json.loads((HERE/(label.replace('/','_')+'_qspace.json')).read_text())
        csfs,p=declared[label]
        with np.load(HERE/(label.replace('/','_')+'_qspace.npz')) as a: h=a['hamiltonian']
        zero_error=abs(r['bare_total_Ha']-ecore-eps[valence[label]])
        max_zero=max(max_zero,zero_error)
        if zero_error>1e-10: raise RuntimeError('common native CI/Fock energy zero gate')
        q=[i for i in range(len(csfs)) if i not in p]
        q0=[]
        for i in q:
            occ={(sh.n,sh.kappa):sh.occupation for sh in csfs[i].subshells}
            energy=ecore
            for key in active:
                reference_occ=2*abs(key[1]) if key in ((4,3),(4,-4)) else 0
                energy+=(occ.get(key,0)-reference_occ)*eps[key]
            q0.append(energy)
        den=r['bare_total_Ha']-np.array(q0)
        if np.min(abs(den))<1e-8: raise RuntimeError('Born control denominator gate')
        coupling=h[p[0],q]
        born=float(np.sum(coupling**2/den))
        fixed=r['full_Q_fixed_energy_Ha']-r['bare_total_Ha']
        rows[label]=dict(**r,Born_one_hole_shift_Ha=born,
            Q_interaction_dressing_cm1=(fixed-born)*HARTREE_CM1,
            energy_argument_response_cm1=(r['shift_Ha']-fixed)*HARTREE_CM1,
            full_minus_Born_cm1=(r['shift_Ha']-born)*HARTREE_CM1,
            P_Fock_zero_error_Ha=zero_error)
        np.savez_compressed(HERE/(label.replace('/','_')+'_born_control.npz'),Q0_Ha=q0,PQ_Ha=coupling)
    g=rows['6s_1/2']
    for label,r in rows.items():
        bare=(r['bare_total_Ha']-g['bare_total_Ha'])*HARTREE_CM1
        full=(r['full_Q_total_Ha']-g['full_Q_total_Ha'])*HARTREE_CM1
        born=bare+(r['Born_one_hole_shift_Ha']-g['Born_one_hole_shift_Ha'])*HARTREE_CM1
        r.update(bare_transition_cm1=bare,full_Q_transition_cm1=full,Born_one_hole_transition_cm1=born,
                 transition_response_cm1=full-bare,experiment_cm1=EXPERIMENT[label],
                 bare_residual_cm1=bare-EXPERIMENT[label],full_residual_cm1=full-EXPERIMENT[label])
    splittings={name:{model:rows[b][model]-rows[a][model] for model in
                      ('bare_transition_cm1','full_Q_transition_cm1','Born_one_hole_transition_cm1','experiment_cm1')}
                for name,a,b in [('5d','5d_3/2','5d_5/2'),('6p','6p_1/2','6p_3/2')]}
    out=dict(rows=rows,splittings=splittings,core_CI_zero_Ha=ecore,
             Fock_expectations_Ha=[dict(key=key,energy=value) for key,value in eps.items()],
             max_CI_Fock_zero_error_Ha=max_zero,seconds=time.monotonic()-start,pass_=True)
    (HERE/'qspace_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(max_CI_Fock_zero_error_Ha=max_zero,splittings=splittings,
        order_response_cm1={label:{k:r[k] for k in ('Q_interaction_dressing_cm1','energy_argument_response_cm1','full_minus_Born_cm1')} for label,r in rows.items()})),flush=True)

if __name__=='__main__': main()
