"""Five-sector frozen-orbital SD and full correlated static-Breit panel."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
from atomx.constants import HARTREE_CM1
from atomx.methods.ci_effective import static_breit_second_order_control
from atomx.methods.ci_sd import plan_sd_ci_space, build_sd_ci_effective_hamiltonian
from atomx.methods.mbpt import _build_closed_shell_fock
from atomx.numerics.bspline.convol import ConvolCache
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.physics.csf.radial_snapshot import CIRadialSnapshot

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'yb_diagnosis_repair_2026_10_08'))
from common import CORE, KEYS, EXPERIMENT, load_reference
SECTORS={'6s_1/2':(1,1),'5d_3/2':(3,1),'5d_5/2':(5,1),'6p_1/2':(1,-1),'6p_3/2':(3,-1)}
FROZEN=('1s','2s','2p','3s','3p','3d','4s','4p','4d','5s','5p')


def plans(rank):
    return {label:plan_sd_ci_space([CORE+' '+v+'1' for v in ('6s','5d','6p')],
                    ('6s','5d','6p'),frozen_orbitals=FROZEN,J2=j,parity=parity,
                    max_excitations=rank,dimension_cap=1000,max_determinants=6000)
            for label,(j,parity) in SECTORS.items()}


def solve(model):
    e0=float(model.self_energy.reference_energies_Ha[0])
    q=model.q_poles_Ha
    below=q[q<e0]; above=q[q>e0]
    lo=max(e0-1.,float(np.max(below))+1e-8 if len(below) else -np.inf)
    hi=min(e0+.02,float(np.min(above))-1e-8 if len(above) else np.inf)
    root=model.solve((lo,hi))
    en,vec=np.linalg.eigh(model.hamiltonian_Ha)
    n=int(np.argmin(abs(en-root.energy_Ha)))
    error=float(abs(en[n]-root.energy_Ha))
    projector=float(np.max(abs(np.outer(vec[:,n],vec[:,n])-np.outer(root.coefficients,root.coefficients))))
    if error>1e-10 or projector>1e-8 or root.p_weight<.5:
        raise RuntimeError('full CI/root identity gate failed')
    return root,dict(bracket_Ha=[lo,hi],dense_energy_error_Ha=error,projector_error=projector,
        P_weight=root.p_weight,residual_Ha=root.residual_Ha,total_Ha=root.energy_Ha,
        bare_P_Ha=e0,shift_cm1=root.shift_Ha*HARTREE_CM1)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    if len(sys.argv)==1:
        records={}
        for rank in (1,2):
            for label,plan in plans(rank).items():
                csfs,p=plan.materialize()
                patterns={csf_signature(c).pattern for c in csfs}
                actual=sum(dict(pattern_m_counts(pat)).get(plan.J2,0) for pat in patterns)
                records[f'{rank}:{label}']=dict(**asdict(plan),P_indices=p,actual_determinants=actual,
                    CSF_identities=[csf_signature(c).key for c in csfs])
        (HERE/'preflight.json').write_text(json.dumps(records,indent=2)+'\n')
        return
    label=sys.argv[1]; plan=plans(2)[label]; single=plans(1)[label]
    started=time.monotonic()
    core,ref,_,identity=load_reference('v68_default')
    ws=core._bsp_data; od=ref.orbitals_dict
    snapshot=CIRadialSnapshot(od,ws,70,ws['ns'],nuclear_potential=core._nuclear_potential)
    options=dict(reference_identity='Yb:SD-frozen:'+label+':'+identity['orbitals_workspace_sha256'],
                 nuclear_potential=core._nuclear_potential,radial_snapshot=snapshot)
    stem=label.replace('/','_')
    single_dc=build_sd_ci_effective_hamiltonian(single,od,ws,70,ws['ns'],**options)
    single_root,single_record=solve(single_dc)
    old=json.loads((HERE.parent/'ci_mbpt_matched_qspace_2026_10_09'/(stem+'_qspace.json')).read_text())
    replay=abs(single_root.energy_Ha-old['full_Q_total_Ha'])
    if replay>1e-10: raise RuntimeError('old complete one-hole DC replay failed')
    print(json.dumps({'stage':'one-hole DC replay','error_Ha':replay,'seconds':time.monotonic()-started}),flush=True)
    dc=build_sd_ci_effective_hamiltonian(plan,od,ws,70,ws['ns'],**options)
    dc_root,dc_record=solve(dc)
    print(json.dumps({'stage':'SD DC','total_Ha':dc_root.energy_Ha,'P_weight':dc_root.p_weight,
                      'seconds':time.monotonic()-started}),flush=True)
    # Store DC evidence before the more expensive DCB assembly.
    np.savez_compressed(HERE/(stem+'_dc_checkpoint.npz'),hamiltonian=dc.hamiltonian_Ha,
                       vector=dc_root.coefficients,p_indices=dc.p_indices)
    dcb=build_sd_ci_effective_hamiltonian(plan,od,ws,70,ws['ns'],**options,include_breit=True)
    dcb_root,dcb_record=solve(dcb)
    b=dcb.hamiltonian_Ha-dc.hamiltonian_Ha
    first_correlated=float(dc_root.coefficients@b@dc_root.coefficients)
    full_response=dcb_root.energy_Ha-dc_root.energy_Ha
    # Native mean-field H0 with the same CI core zero, no experimental shift.
    closed=enumerate_jj_csfs(CORE)
    core_zero=float(build_ci_matrix(closed,od,get_shared_orbital_basis(closed),ws,70,ws['ns'],
                                   nuclear_potential=core._nuclear_potential)[0,0])
    active=list(KEYS.values())+[(4,3),(4,-4)]
    eps={}; cache=ConvolCache(ws)
    for n,k in active:
        f,s=_build_closed_shell_fock(core,k,cache=cache,potential_mode='VN')
        x=np.concatenate(od[n,k]); eps[n,k]=float(x@f@x)
        if abs(float(x@s@x)-1)>1e-7: raise RuntimeError('Fock metric gate')
    csfs,p=plan.materialize()
    h0=[]
    for c in csfs:
        occ={(sh.n,sh.kappa):sh.occupation for sh in c.subshells}
        h0.append(core_zero+sum((occ.get(key,0)-(2*abs(key[1]) if key in ((4,3),(4,-4)) else 0))*eps[key] for key in active))
    control=static_breit_second_order_control(dc,dcb,np.array(h0))
    # Independent signed wavefunction cross-contraction of the mixed term.
    q=list(dc.q_indices); p0=dc.p_indices[0]
    d=np.array(h0)[p0]-np.array(h0)[q]
    first_wave=np.zeros(len(h0)); first_wave[q]=dc.hamiltonian_Ha[q,p0]/d
    independent=float(2*b[p0]@first_wave)
    if abs(independent-control.coulomb_breit_Ha)>1e-12:
        raise RuntimeError('independent C-B wave-response contraction failed')
    report=dict(label=label,plan=asdict(plan),reference=identity,one_hole_DC=single_record,
        DC=dc_record,DCB=dcb_record,old_one_hole_replay_error_Ha=replay,
        two_hole_total_response_cm1=(dc_root.energy_Ha-single_root.energy_Ha)*HARTREE_CM1,
        full_Breit_response_cm1=full_response*HARTREE_CM1,
        first_order_correlated_Breit_cm1=first_correlated*HARTREE_CM1,
        nonlinear_Breit_cm1=(full_response-first_correlated)*HARTREE_CM1,
        restricted_H0_control={k:v for k,v in asdict(control).items() if k!='h0_diagonal_Ha'},
        independent_CB_error_Ha=abs(independent-control.coulomb_breit_Ha),
        seconds=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        DC_terms=dc.hamiltonian_terms,DCB_terms=dcb.hamiltonian_terms,
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'src/atomx').rglob('*.py'))},numerical_pass=True)
    np.savez_compressed(HERE/(stem+'_sd_breit.npz'),DC=dc.hamiltonian_Ha,DCB=dcb.hamiltonian_Ha,
                        H0=np.array(h0),DC_vector=dc_root.coefficients,DCB_vector=dcb_root.coefficients,
                        p_indices=dc.p_indices,one_hole_DC=single_dc.hamiltonian_Ha)
    (HERE/(stem+'_result.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('reference','source_sha256','plan')}),flush=True)

if __name__=='__main__': main()
