"""Native finite Yb+ CCSD/CCSDT and complete-CI same-operator pilot."""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import tarfile
import time
import traceback
from collections.abc import Mapping
from math import comb
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'pyproject.toml').is_file())
OLD=HERE.parent/'yb_dhf_table001'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(x):
    if isinstance(x,Mapping):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [plain(v) for v in x]
    return plain(x.tolist()) if hasattr(x,'tolist') else x


def save(path,value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/atomx').rglob('*.py'))}


def prepare():
    mapping=sources()
    identity=hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    inputs=[Path(__file__),HERE/'independent.py',OLD/'actual004/core_ns180.h5',
        OLD/'actual004/blocks_ns180.npz',OLD/'actual004/result.json',
        ROOT/'tests/test_eom_attachment.py']
    amplitudes={r:comb(4,r)*comb(6,r) for r in (1,2,3)}
    dense=8*sum(amplitudes.values())*(comb(10,4)**2+comb(10,5)**2)
    save(HERE/'source.json',dict(files=mapping,source_map_sha256=identity))
    with tarfile.open(HERE/'source.tar.gz','w:gz') as archive:
        for name in mapping:
            archive.add(ROOT/name,arcname=name,recursive=False)
    contract=dict(hypothesis='Iterative connected core triples and attachment3p2h have separately measurable effects in a complete finite actual-Yb DC pilot; fullCI on the identical operator quantifies the retained omitted-rank defect.',
        reference='https://arxiv.org/html/2504.01438v1 Eq2:common68e reference with separate coreT and valenceSv ranks; this finiteEOM-EA pilot is not that fullFS-RCCSDT implementation.',
        fixed=dict(Z=70,isotope='A171 empiricalFermi, same actual native ns180/R500/ks9/RKB bank as precedingDHF baseline',
            physical_core_electrons=68,target_electrons=69,
            correlated_core_radials=[[5,-1],[5,1]],correlated_core_electrons=4,
            frozen='all other17 complete occupied radial multiplets,64electrons, including4f14 and5p3/2^4; noncontiguous explicit pilot partition',
            virtual_radials=[[6,-1],[5,2]],virtual_magnetic_spinors=6,
            rationale='Mixed5s2/5p1/2^2 permits parity-allowed nonzeroT3 with complete6s/5d3/2 multiplets within the native finite-sector cap. No fitted orbital/cutoff and no silent partition reduction.',
            full_core_determinants=210,full_attached_determinants=252,spinors=10,
            core_amplitudes_by_rank=amplitudes,
            attachment_R2=66,attachment_R3=186,omitted_attachment_R4_R5=66,
            Hamiltonian='nativeDirac-Coulomb; all frozen contractions and frozen constant retained; noBreit/QED/recoil',
            origin='Subtract exactly the original common68e reference energy from both particle sectors; preserve it in the output ledger. This changes only the scalar energy zero.',
            factorial=[['CCSD',2],['CCSD',3],['CCSDT',2],['CCSDT',3]],
            continuation=[.25,.5,.75,1.0],root_matching='Every2/4 roots with bare6s/5d3/2 model weight>=0.5; verify entire multiplet spread andJ2, not nearest experimental energy'),
        acceptance=dict(amplitude_and_projected_Ha=1e-10,independent_Ha=1e-10,
            independent_similarity_Ha=1e-9,multiplet_spread_Ha=1e-8,J_squared_defect=1e-7,
            core_T3_norm_min=1e-12,attached_T3_action_norm_min=1e-12,phase_energy_Ha=1e-9,
            scope='FullCI differences and full-H residuals are diagnostics; do not requireCCSDT=FCI when coreT4 or attachment4p3h/5p4h are omitted.'),
        resources=dict(cpu=27,threads=1,AS_bytes=8*1024**3,seconds=900,load1_max=24,
            available_min_bytes=16*1024**3,determinant_cap=256,amplitude_cap=256,
            generator_memory_MiB=512,dense_generator_bytes_largest=dense,
            estimated_EA_NumPy_upper=dense+8*(16*(210**2+252**2)+8*186**2),
            logical_term_action_cap=2_000_000,estimated_H_term_actions=(1+10**2+comb(10,2)**2)*(210+252),
            note='Public amplitude128 and generator128MiB defaults are explicitly raised for this declared194-amplitude case. No determinant cap or physics reduction. Dense bank chosen because prior same-model state-action timing was slower; no performance claim.',
            calls='four factorial nativeEOM solves plus one signed-radial phase covariance solve; one complete independent4e/5e FCI pair; one fresh native packed integral operator; no external program'),
        stopping='First failed identity/operator/solver/multiplet/phase/resource control stops and preserves evidence. No automatic retry, rank/space/cutoff/threshold changes or borrowed paper correction.',
        limits='Pilot freezes64electrons and omits5d5/2, highl andradial convergence. No paperprecision, full69e atomicRCCSDT or replacement of mainMCDHF route.',
        source_map_sha256=identity,inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    save(HERE/'contract.json',contract)
    print(json.dumps(dict(status='PREPARED',amplitudes=amplitudes,dense_generator_bytes=dense)))


def run():
    contract=json.loads((HERE/'contract.json').read_text())
    source=json.loads((HERE/'source.json').read_text())
    assert sources()==source['files']
    assert all(sha(ROOT/p)==h for p,h in contract['inputs'].items())
    assert all(os.environ.get(k)=='1' for k in THREADS)
    assert os.getloadavg()[0]<=contract['resources']['load1_max']
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    assert available>=contract['resources']['available_min_bytes']
    os.sched_setaffinity(0,{27})
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    def alarm(*_):
        raise TimeoutError('Declared900-second pilot cap')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(900)
    out=HERE/'actual001'
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    report=dict(status='RUNNING',controls=[],arms=[],source_map_sha256=source['source_map_sha256'],contract_sha256=sha(HERE/'contract.json'))
    def gate(name,value,cap=None):
        passed=bool(value) if cap is None else bool(value<=cap)
        report['controls'].append(dict(name=name,measured=value,cap=cap,passed=passed))
        if not passed:
            raise AssertionError(name+': '+str(value))
    try:
        import numpy as np
        from independent import j_squared, levels, similarity_check
        from scipy import linalg

        from atomx.constants import HARTREE_CM1
        from atomx.methods import eom_attachment as ea
        from atomx.methods.dirac_fock import SCFResult
        from atomx.physics.csf.spinor_integrals import (
            SpinorIntegrals,
            build_spinor_integrals,
        )
        from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile

        core=SCFResult.load(OLD/'actual004/core_ns180.h5')
        gate('same_A171_nucleus',core.solver_diagnostics['_nuclear_profile']==nuclear_profile(70,'fermi',171))
        core._nuclear_potential=make_nuclear_potential(70,'fermi',171)
        with np.load(OLD/'actual004/blocks_ns180.npz',allow_pickle=False) as z:
            orbitals={tuple(map(int,k)):(p,q) for k,p,q in zip(z['all_orbital_keys'],z['all_P'],z['all_Q'],strict=True)}
        occupied=((5,-1),(5,1))
        virtual=((6,-1),(5,2))
        frozen=tuple(k for k in core.orbitals_by_subshell() if k not in occupied)
        gate('complete64_frozen_electrons',sum(2*abs(k) for n,k in frozen)==64 and len(frozen)==17)
        print('BUILD_NATIVE_INTEGRALS',flush=True)
        original=build_spinor_integrals(orbitals,core._bsp_data,70,180,
            occupied_keys=occupied,virtual_keys=virtual,frozen_keys=frozen,
            nuclear_potential=core._nuclear_potential,max_integral_bytes=32*1024**2,metric_tolerance=1e-10)
        gate('all10_magnetic_spinors',len(original.labels)==10 and original.noccupied==4)
        gate('reference_energy_ledger',abs(original.reference_energy_Ha-core.total_energy),1e-7)
        origin=original.reference_energy_Ha
        identity=dict(original.identity)
        identity.update(pilot='explicit64frozen+4correlated+6virtualmagnetic; actualYb',energy_origin_Ha=origin,
            original_frozen_energy_Ha=original.frozen_energy_Ha,source_map_sha256=source['source_map_sha256'])
        integrals=SpinorIntegrals(original.labels,4,original.one_body_Ha,original.fock_Ha,
            original.packed_eri_Ha,original.frozen_energy_Ha-origin,0.0,identity)
        save(out/'operator.json',dict(labels=integrals.labels,identity=identity,
            occupied=occupied,virtual=virtual,frozen=frozen,absolute_energy_origin_Ha=origin,
            original_frozen_energy_Ha=original.frozen_energy_Ha,shifted_frozen_energy_Ha=integrals.frozen_energy_Ha))
        np.savez(out/'integrals.npz',h=integrals.one_body_Ha,fock=integrals.fock_Ha,g=integrals.packed_eri_Ha)
        print('INDEPENDENT_COMPLETE_FCI',flush=True)
        spec=importlib.util.spec_from_file_location('independent_ea_oracle',ROOT/'tests/test_eom_attachment.py')
        oracle=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        cm,hc=oracle.independent_hamiltonian(integrals,integrals.one_body_Ha.real,integrals.eri,4)
        am,ha=oracle.independent_hamiltonian(integrals,integrals.one_body_Ha.real,integrals.eri,5)
        jc,ja=j_squared(integrals.labels,cm),j_squared(integrals.labels,am)
        ec,vc=linalg.eigh(hc)
        ev,vv=linalg.eigh(ha)
        gate('FCI_core_connected_branch',abs(vc[0,0])**2>=.5)
        gate('FCI_core_J0',float(np.linalg.norm(jc@vc[:,0])),1e-7)
        fci=levels(ev,vv,am,integrals.labels,ja)
        for name,row in fci.items():
            gate('FCI_multiplet_spread:'+name,row['spread_Ha'],1e-8)
            gate('FCI_J2:'+name,row['J_squared_defect'],1e-7)
        report['FCI']=dict(core_energy_relative_Ha=float(ec[0]),levels=fci,
            D3_line_cm1=(fci['D3']['mean_energy_Ha']-fci['S']['mean_energy_Ha'])*HARTREE_CM1)
        np.savez(out/'independent_FCI.npz',core_H=hc,core_E=ec,core_C=vc,
            attached_H=ha,attached_E=ev,attached_C=vv,core_masks=np.asarray(cm),attached_masks=np.asarray(am),core_J2=jc,attached_J2=ja)
        report['native_vs_independent_H']=dict(core=float(np.max(abs(ea._matrix(ea._hamiltonian_terms(integrals),cm)-hc))),
            attached=float(np.max(abs(ea._matrix(ea._hamiltonian_terms(integrals),am)-ha))))
        for name,error in report['native_vs_independent_H'].items():
            gate('native_vs_independent_H:'+name,error,1e-10)
        computed={}
        for method,rank in contract['fixed']['factorial']:
            label=f'{method}_R{rank}'
            print('SOLVE',label,flush=True)
            result=ea.compute(integrals,core_method=method,attachment_particle_rank=rank,
                amplitude_cap=256,generator_memory_MiB=512,residual_tolerance_Ha=1e-10)
            computed[label]=result
            targets=levels(result.attachment_energies_Ha,result.coefficients,
                           result.determinant_masks,result.provenance['labels'],ja)
            t3=[value for value,(holes,particles) in zip(result.core.amplitudes,result.core.provenance['excitations'],strict=True) if len(holes)==3]
            triple_norm=float(np.linalg.norm(t3))
            check=similarity_check(result,ha)
            row=dict(label=label,method=result.method,core_energy_relative_Ha=result.core.total_energy_Ha,
                core_correlation_Ha=result.core.correlation_energy,core_amplitude_residual_Ha=result.core.amplitude_residual_Ha,
                core_full_residual_Ha=result.core.full_eigenstate_residual_Ha,core_FCI_energy_error_Ha=result.core.total_energy_Ha-ec[0],
                core_T3_norm=triple_norm,levels=targets,similarity=check,
                D3_line_cm1=(targets['D3']['mean_energy_Ha']-targets['S']['mean_energy_Ha'])*HARTREE_CM1,
                maximum_projected_residual_Ha=float(np.max(result.projected_residuals_Ha)),
                maximum_full_H_residual_Ha=float(np.max(result.full_eigenstate_residuals_Ha)),provenance=result.provenance)
            report['arms'].append(row)
            for name,target in targets.items():
                gate(label+':spread:'+name,target['spread_Ha'],1e-8)
                gate(label+':J2:'+name,target['J_squared_defect'],1e-7)
            gate(label+':core_J0',float(np.linalg.norm(jc@result.core.coefficients)),1e-7)
            gate(label+':directed_similarity',check['maximum_directed_Hbar_difference_Ha'],1e-9)
            if method=='CCSDT':
                gate(label+':nonzero_core_T3',triple_norm>1e-12)
                gate(label+':nonzero_attached_T3_action',check['attached_T3_action_norm']>1e-12)
            np.savez(out/(label+'.npz'),EA=result.attachment_energies_Ha,total=result.total_energies_Ha,
                physical_C=result.coefficients,projected_C=result.projected_right_vectors,
                projected_H=result.projected_operator_Ha,projected_masks=np.asarray(result.projected_masks),
                full_residuals=result.full_eigenstate_residuals_Ha,projected_residuals=result.projected_residuals_Ha,
                core_amplitudes=result.core.amplitudes,core_C=result.core.coefficients)
            save(out/(label+'.json'),row)
            save(out/'partial.json',report)
            print('LINE',label,row['D3_line_cm1'],'T3norm',triple_norm,flush=True)
        # One independently signed radial basis phase, preserving whole multiplets.
        phases=np.array([-1. if label[:2]==(5,1) else 1. for label in integrals.labels])
        packed=np.array([integrals.eri(p,q,r,s)*phases[p]*phases[q]*phases[r]*phases[s]
            for p in range(9) for q in range(p+1,10) for r in range(p,9) for s in range(r+1,10)])
        changed=SpinorIntegrals(integrals.labels,4,phases[:,None]*integrals.one_body_Ha*phases,
            phases[:,None]*integrals.fock_Ha*phases,packed,integrals.frozen_energy_Ha,0.,dict(identity,signed_radial_phase='5p1/2wholemultiplet'))
        print('SIGNED_PHASE_CONTROL',flush=True)
        signed=ea.compute(changed,core_method='CCSDT',attachment_particle_rank=3,
            amplitude_cap=256,generator_memory_MiB=512,residual_tolerance_Ha=1e-10)
        base=computed['CCSDT_R3']
        detphase=np.array([np.prod([phases[i] for i in range(10) if mask&(1<<i)]) for mask in base.projected_masks])
        phase_error=float(np.max(abs(signed.projected_operator_Ha-detphase[:,None]*base.projected_operator_Ha*detphase)))
        gate('complete_signed_projected_matrix',phase_error,1e-9)
        gate('signed_complete_spectrum',float(np.max(abs(signed.attachment_energies_Ha-base.attachment_energies_Ha))),1e-9)
        report['phase_control']=dict(complete_matrix_error_Ha=phase_error)
        np.savez(out/'signed_phase.npz',projected_H=signed.projected_operator_Ha,EA=signed.attachment_energies_Ha,physical_C=signed.coefficients,detphase=detphase)
        values={a['label']:a['D3_line_cm1'] for a in report['arms']}
        report['same_model_effects_cm1']=dict(core_T3_at_R2=values['CCSDT_R2']-values['CCSD_R2'],
            core_T3_at_R3=values['CCSDT_R3']-values['CCSD_R3'],
            attachment_R3_at_CCSD=values['CCSD_R3']-values['CCSD_R2'],
            attachment_R3_at_CCSDT=values['CCSDT_R3']-values['CCSDT_R2'],
            combined=values['CCSDT_R3']-values['CCSD_R2'])
        gate('production_source_unchanged',sources()==source['files'])
        gate('all_inputs_unchanged',all(sha(ROOT/p)==h for p,h in contract['inputs'].items()))
        report['status']='PASS_BOUNDED_NATIVE_YB_CCSDT_PILOT'
    except Exception:
        report.update(status='FAILED',exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-start
        report['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save(out/'result.json',report)
        print('END',report['status'],report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else run()
