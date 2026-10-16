"""Repair complete physical-metric states and repeat their native CI ledger."""
import hashlib
import json
import os
import resource
import shutil
import signal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'pyproject.toml').is_file())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    os.sched_setaffinity(0,{27})
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    def stop(*_):
        raise TimeoutError('120-second saved-state correction cap')
    signal.signal(signal.SIGALRM,stop)
    signal.alarm(120)
    inputs=[HERE/'actual003/result.json',HERE/'contract003.json',HERE/'source003.json',
            HERE/'constrained_pair.py',Path(__file__),HERE/'saved_data_review.json',
            HERE/'physical_pair_diagnosis002.json']
    inputs += [HERE/f'actual003/{name}_ns{ns}.{ext}' for ns in (180,220)
               for name,ext in [('core','h5'),('blocks','npz')]]
    contract=dict(hypothesis='Original-pencil constrained Newton and preserving that exact leading vector through complete physical-metric reconstruction remove the embedding defect without changing any physical input, branch, column count or gate.',
        parent='contract003.json; all physical inputs and scientific gates unchanged',
        method='Exactly two bordered corrections in original H/S with all occupied constraints; one whole-branch original-metric projection/ordered QR; preserve corrected leading column verbatim. No step/rank/grid/cutoff/tolerance scan.',
        source='source003.json, unchanged359 production Python files',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        limits=dict(cpu=27,threads=1,AS_bytes=8*1024**3,seconds=120,
                    memory='complete two saved sets <128MiB NumPy inputs; <3GiB anticipated RSS',
                    Fock_builds=0,SCF_solves=0,CI_calls=4,raw_determinants=32),
        acceptance=dict(target_original_dual_Ha=1e-8,physical_Gram=1e-10,
            core_overlap=1e-10,column_change_S_norm=1e-8,CI_ledger_Ha=1e-7,
            CI_residual_Ha=1e-8,radial_drift_cm1=1.0),
        stopping='First failed gate stops and preserves evidence; no fallback or gate relaxation',
        scope='Independent DC DHF baseline and static properties only; no full paper RCCSDT,69e-MCDHF convergence or experimental precision claim')
    (HERE/'contract004.json').write_text(json.dumps(contract,indent=2)+'\n')
    out=HERE/'actual004'
    out.mkdir(exist_ok=False)
    import numpy as np
    from constrained_pair import independent_controls, refine
    from scipy import linalg

    from atomx.config.csf import enumerate_jj_csfs
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci import ci_solve
    from atomx.methods.dirac_fock import SCFResult
    from atomx.potentials.nuclear import make_nuclear_potential

    report=dict(status='RUNNING',controls=[],arms=[],independent=independent_controls())
    def gate(name,value,cap=None):
        passed=bool(value) if cap is None else bool(value<=cap)
        report['controls'].append(dict(name=name,measured=value,cap=cap,passed=passed))
        if not passed:
            raise AssertionError(name+': '+str(value))
    try:
        previous=json.loads((HERE/'actual003/result.json').read_text())
        source=json.loads((HERE/'source003.json').read_text())['all_atomx_python_sha256']
        gate('actual_source_epoch',all(sha(ROOT/p)==h for p,h in source.items()))
        wide=np.longdouble
        core_config='1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14'
        for ns in (180,220):
            core=SCFResult.load(HERE/f'actual003/core_ns{ns}.h5')
            core._nuclear_potential=make_nuclear_potential(70,'fermi',171)
            shutil.copyfile(HERE/f'actual003/core_ns{ns}.h5',out/f'core_ns{ns}.h5')
            with np.load(HERE/f'actual003/blocks_ns{ns}.npz',allow_pickle=False) as a:
                arrays={k:a[k] for k in a if not k.startswith(('white_','metric_Cholesky_','prepared_'))}
            keep=arrays['active_coordinates']
            arm=dict(ns=ns,roots=[],CI=[])
            report['arms'].append(arm)
            energies={}
            for key in ((6,-1),(5,2),(5,-3)):
                k=key[1]
                h,s,c,b,e,full=[arrays[f'{name}_k{k}'] for name in
                               ('H','S','occupied','complement','positive_E','positive_C')]
                v=full[keep].copy()
                energy,x=refine(h,s,c,e[0],v[:,0])
                v[:,0]=x
                lower=linalg.cholesky(s,lower=True)
                occupied,_=linalg.qr(lower.T@c,mode='economic')
                virtual=lower.T@v
                virtual-=occupied@(occupied.T@virtual)
                q,r=linalg.qr(virtual,mode='economic')
                q*=np.sign(np.diag(r))
                fixed=linalg.solve_triangular(lower.T,q,lower=False)
                fixed[:,0]=x  # Avoid a last round-trip through stiff coordinates.
                gram=float(np.max(abs(fixed.astype(wide).T@s.astype(wide)@fixed.astype(wide)-np.eye(len(e)))))
                gram_float=float(np.max(abs(fixed.T@s@fixed-np.eye(len(e)))))
                overlap=float(np.max(abs(c.astype(wide).T@s.astype(wide)@fixed.astype(wide))))
                sp=b.T@s@b
                sp=np.tril(sp)+np.tril(sp,-1).T
                residual=np.asarray(b.astype(wide).T@(h.astype(wide)@x.astype(wide)
                    -wide(energy)*(s.astype(wide)@x.astype(wide))),dtype=float)
                dual=float(np.linalg.norm(linalg.solve_triangular(linalg.cholesky(sp,lower=True),residual,lower=True)))
                change=float(np.linalg.norm(lower.T@(fixed-full[keep]),axis=0).max())
                gate(f'{ns}:{key}:whole_Gram',max(gram,gram_float),1e-10)
                gate(f'{ns}:{key}:occupied_overlap',overlap,1e-10)
                gate(f'{ns}:{key}:original_pencil_root',dual,1e-8)
                gate(f'{ns}:{key}:column_change',change,1e-8)
                gate(f'{ns}:{key}:same_leading_root',abs(energy-e[0])/min(abs(e[1:]-e[0])),.25)
                gate(f'{ns}:{key}:same_whole_branch',fixed.shape[1]==full.shape[1])
                e=e.copy()
                e[0]=energy
                full=full.copy()
                full[keep]=fixed
                arrays[f'positive_C_k{k}']=full
                arrays[f'positive_E_k{k}']=e
                index=list(map(tuple,arrays['all_orbital_keys'])).index(key)
                arrays['all_P'][index]=full[:ns,0]
                arrays['all_Q'][index]=full[ns:,0]
                energies[key]=energy
                arm['roots'].append(dict(key=key,EA_Ha=energy,EA_cm1=energy*HARTREE_CM1,
                    original_pencil_dual_Ha=dual,whole_physical_Gram=gram,
                    whole_physical_Gram_float=gram_float,core_overlap=overlap,
                    retained_electron_columns=len(e),maximum_column_change_S_norm=change))
            orbitals={tuple(map(int,key)):(p,q) for key,p,q in zip(arrays['all_orbital_keys'],arrays['all_P'],arrays['all_Q'],strict=True)}
            for configs in ((core_config,),(core_config+' 6s1',core_config+' 5d1')):
                csfs=[c for cfg in configs for c in enumerate_jj_csfs(cfg)]
                n=csfs[0].n_electrons
                gate(f'{ns}:fixed_N:{n}',all(c.n_electrons==n for c in csfs))
                ci=ci_solve(csfs,orbitals,sorted(orbitals),core._bsp_data,70,ns,
                    solver='dense',nuclear_potential=core._nuclear_potential,
                    max_determinants=32,retain_hamiltonian=True)
                h=ci.hamiltonian_snapshot.matrix_Ha
                for i,c in enumerate(csfs):
                    root=int(np.argmax(abs(ci.mixing_coefficients[i])))
                    val=[(s.n,s.kappa) for s in c.subshells if s.occupation==1]
                    expected=core.total_energy+(energies[val[0]] if val else 0)
                    error=abs(float(ci.state_energies[root])-expected)
                    gate(f'{ns}:ledger:{n}:{i}',error,1e-7)
                    arm['CI'].append(dict(electrons=n,CSF=c.label,energy_Ha=float(ci.state_energies[root]),error_Ha=error))
                gate(f'{ns}:CI_residual:{n}',float(np.linalg.norm(h@ci.mixing_coefficients-ci.mixing_coefficients*ci.state_energies,axis=0).max()),1e-8)
                arrays[f'CI_H_N{n}']=h
                arrays[f'CI_C_N{n}']=ci.mixing_coefficients
                arrays[f'CI_E_N{n}']=ci.state_energies
            arm['EA_cm1']=[energies[k]*HARTREE_CM1 for k in ((6,-1),(5,2),(5,-3))]
            arm['lines_cm1']=[arm['EA_cm1'][i]-arm['EA_cm1'][0] for i in (1,2)]
            arm['core']=next(a for a in previous['arms'] if a['ns']==ns)
            np.savez(out/f'blocks_ns{ns}.npz',**arrays)
        report['radial_EA_drift_cm1']=np.subtract(report['arms'][1]['EA_cm1'],report['arms'][0]['EA_cm1']).tolist()
        report['radial_line_drift_cm1']=np.subtract(report['arms'][1]['lines_cm1'],report['arms'][0]['lines_cm1']).tolist()
        gate('two_basis_radial_drift',max(abs(x) for x in report['radial_EA_drift_cm1']+report['radial_line_drift_cm1']),1.0)
        gate('all_input_bytes_unchanged',all(sha(ROOT/p)==h for p,h in contract['input_sha256'].items()))
        report['status']='PASS_NATIVE_DHF_BASELINE_ONLY'
    except Exception as error:
        report.update(status='FAILED',exception=repr(error))
        raise
    finally:
        signal.alarm(0)
        report['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(report['status'],len(report['controls']),flush=True)


if __name__=='__main__':
    main()
