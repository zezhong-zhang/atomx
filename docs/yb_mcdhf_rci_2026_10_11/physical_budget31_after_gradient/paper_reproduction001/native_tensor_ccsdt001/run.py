"""Source-bound native tensor replay, then complete5d-pair Yb expansion."""
import argparse
import hashlib
import json
import os
import resource
import signal
import tarfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'pyproject.toml').is_file())
OLD = HERE.parent/'yb_native_rccsdt001'
DHF = HERE.parent/'yb_dhf_table001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'src/atomx').rglob('*.py'))}


def prepare():
    import sys
    sys.path.insert(0, str(ROOT/'src'))
    from audit import sector

    from atomx.methods.tensor_coupled_cluster import resource_plan
    labels = [(n, k, m) for n, k in ((5, -1), (5, 1), (6, -1), (5, 2), (5, -3))
              for m in range(-2*abs(k)+1, 2*abs(k), 2)]
    mapping = sources()
    identity = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    inputs = [Path(__file__), HERE/'audit.py', HERE/'equation_reference.json',
              ROOT/'src/atomx/physics/csf/CCSDT_EQUATIONS_LICENSE.txt',
              ROOT/'tests/test_tensor_coupled_cluster.py',
              OLD/'source.json', OLD/'actual001/integrals.npz', OLD/'actual001/operator.json',
              OLD/'actual001/CCSDT_R3.npz', OLD/'actual001/CCSDT_R3.json',
              DHF/'actual004/core_ns180.h5', DHF/'actual004/blocks_ns180.npz']
    save(HERE/'source.json', dict(files=mapping, source_map_sha256=identity))
    with tarfile.open(HERE/'source.tar.gz', 'w:gz') as archive:
        for name in mapping:
            archive.add(ROOT/name, arcname=name, recursive=False)
    contract = dict(hypothesis='A complete native CCSDT/EA tensor realization reproduces the prior directed Yb finite pencil and permits adding the entire5d5/2 multiplet without full-Fock production enumeration. Independent sector FCI/expm quantifies same-model omitted ranks; experiment improvement is measured, not an admission gate.',
                    fixed=dict(Z=70, nucleus='same empiricalA171 Fermi', radial='same owned ns180/ks9/R500/RKB bank', correlated_core='5s2+5p1/2^2,4electrons', frozen='64electrons including4f14 and5p3/2^4', virtual='complete6s1/2+5d3/2+5d5/2,12magneticspinors', operator='same native DC with full frozen contractions; noBreit/QED/recoil', origin='subtract exactly common68e HF reference from both sectors; retain original origin', comparisons=['CCSD/R2', 'CCSD/R3', 'CCSDT/R2', 'CCSDT/R3', 'independent full sector FCI'], sectors_m2=[-5, -3, -1, 1, 3, 5], parity=0, matching='unique normalized physical bare6s/5d radial projector weight>=0.5, target imag<1e-10Ha andJ2; all2/4/6 partners, no experimental root matching', solver='simultaneous Krylov nonlinear SD or SDT; four fixed coupling continuation points; no postfailure retry', scope='EA-CCSDT/R3, not valence-universal FS-RCCSDT or full paper model'),
                    acceptance=dict(replay_core_residual_Ha=1e-10, replay_directed_pencil_Ha=1e-9, native_integral_restriction_Ha=1e-10, amplitude_Ha=1e-10, complete_directed_sector_Hbar_Ha=1e-9, symmetry_Ha=1e-10, eigen_Ha=1e-10, J2_defect=1e-7, multiplet_spread_Ha=1e-8, nonzero_T3_min=1e-12, full_H='diagnostic; never force omitted-rank residual to zero'),
                    resources=dict(cpu=27, threads=1, AS_bytes=8*1024**3, wall_seconds=1200, load1_max=24, available_min_bytes=16*1024**3, tensor_memory_MiB=512, batch_size=4, full_core_count=1820, full_attachment_count=4368, independent_core_M0_even=len(sector(labels, 4, 0)), independent_attachment_sector_counts={str(m):len(sector(labels,5,m)) for m in (-5,-3,-1,1,3,5)}, tensor_plan=resource_plan(4,12,batch_size=4), scalar_only_all68_plan=resource_plan(68,12,batch_size=4)),
                    stopping='First failed source/input/operator/equation/multiplet/resource gate stops; preserve evidence. No altered shell, rank, tolerance, experimental fit, automatic retry or borrowed correction.', source_map_sha256=identity, inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    save(HERE/'contract.json', contract)
    print(json.dumps(dict(status='PREPARED', source_files=len(mapping), resources=contract['resources']), indent=2))


def run():
    import numpy as np
    from audit import (
        cluster,
        coordinates,
        hamiltonian,
        j_squared,
        quantum,
        sector,
        select,
    )
    from scipy.linalg import eig, eigh, expm

    from atomx.constants import HARTREE_CM1
    from atomx.methods.dirac_fock import SCFResult
    from atomx.methods.tensor_coupled_cluster import (
        CCSDTTensorOperator,
        TensorCoreResult,
    )
    from atomx.physics.csf.spinor_integrals import (
        SpinorIntegrals,
        build_spinor_integrals,
    )
    from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile

    contract = json.loads((HERE/'contract.json').read_text())
    source = json.loads((HERE/'source.json').read_text())
    assert sources() == source['files']
    assert all(sha(ROOT/n) == h for n, h in contract['inputs'].items())
    assert os.getloadavg()[0] <= 24
    available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    assert available >= 16*1024**3
    assert all(os.environ.get(name) == '1' for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'])
    os.sched_setaffinity(0, {27})
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    def alarm(*_):
        raise TimeoutError('declared1200-second native tensor pilot limit')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1200)
    out = HERE/'actual001'
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', controls=[], sectors=[], source_map_sha256=source['source_map_sha256'], contract_sha256=sha(HERE/'contract.json'))
    started = time.monotonic()
    def gate(name, measured, cap=None):
        value = float(measured) if cap is not None else bool(measured)
        passed = bool(value <= cap) if cap is not None else value
        report['controls'].append(dict(name=name, measured=value, cap=cap, passed=passed))
        if not passed:
            raise AssertionError((name, value, cap))
    try:
        oldsource = json.loads((OLD/'source.json').read_text())
        gate('all359_previous_production_files_unchanged', all(source['files'].get(k) == h for k,h in oldsource['files'].items()))
        oldledger = json.loads((OLD/'actual001/operator.json').read_text())
        with np.load(OLD/'actual001/integrals.npz') as z:
            oldobj = SpinorIntegrals(tuple(map(tuple,oldledger['labels'])), 4, z['h'], z['fock'], z['g'], oldledger['shifted_frozen_energy_Ha'], 0., oldledger['identity'])
        with np.load(OLD/'actual001/CCSDT_R3.npz') as z:
            old = {k:z[k] for k in z.files}
        opold = CCSDTTensorOperator(oldobj)
        _, corephases, _ = coordinates(opold, False)
        vector = old['core_amplitudes']*corephases
        gate('prior_same_T_complete_tensor_residual', np.max(abs(opold.core_residual(vector))), 1e-10)
        oldrow = json.loads((OLD/'actual001/CCSDT_R3.json').read_text())
        energy = opold.core_energy(vector)
        gate('prior_same_T_core_energy', abs(energy-oldrow['core_energy_relative_Ha']), 1e-10)
        coreold = TensorCoreResult(vector, energy, energy, float(np.max(abs(opold.core_residual(vector)))), 0, opold.operator_sha256)
        masks, phases, _ = coordinates(opold, True)
        lookup = {int(mask):i for i,mask in enumerate(old['projected_masks'])}
        order = [lookup[int(mask)] for mask in masks]
        expected = phases[:,None]*old['projected_H'][np.ix_(order,order)]*phases[None,:]
        replay = opold.attachment_matrix(coreold, batch_size=4)
        gate('prior_all_directed_tensor_EA_entries', np.max(abs(replay-expected)), 1e-9)
        solvedold = opold.solve_core()
        gate('prior_tensor_solved_core_energy', abs(solvedold.total_energy_Ha-energy), 1e-10)
        np.savez(out/'prior_replay.npz', tensor_H=replay, finite_H_in_tensor_coordinates=expected, core_amplitudes=solvedold.amplitudes)
        print('PRIOR_REPLAY_PASS', flush=True)

        radial = SCFResult.load(DHF/'actual004/core_ns180.h5')
        gate('same_A171_nucleus', radial.solver_diagnostics['_nuclear_profile'] == nuclear_profile(70,'fermi',171))
        radial._nuclear_potential = make_nuclear_potential(70,'fermi',171)
        with np.load(DHF/'actual004/blocks_ns180.npz') as z:
            orbitals = {tuple(map(int,k)):(p,q) for k,p,q in zip(z['all_orbital_keys'], z['all_P'], z['all_Q'], strict=True)}
        occupied, virtual = ((5,-1),(5,1)), ((6,-1),(5,2),(5,-3))
        frozen = tuple(k for k in radial.orbitals_by_subshell() if k not in occupied)
        original = build_spinor_integrals(orbitals, radial._bsp_data,70,180, occupied_keys=occupied,virtual_keys=virtual,frozen_keys=frozen,nuclear_potential=radial._nuclear_potential,max_integral_bytes=32*1024**2,metric_tolerance=1e-10)
        gate('complete16_spinors_4core_64frozen', len(original.labels)==16 and original.noccupied==4 and sum(2*abs(k) for n,k in frozen)==64)
        origin = original.reference_energy_Ha
        identity = dict(original.identity)
        identity.update(source_map_sha256=source['source_map_sha256'], energy_origin_Ha=origin, frozen_partition='64frozen including4f14;4correlated5s/5p1/2')
        obj = SpinorIntegrals(original.labels,4,original.one_body_Ha,original.fock_Ha,original.packed_eri_Ha,original.frozen_energy_Ha-origin,0.,identity)
        gate('old10_spinors_are_exact_prefix', obj.labels[:10]==oldobj.labels)
        gate('nested_one_body', np.max(abs(obj.one_body_Ha[:10,:10]-oldobj.one_body_Ha)),1e-10)
        gate('nested_fock', np.max(abs(obj.fock_Ha[:10,:10]-oldobj.fock_Ha)),1e-10)
        gate('nested_origin', abs(origin-oldledger['absolute_energy_origin_Ha']),1e-10)
        gate('nested_signed_ERI', max(abs(obj.eri(p,q,r,s)-oldobj.eri(p,q,r,s)) for p in range(10) for q in range(10) for r in range(10) for s in range(10)),1e-10)
        save(out/'operator.json', dict(labels=obj.labels,identity=identity,absolute_energy_origin_Ha=origin,shifted_frozen_energy_Ha=obj.frozen_energy_Ha))
        np.savez(out/'integrals.npz', h=obj.one_body_Ha,fock=obj.fock_Ha,g=obj.packed_eri_Ha)
        op = CCSDTTensorOperator(obj)
        coremask, corephase, coreops = coordinates(op, False)
        attached_masks, attached_phase, _ = coordinates(op, True)
        cm = sector(obj.labels,4,0)
        hc, outside = hamiltonian(obj.one_body_Ha.real,op.g,obj.frozen_energy_Ha,cm)
        gate('independent_core_sector_closed', outside,1e-10)
        gate('independent_core_Hermitian', np.max(abs(hc-hc.T)),1e-10)
        ec, vc = eigh(hc)
        jc = j_squared(obj.labels,cm)
        gate('fullCI_core_J0', np.linalg.norm(jc@vc[:,0]),1e-7)
        gate('fullCI_core_equation', np.max(abs(hc@vc-vc*ec)),1e-10)
        np.savez(out/'core_FCI.npz', masks=cm,H=hc,E=ec,C=vc,J2=jc)
        cores = {}
        for rank,name in [(2,'CCSD'),(3,'CCSDT')]:
            print('SOLVE_CORE',name,flush=True)
            result = op.solve_core(retained_rank=rank)
            gate(name+':core_equation',result.amplitude_residual_Ha,1e-10)
            t, outside = cluster(result.amplitudes,coreops,cm)
            gate(name+':T_preserves_M_parity',outside,1e-10)
            ep, em = expm(t), expm(-t)
            reference = cm.index(15)
            hbar = em@hc@ep
            gate(name+':independent_core_energy',abs(hbar[reference,reference]-result.total_energy_Ha),1e-10)
            wave = ep[:,reference]
            wave /= np.linalg.norm(wave)
            gate(name+':core_J0',np.linalg.norm(jc@wave),1e-7)
            gate(name+':nonzero_T3' if rank==3 else name+':zero_T3',np.linalg.norm(result.amplitudes[op.core.blocks[2][0]])>1e-12 if rank==3 else np.all(result.amplitudes[op.core.blocks[2][0]]==0))
            coremeta = dict(energy_Ha=result.total_energy_Ha,correlation_Ha=result.correlation_energy_Ha,residual_Ha=result.amplitude_residual_Ha,full_H_residual_Ha=float(np.linalg.norm(hc@wave-result.total_energy_Ha*wave)),FCI_error_Ha=float(result.total_energy_Ha-ec[0]),T3_norm=float(np.linalg.norm(result.amplitudes[op.core.blocks[2][0]])),evaluations=result.evaluations)
            report.setdefault('cores',{})[name] = coremeta
            np.savez(out/(name+'_core.npz'),amplitudes=result.amplitudes,physical_C=wave)
            cores[name]=result
        print('CORE_SOLVES_PASS',flush=True)
        alltargets = {name:[] for name in ('FCI','CCSD_R2','CCSD_R3','CCSDT_R2','CCSDT_R3')}
        for m2 in contract['fixed']['sectors_m2']:
            print('SECTOR',m2,flush=True)
            ma = sector(obj.labels,5,m2)
            ha, outside = hamiltonian(obj.one_body_Ha.real,op.g,obj.frozen_energy_Ha,ma)
            gate(f'M{m2}:independent_sector_closed',outside,1e-10)
            gate(f'M{m2}:H_Hermitian',np.max(abs(ha-ha.T)),1e-10)
            ef,vf = eigh(ha)
            ja = j_squared(obj.labels,ma)
            gate(f'M{m2}:FCI_equation',np.max(abs(ha@vf-vf*ef)),1e-10)
            ft = select(ef,vf,ma,obj.labels,ja,m2,4)
            for target,v in ft.items():
                gate(f'M{m2}:FCI_J2:{target}',v['J2_defect'],1e-7)
                alltargets['FCI'].append(dict(target=target,m2=m2,**v))
            np.savez(out/f'M{m2}_FCI.npz',masks=ma,H=ha,E=ef,C=vf,J2=ja)
            indices = np.array([i for i,mask in enumerate(attached_masks) if quantum(obj.labels,int(mask))==(m2,0)])
            full_lookup = {mask:i for i,mask in enumerate(ma)}
            rows = np.array([full_lookup[int(attached_masks[i])] for i in indices])
            signs = attached_phase[indices]
            for name,core in cores.items():
                pencil = np.empty((len(indices),len(indices)))
                forbidden = np.ones(op.attachment.size,dtype=bool)
                forbidden[indices]=False
                leakage = 0.
                for begin in range(0,len(indices),4):
                    chunk=indices[begin:begin+4]
                    vectors=np.zeros((op.attachment.size,len(chunk)))
                    vectors[chunk,np.arange(len(chunk))]=1
                    product=op.attachment_action(core,vectors)
                    leakage=max(leakage,float(np.max(abs(product[forbidden]))))
                    pencil[:,begin:begin+len(chunk)]=product[indices]
                gate(f'M{m2}:{name}:symmetry_leakage',leakage,1e-10)
                t,outside=cluster(core.amplitudes,coreops,ma)
                gate(f'M{m2}:{name}:T_sector_closed',outside,1e-10)
                ep,em=expm(t),expm(-t)
                transformed=em@ha@ep-core.total_energy_Ha*np.eye(len(ma))
                expected=signs[:,None]*transformed[np.ix_(rows,rows)]*signs[None,:]
                gate(f'M{m2}:{name}:all_directed_Hbar',np.max(abs(pencil-expected)),1e-9)
                for rank in (2,3):
                    keep=np.flatnonzero(indices<op.attachment.blocks[rank-1][0].stop)
                    projected=pencil[np.ix_(keep,keep)]
                    roots,vectors=eig(projected)
                    gate(f'M{m2}:{name}:R{rank}:projected_equation',np.max(abs(projected@vectors-vectors*roots)),1e-10)
                    embed=np.zeros((len(ma),len(roots)),dtype=complex)
                    embed[rows[keep]]=signs[keep,None]*vectors
                    physical=ep@embed
                    physical/=np.linalg.norm(physical,axis=0)
                    targets=select(roots+core.total_energy_Ha,physical,ma,obj.labels,ja,m2,4)
                    for target,v in targets.items():
                        gate(f'M{m2}:{name}:R{rank}:J2:{target}',v['J2_defect'],1e-7)
                        wave=physical[:,v['root']]
                        v['full_H_residual_Ha']=float(np.linalg.norm(ha@wave-v['energy_Ha']*wave))
                        alltargets[name+f'_R{rank}'].append(dict(target=target,m2=m2,**v))
                    np.savez(out/f'M{m2}_{name}_R{rank}.npz',indices=indices[keep],projected_H=projected,E=roots,C=vectors,physical_C=physical,full_masks=ma)
            report['sectors'].append(dict(m2=m2,full_FCI_dimension=len(ma),complete_R3_dimension=len(indices)))
            save(out/'partial.json',report)
        report['targets']=alltargets
        report['lines_cm1']={}
        for name,entries in alltargets.items():
            energies={}
            for target,count in [('S',2),('D3',4),('D5',6)]:
                values=[v['energy_Ha'] for v in entries if v['target']==target]
                gate(name+':complete_partners:'+target,len(values)==count)
                gate(name+':multiplet_spread:'+target,np.ptp(values),1e-8)
                energies[target]=float(np.mean(values))
            report['lines_cm1'][name]={target:(energies[target]-energies['S'])*HARTREE_CM1 for target in ('D3','D5')}
        report['same_model_T3_effects_cm1']={f'at_R{rank}':{target:report['lines_cm1'][f'CCSDT_R{rank}'][target]-report['lines_cm1'][f'CCSD_R{rank}'][target] for target in ('D3','D5')} for rank in (2,3)}
        gate('source_unchanged',sources()==source['files'])
        gate('inputs_unchanged',all(sha(ROOT/n)==h for n,h in contract['inputs'].items()))
        report['status']='PASS_NATIVE_TENSOR_YB_5D_PAIR_PILOT'
    except Exception:
        report.update(status='FAILED',exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save(out/'result.json',report)
        print('END',report['status'],report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else run()
