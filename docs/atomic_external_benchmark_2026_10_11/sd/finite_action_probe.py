"""Complete Na+ public CoreSD/ValenceSD maps versus pinned pCI-derived original diagrams."""
from __future__ import annotations
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

OWNER=Path(__file__).resolve().parent
ROOT=OWNER.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.methods.valence_sd import ValenceSD, plan_valence_sd
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures
from build_pci_valence_backend import build_valence

BASE=ROOT/'docs/atomic_pci_sd_native_2026_10_09'
REF=Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
CONTRACT=json.loads((OWNER.parent/'contract.json').read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    out=OWNER/'finite_action001';out.mkdir(exist_ok=False)
    cap=CONTRACT['timing_contract']['max_per_process_address_bytes']
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    guard=threading.Timer(600,lambda:os._exit(124));guard.daemon=True;guard.start()
    occupied=((1,-1),(2,-1),(2,1),(2,-2))
    virtual=((3,-1),(4,-1),(3,1),(4,1),(3,-2),(4,-2),(3,2),(3,-3))
    keys=((3,-1),(3,1),(3,-2),(3,2),(3,-3))
    local=dict(hypothesis='Current exact streamed term2 preserves complete public fixed-energy affine map and total response versus preserved pCI-derived original diagrams',
        fixed='Na+ Z11 point nucleus ns38 ks7; all 10 core electrons; all declared 3s4s3p4p3d radial virtuals with every magnetic partner; all five external s/p/d columns; fixed signed numerator and derivative seed7411041; same solved complete core amplitudes',
        gate_Ha=1e-11,relative_gate=1e-10,workers=[1,2,4,8],rounds=5,repeats=10,
        orders=[['original',1,2,4,8],[8,4,2,1,'original'],[2,'original',8,1,4],[4,1,8,'original',2],['original',8,2,4,1]],
        scope='Complete public ValenceSD.evaluate for all 5 columns, including all retained affine source/Fock/diagram terms and total supplied response, context/channel construction and native C/Python copies; excludes backend build/native integral setup/SCF/core solve',
        oracle_lineage='pCI-derived pinned original scalar DC diagram suite compiled by Atomx native adapter. Shared upstream formulas/radial service; not independent standalone pCI production software or MPI scaling.',
        end_to_end_solve_claim=False,original_source_sha256=digest(REF/'pCI/lib/all-order/valsd-ci.f'))
    write(out/'contract.json',local)
    sources=sorted((ROOT/'src/atomx').rglob('*.py'))+sorted((ROOT/'src/atomx/methods').glob('_pci*.f90'))+[Path(__file__),ROOT/'tools/build_pci_sd_backend.py',ROOT/'tools/build_pci_valence_backend.py']
    identities={}
    for p in sources:
        rel=p.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);identities[str(rel)]=digest(p)
    write(out/'source_sha256.json',identities)
    started=time.perf_counter()
    atom=DFSolver(Z=11,subshells=[Subshell(n,k,2*abs(k)) for n,k in occupied],ns=38,ks=7).solve()
    waves={k:atom.orbitals_by_subshell()[k][:2] for k in occupied}
    for kappa in sorted({k[1] for k in virtual}):
        _,states=build_virtual_basis(atom,kappa,e_max=50,potential_mode='VN',orthogonal_to_occupied=True)
        matches=[k for k in virtual if k[1]==kappa]
        if len(states)<len(matches):raise AssertionError('complete declared virtual multiplets unavailable')
        for k,state in zip(matches,states):waves[k]=state[:2]
    np.savez_compressed(out/'physical_inputs.npz',**{f'{c}_{k[0]}_{k[1]}':v[i] for k,v in waves.items() for i,c in enumerate(('P','Q'))},**{f'workspace_{k}':v for k,v in atom._bsp_data.items() if np.isscalar(v) or isinstance(v,np.ndarray)})
    options=dict(term21_threads=1,term21_mode='streamed',core_singles_blas=True,core_single_to_doubles_blas=True)
    plan=plan_core_sd(occupied,virtual,PolynomialCellCoulombFeatures(atom._bsp_data).identity,**options)
    plans={n:plan_valence_sd(plan,keys,valence_term2_blas=n!='original',valence_term2_threads=1 if n=='original' else n) for n in ('original',1,2,4,8)}
    bound=sum(p['complete_payload_bound_bytes'] for p in plans.values())
    preflight=dict(complete_all_five_backend_payload_bytes=bound,cap_bytes=cap,
        available_bytes=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,
        minimum_headroom_bytes=8*1024**3,virtual_multiplets=len(virtual),core_electrons=10,dimensions=plan['dimensions'],raw_core_coordinates=plan['raw_amplitudes'],raw_valence_coordinates={str(k):int(v) for k,v in plans['original']['raw_amplitudes'].items()},affinity=list(os.sched_getaffinity(0)))
    preflight['passed']=bound<cap and preflight['available_bytes']>=cap+8*1024**3
    write(out/'preflight.json',preflight)
    if not preflight['passed']:raise MemoryError('five complete physical backend payloads exceed fixed cap')
    blas=BASE/'openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0'
    problems={};cores={};setup={};core_control={};reference_result=None
    for n in ('original',1,2,4,8):
        t=time.perf_counter();directory=out/f'backend_{n}'
        build_valence(REF/'pCI',directory,plan['dimensions'],REF/'gfortran',nxv=plans[n]['nxv'],nvh=plans[n]['nvh'],
            runtime_directory=REF/'toolchain/usr/lib/x86_64-linux-gnu',blas_library=blas,
            valence_term2_blas=n!='original',valence_term2_threads=1 if n=='original' else n,**options)
        problem=CoreSD(waves,atom._bsp_data,11,38,occupied_keys=occupied,virtual_keys=virtual,nuclear_potential=atom._nuclear_potential,backend_directory=directory,radial_representation='polynomial',memory_bytes=cap,**options)
        if reference_result is None:
            reference_result=problem.solve(residual_tolerance_Ha=1e-11)
            write(out/'core_result.json',dict(reference_energy_Ha=reference_result.reference_energy_Ha,correlation_energy_Ha=reference_result.correlation_energy_Ha,raw_residual_Ha=reference_result.residual_max_Ha,matvecs=reference_result.matrix_vector_products,provenance=dict(reference_result.provenance)))
            np.savez_compressed(out/'core_result.npz',amplitudes=reference_result.amplitudes,source=problem.source,denominators=problem.denominators)
        result=replace(reference_result,provenance=dict(problem.provenance))
        residual=problem.action(result.amplitudes)-problem.source
        core_control[str(n)]=dict(maximum_full_raw_residual_Ha=float(np.max(abs(residual))),same_physical_source=bool(np.array_equal(problem.source,cores['original'].source)) if cores else True,
            same_denominators=bool(np.array_equal(problem.denominators,cores['original'].denominators)) if cores else True)
        if core_control[str(n)]['maximum_full_raw_residual_Ha']>1e-11 or not all(core_control[str(n)][k] for k in ('same_physical_source','same_denominators')):raise AssertionError('complete fixed core wave or source changed')
        cores[n]=problem;problems[n]=ValenceSD(problem,result,valence_keys=keys);setup[str(n)]=time.perf_counter()-t
    input_arrays={};rng=np.random.default_rng(7411041);energies={}
    for key in keys:
        count=plans['original']['raw_amplitudes'][key]
        input_arrays[key]=(rng.normal(size=count)*1e-3,rng.normal(size=count)*1e-4)
        i=next(i for i,x in enumerate(cores['original'].integrals.labels) if x[:2]==key)
        energies[key]=float(cores['original'].integrals.fock_Ha[i,i].real+.075)
    np.savez_compressed(out/'numerator_inputs.npz',**{f'{kind}_{k[0]}_{k[1]}':v[i] for k,v in input_arrays.items() for i,kind in enumerate(('signed','response'))})
    references={};warmups=[]
    def batch(n,repeats):
        returned={}
        for _ in range(repeats):
            for key in keys:
                v,dv=input_arrays[key]
                returned[key]=problems[n].evaluate(key,energies[key],v,numerator_derivative=dv)
        return returned
    def compare(actual):
        errors=[float(np.max(abs(a-b))) for k in keys for a,b in zip(actual[k],references[k])]
        relative=[float(np.max(abs(a-b)))/max(float(np.max(abs(b))),1e-300) for k in keys for a,b in zip(actual[k],references[k])]
        return max(errors),max(relative)
    for n in ('original',1,2,4,8):
        with open(CONTRACT['timing_contract']['lock_file'],'a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX);t=time.perf_counter();actual=batch(n,1);seconds=time.perf_counter()-t
        if n=='original':references=actual
        err,rel=compare(actual)
        np.savez_compressed(out/f'warmup_{n}.npz',**{f'{kind}_{k[0]}_{k[1]}':arr for k,v in actual.items() for kind,arr in zip(('singles','doubles','singles_prime','doubles_prime'),v)})
        warmups.append(dict(backend=n,seconds=seconds,maximum_signed_error_Ha=err,relative_error=rel))
        if err>1e-11 or rel>1e-10:raise AssertionError('fresh complete affine map gate failed')
    observations=[]
    for round_id,order in enumerate(local['orders']):
        for n in order:
            workers=1 if n=='original' else n
            with open(CONTRACT['timing_contract']['lock_file'],'a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX);before=os.getloadavg();t=time.perf_counter();actual=batch(n,local['repeats']);seconds=time.perf_counter()-t;after=os.getloadavg()
            err,rel=compare(actual)
            valid=max(before+after)+workers<=len(os.sched_getaffinity(0)) and max(abs(a-b) for a,b in zip(before,after))<=4
            row=dict(round=round_id,backend=n,workers=workers,batch_seconds=seconds,per_five_column_map_seconds=seconds/local['repeats'],load_before=before,load_after=after,timing_guard_pass=valid,maximum_signed_error_Ha=err,maximum_relative_error=rel)
            observations.append(row);write(out/'observations.json',observations)
            np.savez_compressed(out/f'round_{round_id}_{n}.npz',**{f'{kind}_{k[0]}_{k[1]}':arr for k,v in actual.items() for kind,arr in zip(('singles','doubles','singles_prime','doubles_prime'),v)})
            print(json.dumps(row),flush=True)
            if err>1e-11 or rel>1e-10:raise AssertionError('complete public map/response gate failed')
    all_guard=all(r['timing_guard_pass'] for r in observations)
    scaling=[]
    for n in ('original',1,2,4,8):
        rows=[x for x in observations if x['backend']==n]
        ratios=[next(x['batch_seconds'] for x in observations if x['round']==r['round'] and x['backend']==1)/r['batch_seconds'] for r in rows]
        vs_original=[next(x['batch_seconds'] for x in observations if x['round']==r['round'] and x['backend']=='original')/r['batch_seconds'] for r in rows]
        scaling.append(dict(backend=n,median_per_five_column_map_seconds=float(np.median([r['per_five_column_map_seconds'] for r in rows])),observed_speedups_vs_current_one=ratios,speedup=float(np.median(ratios)) if all_guard else None,efficiency=float(np.median(ratios))/n if all_guard and n!='original' else None,observed_vs_original_diagrams=vs_original,versus_original_diagrams_speedup=float(np.median(vs_original)) if all_guard else None))
    changed={str(p.relative_to(ROOT)):digest(p) for p in sources if digest(p)!=identities[str(p.relative_to(ROOT))]}
    result=dict(status='complete public signed maps and responses passed',full_numerical_gate=not changed,maximum_signed_error_Ha=max(r['maximum_signed_error_Ha'] for r in observations),maximum_relative_error=max(r['maximum_relative_error'] for r in observations),timing_guard_pass=all_guard,scaling=scaling,setup_seconds=setup,scf_and_virtual_seconds=sum([started and 0]),complete_prepare_wall_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope=local['scope'],core_control=core_control,core_provenance={str(n):dict(c.provenance) for n,c in cores.items()},physical_inputs_sha256=digest(out/'physical_inputs.npz'),numerator_inputs_sha256=digest(out/'numerator_inputs.npz'),source_changes=changed,full_solve_speedup_claim=False,independent_standalone_pci_claim=False,pci_oracle_lineage=local['oracle_lineage'],warmups=warmups)
    write(out/'result.json',result)
    if changed:raise AssertionError('source epoch changed during run')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        p=OWNER/'finite_action001'
        if p.exists():write(p/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise
