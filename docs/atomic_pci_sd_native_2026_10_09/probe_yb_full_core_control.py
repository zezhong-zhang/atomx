"""Finite complete-core high-kappa equation control, never a production space."""
import hashlib,json,resource,sys,time,os
from pathlib import Path
import numpy as np
owner=Path(__file__).resolve().parent
root=owner.parents[1]
sys.path.insert(0,str(root/'tools'))
from build_pci_sd_backend import build
from atomx.methods.core_sd import CoreSD,plan_core_sd
from atomx.numerics.bspline.coulomb_features import CellCoulombFeatures,PolynomialCellCoulombFeatures
from atomx.potentials.nuclear import make_nuclear_potential
ref=Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
source=owner/'yb_full_run007'
out=owner/'yb_full_core_equation001'
out.mkdir(exist_ok=False)
started=time.perf_counter()
full=json.loads((source/'yb_full_plan.json').read_text())
occupied=tuple(map(tuple,full['occupied_keys']))
virtual=[]
for k in (-1,1,-2,2,-3,3,-4):
    virtual.extend([tuple(a) for a in full['virtual_keys'] if a[1]==k][:2])
virtual=tuple(virtual)
assert len(occupied)==19 and len(virtual)==14
cap=4*1024**3
contract=dict(scope='Finite equation identity anchor only. All 68 core electrons, explicit two native virtual roots per kappa. Production414 virtual request is unchanged and remains separate.',hypothesis='Exact terma4 BLAS and polynomial coordinates preserve all high-kappa complete-core signed pCI DC diagrams.',fixed='Yb original Fermi/V68/ns123 spinors, all core shells, complete finite scalar rho channels and noncanonical Fock',seed=821009,action_gate_Ha=1e-11,residual_gate_Ha=1e-10,root_gate=1e-9,memory_bytes=cap,workers=[0,1,4],load=os.getloadavg(),stop='any complete signed action or raw residual failure rejects optimization, not a reduced production substitute',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/atomx').rglob('*.py'))+sorted((root/'src/atomx').rglob('*.f90'))+[root/'tools/build_pci_sd_backend.py']})
(out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
with np.load(source/'yb_full_inputs.npz',allow_pickle=False) as data:
    workspace={k.removeprefix('workspace_'):data[k].copy() for k in data if k.startswith('workspace_')}
    workspace={k:a.item() if a.ndim==0 else a for k,a in workspace.items()}
    waves={a:(data[f'P_{a[0]}_{a[1]}'].copy(),data[f'Q_{a[0]}_{a[1]}'].copy()) for a in occupied+virtual}
    nuclear=make_nuclear_potential(70,'fermi')
    if np.max(abs(np.vectorize(nuclear,otypes=[float])(workspace['gr'])-data['nuclear_on_outer_gauss']))>1e-11:
        raise ValueError('Fermi operator failed saved identity')
problems={};actions={};roots={};times={}
for workers in (0,1,4):
    features=(CellCoulombFeatures if workers==0 else PolynomialCellCoulombFeatures)(workspace)
    plan=plan_core_sd(occupied,virtual,features.identity,term21_threads=workers,term21_mode='streamed',core_singles_blas=bool(workers))
    (out/f'plan_{workers}.json').write_text(json.dumps(plan,indent=2)+'\n')
    if plan['complete_payload_bound_bytes']+40*plan['krylov_vector_bytes']>cap:
        raise MemoryError('complete finite-core control exceeds declared cap')
    build(ref/'pCI',out/f'backend_{workers}',plan['dimensions'],ref/'gfortran',runtime_directory=ref/'toolchain/usr/lib/x86_64-linux-gnu',blas_library=owner/'openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0',term21_threads=workers,term21_mode='streamed',core_singles_blas=bool(workers))
    t=time.perf_counter()
    p=CoreSD(waves,workspace,70,workspace['ns'],occupied_keys=occupied,virtual_keys=virtual,backend_directory=out/f'backend_{workers}',nuclear_potential=nuclear,memory_bytes=cap,term21_threads=workers,term21_mode='streamed',core_singles_blas=bool(workers),radial_representation='polynomial' if workers else 'native')
    times[workers]=dict(prepare_seconds=time.perf_counter()-t)
    problems[workers]=p
    if workers==0:
        initial=p.source/p.denominators
        random=np.random.default_rng(821009).normal(size=len(initial))*1e-3
    else:
        assert np.max(abs(p.source-problems[0].source))<1e-11
        assert np.max(abs(p.denominators-problems[0].denominators))<1e-11
    t=time.perf_counter()
    actions[f'initial_{workers}']=p.action(initial)
    actions[f'random_{workers}']=p.action(random)
    times[workers]['two_actions_seconds']=time.perf_counter()-t
    print(json.dumps(dict(stage='complete signed actions',workers=workers,raw=plan['raw_amplitudes'],times=times[workers])),flush=True)
errors={str(n):{k:float(np.max(abs(actions[f'{k}_{n}']-actions[f'{k}_0']))) for k in ('initial','random')} for n in (1,4)}
np.savez_compressed(out/'signed_actions.npz',initial=initial,random=random,**actions)
(out/'action_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
assert max(e for row in errors.values() for e in row.values())<1e-11
for n,p in problems.items():
    result=p.solve(residual_tolerance_Ha=1e-10,restart=30,maxiter=100)
    roots[str(n)]=dict(correlation_energy_Ha=result.correlation_energy_Ha,residual_max_Ha=result.residual_max_Ha,solve_seconds=result.wall_seconds,matvecs=result.matrix_vector_products)
    if n==0: baseline=result.amplitudes.copy()
    else:
        roots[str(n)]['root_error']=float(np.max(abs(result.amplitudes-baseline)))
        assert roots[str(n)]['root_error']<1e-9
    np.savez_compressed(out/f'solution_{n}.npz',amplitudes=result.amplitudes)
    print(json.dumps(dict(stage='finite-core solve',workers=n,**roots[str(n)])),flush=True)
result=dict(errors_Ha=errors,roots=roots,times=times,seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,production_space_converged=False,experimental_gate=False,timing_scope='diagnostic single observations only; not paired controlled performance')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
