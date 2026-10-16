"""Same finite complete-core roots certified with original signed equations."""
import hashlib,json,resource,time,os
from pathlib import Path
import numpy as np
from atomx.methods.core_sd import CoreSD
from atomx.potentials.nuclear import make_nuclear_potential
owner=Path(__file__).resolve().parent
root=owner.parents[1]
source=owner/'yb_full_run007'
previous=owner/'yb_full_core_equation001'
out=owner/'yb_full_core_equation002'
out.mkdir(exist_ok=False)
started=time.perf_counter()
plan=json.loads((previous/'plan_0.json').read_text())
occupied,virtual=tuple(map(tuple,plan['occupied_keys'])),tuple(map(tuple,plan['virtual_keys']))
errors=json.loads((previous/'action_errors.json').read_text())
assert max(e for row in errors.values() for e in row.values())<1e-11
contract=dict(scope='Finite all-68-core equation control; fourteen explicit virtual radial roots, not a production/convergence/spectrum result.',fixed='Original Fermi/V68/ns123, same complete finite virtual keys, signed noncanonical pCI equations.',action_evidence=errors,source_action_sha256=hashlib.sha256((previous/'signed_actions.npz').read_bytes()).hexdigest(),acceptance=dict(original_raw_residual_Ha=1e-10,candidate_raw_residual_Ha=1e-10,signed_energy_error_Ha=1e-11),root_validation='Each candidate root is checked by one complete ORIGINAL pCI action. Reuse only an exactly bitwise identical vector. This avoids redundant original GMRES iterations without weakening the equation gate.',stopping='Any original equation or signed energy mismatch rejects candidate.',memory_bytes=4*1024**3,load=os.getloadavg(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/atomx').rglob('*.py'))+sorted((root/'src/atomx').rglob('*.f90'))})
(out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
with np.load(source/'yb_full_inputs.npz',allow_pickle=False) as data:
    workspace={k.removeprefix('workspace_'):data[k].copy() for k in data if k.startswith('workspace_')}
    workspace={k:a.item() if a.ndim==0 else a for k,a in workspace.items()}
    waves={a:(data[f'P_{a[0]}_{a[1]}'].copy(),data[f'Q_{a[0]}_{a[1]}'].copy()) for a in occupied+virtual}
nuclear=make_nuclear_potential(70,'fermi')
problems={};roots={};vectors={}
for n in (1,4,0):
    p=CoreSD(waves,workspace,70,workspace['ns'],occupied_keys=occupied,virtual_keys=virtual,backend_directory=previous/f'backend_{n}',nuclear_potential=nuclear,memory_bytes=4*1024**3,term21_threads=n,term21_mode='streamed',core_singles_blas=bool(n),radial_representation='polynomial' if n else 'native')
    problems[n]=p
    if n:
        result=p.solve(residual_tolerance_Ha=1e-10,restart=30,maxiter=100)
        vectors[n]=result.amplitudes.copy()
        roots[str(n)]=dict(correlation_energy_Ha=result.correlation_energy_Ha,candidate_residual_max_Ha=result.residual_max_Ha,solve_seconds=result.wall_seconds,matvecs=result.matrix_vector_products)
        np.savez_compressed(out/f'solution_{n}.npz',amplitudes=result.amplitudes)
        print(json.dumps(dict(stage='candidate root',workers=n,**roots[str(n)])),flush=True)
    else:
        for cand in (1,4):
            assert np.max(abs(p.source-problems[cand].source))<1e-11
            assert np.max(abs(p.denominators-problems[cand].denominators))<1e-11
original=problems[0]
original.set_diagram_progress(True)
last=None;last_residual=None
for n,t in vectors.items():
    if last is not None and np.array_equal(last,t):
        residual=last_residual
        roots[str(n)]['original_action_reused']='bitwise identical full amplitude vector'
    else:
        print(json.dumps(dict(stage='original root residual check',workers=n)),flush=True)
        residual=original.action(t)-original.source
        last=t.copy();last_residual=residual.copy()
    maximum=float(np.max(abs(residual)))
    energy=original.energy(t)
    energy_error=energy-roots[str(n)]['correlation_energy_Ha']
    roots[str(n)].update(original_residual_max_Ha=maximum,original_correlation_energy_Ha=energy,signed_energy_error_Ha=energy_error)
    np.savez_compressed(out/f'original_residual_{n}.npz',residual=residual)
    (out/f'root_{n}.json').write_text(json.dumps(roots[str(n)],indent=2)+'\n')
    print(json.dumps(dict(stage='original equation gate',workers=n,**roots[str(n)])),flush=True)
    assert maximum<=1e-10 and abs(energy_error)<=1e-11
result=dict(roots=roots,seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,production_space_converged=False,experimental_gate=False,timing_scope='Single diagnostic observations; no controlled speedup claim')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
