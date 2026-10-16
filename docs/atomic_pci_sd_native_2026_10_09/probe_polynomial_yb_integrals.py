"""Signed native Yb radial control, full retained input and original rules."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from atomx.numerics.bspline.convol import ConvolCache,compute_slater_rk_cell
from atomx.numerics.bspline.coulomb_features import CellCoulombFeatures,PolynomialCellCoulombFeatures
owner=Path(__file__).resolve().parent
source=owner/'yb_full_run007'
out=owner/'yb_polynomial_integrals001'
out.mkdir(exist_ok=False)
started=time.perf_counter()
plan=json.loads((source/'yb_full_plan.json').read_text())
keys=tuple(map(tuple,plan['occupied_keys']+plan['virtual_keys']))
with np.load(source/'yb_full_inputs.npz',allow_pickle=False) as data:
    workspace={k.removeprefix('workspace_'):data[k].copy() for k in data if k.startswith('workspace_')}
    workspace={k:a.item() if a.ndim==0 else a for k,a in workspace.items()}
    waves={a:(data[f'P_{a[0]}_{a[1]}'].copy(),data[f'Q_{a[0]}_{a[1]}'].copy()) for a in keys}
contract=dict(hypothesis='exact polynomial density coordinates preserve native signed radial values',fixed='all saved Yb Fermi/V68/ns123 spinors and original native quadrature',input_sha256=hashlib.sha256((source/'yb_full_inputs.npz').read_bytes()).hexdigest(),feature_source_sha256=hashlib.sha256(Path(__import__('atomx.numerics.bspline.coulomb_features',fromlist=['x']).__file__).read_bytes()).hexdigest(),seed=507901,quartets=256,integral_gate_Ha=1e-11,stop='any signed value above unchanged gate rejects compressed representation')
(out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
original=CellCoulombFeatures(workspace)
compressed=PolynomialCellCoulombFeatures(workspace)
(out/'resources.json').write_text(json.dumps(dict(original=dict(original.identity),compressed=dict(compressed.identity)),indent=2)+'\n')
cache=ConvolCache(workspace)
rng=np.random.default_rng(507901)
rows=[]
for index in range(256):
    quartet=[keys[a] for a in rng.integers(len(keys),size=4)]
    if index<32:
        quartet[0]=keys[index%len(plan['occupied_keys'])]
        quartet[1]=quartet[0]
        quartet[3]=quartet[2]
    rank=index%8
    pairs=[waves[a] for a in quartet]
    native=float(compute_slater_rk_cell(*(v for pair in pairs for v in pair),workspace,rank,cache))
    transformed=float(compressed.integral(pairs,rank))
    rows.append(dict(keys=quartet,rank=rank,native_Ha=native,polynomial_Ha=transformed,error_Ha=transformed-native))
maximum=max(abs(r['error_Ha']) for r in rows)
result=dict(max_signed_error_Ha=maximum,rows=rows,seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,space_converged=False,experimental_gate=False)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
assert maximum<=contract['integral_gate_Ha']
