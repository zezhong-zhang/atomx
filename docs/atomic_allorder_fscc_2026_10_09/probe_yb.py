"""Declared finite radial pools; all core holes handled by tensor amplitudes."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'yb_diagnosis_repair_2026_10_08'))
from common import load_reference
from probe_native import HERE,EXE,ENV
from atomx.methods.mbpt import build_virtual_basis
from atomx.physics.csf.spinor_integrals import build_spinor_integrals,packed_size
from atomx.methods.fscc import compute,resource_estimate
ap=argparse.ArgumentParser();ap.add_argument('--correlated',type=int,choices=[14,22,68],required=True);ap.add_argument('--radials',type=int,choices=[1,2],required=True);ap.add_argument('--model',choices=['CCSD','CCSDT'],default='CCSD');args=ap.parse_args()
start=time.monotonic();core,*_=load_reference()
od={k:v[:2] for k,v in core.orbitals_by_subshell().items()}
frozen=tuple(k for k in od if args.correlated==14 and k[1] not in (3,-4) or args.correlated==22 and k[0:2] not in ((4,3),(4,-4),(5,-1),(5,1),(5,-2))) if args.correlated!=68 else ()
occupied=tuple(k for k in od if k not in frozen)
assert sum(2*abs(k) for _,k in occupied)==args.correlated
vir=[];eps={}
for k in (-1,1,-2,2,-3):
 _,states=build_virtual_basis(core,k,e_max=5,potential_mode='VN',orthogonal_to_occupied=True)
 # This run declares 1 or 2 radial states per kappa in advance, not a capped band run.
 n=max(n for n,k0 in od if k0==k)+1
 for j,(P,Q,e) in enumerate(states[:args.radials]):
  key=n+j,k;od[key]=P,Q;vir.append(key);eps[str(key)]=e
n=args.correlated+sum(2*abs(k) for _,k in vir)
pre=dict(correlated=args.correlated,frozen_keys=frozen,virtual_keys=vir,virtual_radial_definition=f'lowest {args.radials} VN complement eigenstates in each s/p/d kappa',virtual_energies_Ha=eps,spinors=n,packed_integral_bytes=16*packed_size(n),memory_cap_GiB=32,timeout_seconds=1800)
name=f'yb_o{args.correlated}_r{args.radials}'
if args.model!='CCSD': name+='_'+args.model.lower()
(HERE/(name+'_preflight.json')).write_text(json.dumps(pre,indent=2)+'\n')
print(json.dumps(pre),flush=True)
a=build_spinor_integrals(od,core._bsp_data,70,123,occupied_keys=occupied,virtual_keys=vir,frozen_keys=frozen,nuclear_potential=core._nuclear_potential,max_integral_bytes=2*1024**3)
active=tuple(i for i,(n,k,m) in enumerate(a.labels) if (n,k) in vir[::args.radials])
print('integrals ready',time.monotonic()-start,resource_estimate(a,model=args.model,sector='0h1p',active_count=len(active)),flush=True)
r=compute(a,executable=EXE,directory=HERE/name,model=args.model,sector='0h1p',active_indices=active,tolerance=1e-13,residual_bound_tolerance_Ha=1e-8,memory_bytes=32*1024**3,tile_size=16,threads=2,timeout_seconds=1800,environment=ENV)
energies=np.array(r.sector_energies_Ha); transitions=(energies-energies[0])*219474.6313632
summary=dict(core_energy_Ha=r.core_energy_Ha,sector_energies_Ha=r.sector_energies_Ha,transitions_sorted_cm=transitions.tolist(),amplitude_update_max=r.amplitude_update_max,equation_residual_bound_Ha=r.equation_residual_bound_Ha,seconds=time.monotonic()-start,gate='numerical only; no final J/experimental matching here',preflight=pre)
(HERE/(name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
