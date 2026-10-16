"""Full retained energy-band virtuals, no count slicing."""
import json,time,sys
import numpy as np
from atomx.methods.dirac_fock import DFSolver,Subshell
from atomx.methods.fscc import atomic_integrals,compute
from probe_native import HERE,EXE,ENV
start=time.monotonic()
core=DFSolver(Z=10,subshells=[Subshell(1,-1,2),Subshell(2,-1,2),Subshell(2,1,2),Subshell(2,-2,4)],ns=40,ks=7).solve()
a=atomic_integrals(core,virtual_lmax=2,virtual_energy_max_Ha=1,max_integral_bytes=2*1024**3)
print('full Ne band',len(a.labels),a.noccupied,len(a.identity['virtual_keys']),time.monotonic()-start,flush=True)
r=compute(a,executable=EXE,directory=HERE/'ne_full_band',memory_bytes=32*1024**3,tile_size=16,threads=2,tolerance=1e-13,timeout_seconds=1800,environment=ENV)
(HERE/'ne_full_band_summary.json').write_text(json.dumps(dict(core_energy_Ha=r.core_energy_Ha,update=r.amplitude_update_max,residual_bound=r.equation_residual_bound_Ha,spinors=len(a.labels),occupied=a.noccupied,virtual=len(a.labels)-a.noccupied,radial_virtual_count=len(a.identity['virtual_keys']),band_Ha=[0,1],lmax=2,seconds=time.monotonic()-start,identity=dict(a.identity)),indent=2)+'\n')
print('Ne result',r.core_energy_Ha,time.monotonic()-start,flush=True)
