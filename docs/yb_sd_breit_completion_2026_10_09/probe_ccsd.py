"""Native two-electron complete-space CCSD/CI numerical capture."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from atomx.config.csf import enumerate_jj_orbital_space, get_shared_orbital_basis
from atomx.methods.coupled_cluster import compute
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.physics.csf.sparse_ci import assemble_sparse_hamiltonian, enumerate_determinants, patterns_from_csfs
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
start=time.monotonic()
core=DFSolver(Z=4,subshells=[Subshell(1,-1,2)],ns=30,ks=7).solve()
od={(1,-1):core.orbitals_by_subshell()[1,-1][:2]}
for k in (-1,1,-2):
    _,virtual=build_virtual_basis(core,k,e_max=50,potential_mode='VN',orthogonal_to_occupied=True)
    od[2,k]=virtual[0][:2]
csfs=enumerate_jj_orbital_space(od,2,magnetic_state_cap=256)
dets,spinors=enumerate_determinants(patterns_from_csfs(csfs),od,m2_total=None)
raw=assemble_sparse_hamiltonian(dets,spinors,od,core._bsp_data,4,30).toarray()
asym=float(np.max(abs(raw-raw.T)))
if asym>1e-12: raise RuntimeError('producer Hermitian roundoff gate failed')
h=.5*(raw+raw.T)
ref=sum(1<<i for i,(n,k,m) in enumerate(spinors) if (n,k)==(1,-1))
cc=compute(h,dets,ref,spinor_count=len(spinors),reference_identity='Be2+:point-DC:ns30:complete-1s2s2p:N2')
sector=[c for c in csfs if (c.total_J2,c.parity)==(0,1)]
jj=build_ci_matrix(sector,od,get_shared_orbital_basis(sector),core._bsp_data,4,30,nuclear_potential=core._nuclear_potential)
anchor=float(np.linalg.eigvalsh(jj)[0]); en,vec=np.linalg.eigh(h)
err=abs(cc.total_energy_Ha-anchor)
proj=float(np.max(abs(np.outer(cc.coefficients,cc.coefficients)-np.outer(vec[:,0],vec[:,0]))))
if err>1e-10 or cc.full_eigenstate_residual_Ha>1e-10 or proj>1e-8:
    raise RuntimeError('native CCSD/CI gate failed')
np.savez_compressed(HERE/'native_ccsd_anchor.npz',hamiltonian=h,raw_hamiltonian=raw,
    determinant_masks=dets,reference_mask=ref,spinors=spinors,CC_amplitudes=cc.amplitudes,
    CC_vector=cc.coefficients,CI_ground_vector=vec[:,0],CSF_hamiltonian=jj)
r=dict(theory='finite nonlinear CCSD; exact nilpotent T1+T2',nuclear='point',Z=4,ns=30,
    CC_energy_Ha=cc.total_energy_Ha,CSF_CI_energy_Ha=anchor,energy_error_Ha=err,
    projected_SD_residual_Ha=cc.amplitude_residual_Ha,full_residual_Ha=cc.full_eigenstate_residual_Ha,
    projector_error=proj,producer_antisymmetry_Ha=asym,
    explicit_producer_projection_max_Ha=float(np.max(abs(raw-h))),provenance=dict(cc.provenance),
    seconds=time.monotonic()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'src/atomx').rglob('*.py'))},passed=True)
(HERE/'native_ccsd_anchor.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ('sources','provenance')},indent=2))
