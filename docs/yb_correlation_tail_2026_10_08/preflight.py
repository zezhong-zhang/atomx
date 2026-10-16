"""Enumerate the actual native virtual spaces before the expensive tail controls."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
from pathlib import Path
import json
import sys
import time
import signal
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import load_reference,save,sha


def main():
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.numerics.bspline.convol import ConvolCache
    from atomx.numerics.radial_workspace import workspace_arrays,eval_orbital_pair
    out=ROOT/"preflight"
    out.mkdir(exist_ok=False)
    t0=time.monotonic()
    def alarm(*_):raise TimeoutError("180-second native preflight cap")
    signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    entries=[]
    try:
        for base,lmax,emax in (("v68_default",8,200.),("v68_refined_newton",6,50.)):
            core,result,levels,identity=load_reference(base)
            cache=ConvolCache(core._bsp_data)
            _,w=workspace_arrays(core._bsp_data)
            blocks={}
            arrays={}
            for kap in [-1]+[k for l in range(1,lmax+1) for k in (l,-l-1)]:
                _,states=build_virtual_basis(core,kap,cache=cache,e_max=emax,
                    potential_mode="VN",orthogonal_to_occupied=True)
                blocks[kap]=states
                counts={str(c):sum(e<c for P,Q,e in states) for c in (5.,20.,50.,200.)}
                metric=0.
                core_overlap=0.
                # Native quadrature checks normalization and occupied-space projection.
                for P,Q,e in states:
                    p,q=eval_orbital_pair(P,Q,core._bsp_data)
                    metric=max(metric,abs(float(np.sum(w*(p*p+q*q)))-1.))
                    for Pc,Qc,occ in core.orbitals_raw.get(kap,[]):
                        if occ<=0.01:continue
                        pc,qc=eval_orbital_pair(Pc,Qc,core._bsp_data)
                        core_overlap=max(core_overlap,abs(float(np.sum(w*(p*pc+q*qc)))))
                if metric>1e-7 or core_overlap>1e-7:
                    raise ValueError(f"Virtual metric preflight failed {base} {kap}")
                arrays[f"P_{kap}"]=np.asarray([s[0] for s in states])
                arrays[f"Q_{kap}"]=np.asarray([s[1] for s in states])
                arrays[f"energy_{kap}"]=np.asarray([s[2] for s in states])
                entries.append(dict(base=base,kappa=kap,counts=counts,
                    normalization_error=metric,core_overlap=core_overlap))
                print("BLOCK",base,kap,counts,"metric",metric,flush=True)
            np.savez_compressed(out/f"{base}_virtuals.npz",**arrays)
            save(out/f"{base}_identity.json",identity)
        save(out/"counts.json",dict(blocks=entries,elapsed_seconds=time.monotonic()-t0,
            machine_load=os.getloadavg(),script_sha256=sha(__file__),
            contract_sha256=sha(ROOT/"preflight_contract.json")))
        print("PREFLIGHT DONE",time.monotonic()-t0,flush=True)
    finally:signal.alarm(0)


if __name__=="__main__":main()
