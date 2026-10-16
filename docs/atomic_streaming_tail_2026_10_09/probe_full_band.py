"""Construct complete native Yb band without a quartic resident ERI array."""
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'docs/yb_diagnosis_repair_2026_10_08'))
from common import load_reference
from atomx.methods.fscc import atomic_integrals, compute, export_integrals, resource_estimate


def main():
    start=time.monotonic()
    core,*_=load_reference()
    print(json.dumps(dict(stage='constructing complete VN band',load=os.getloadavg())),flush=True)
    a=atomic_integrals(core,virtual_lmax=3,virtual_energy_max_Ha=5,
        representation='on_demand',max_integral_bytes=2*1024**3,radial_pair_cache_size=16384)
    elapsed=time.monotonic()-start
    o,n=a.noccupied,len(a.labels)
    probes=[]
    # Fixed probes include occupied/first/highest virtual multiplets. These
    # samples validate access and symmetry, not the full continuum tail.
    for p,q,r,s in [(0,1,0,1),(o,o+1,o,o+1),(n-2,n-1,n-2,n-1),(0,n-1,0,n-1)]:
        g=a.eri(p,q,r,s)
        assert abs(g+a.eri(q,p,r,s))<1e-12
        assert abs(g-a.eri(r,s,p,q).conjugate())<1e-12
        probes.append(dict(indices=(p,q,r,s),value_Ha=(g.real,g.imag)))
    estimate=resource_estimate(a)
    rejects={}
    for name,operation in [('export',lambda:export_integrals(a,HERE/'forbidden_full_export')),
                           ('solve',lambda:compute(a,executable='/missing',directory=HERE/'forbidden_full_solve',memory_bytes=32*1024**3))]:
        try:operation()
        except MemoryError as error:rejects[name]=str(error)
        else:raise AssertionError('full backend/storage overbudget request was not rejected')
    assert not (HERE/'forbidden_full_export').exists() and not (HERE/'forbidden_full_solve').exists()
    result=dict(spinors=n,occupied=o,virtual=n-o,virtual_radial_keys=len(a.identity['virtual_keys']),
        complete_declared_band=True,identity=dict(a.identity),producer=a.diagnostics(),
        construction_seconds=elapsed,seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        resources=estimate,probes=probes,rejected_whole_backend_requests=rejects,
        scope='Complete native on-demand operator is constructed; full C1 tensor solve/export not performed; no truncation')
    (HERE/'full_yb_band.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='identity'}),flush=True)


if __name__=='__main__':main()
