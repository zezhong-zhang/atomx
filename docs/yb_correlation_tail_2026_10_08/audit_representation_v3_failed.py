"""Audit full 200-Ha virtual metrics and preserve prior sealed evidence."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name]="1"
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import load_reference,save,sha,source_identity


def main():
    from atomx.numerics.radial_workspace import workspace_arrays,eval_orbital_pair
    out=ROOT/"representation_audit_v3"
    out.mkdir(exist_ok=False)
    save(out/"contract.json",dict(script_sha256=sha(__file__),cap_seconds=120,
        inherited_primary_contract_sha256=sha(ROOT/"contract.json"),
        metric_gate=1e-7,source_sha256=source_identity(),
        scope="Full native quadrature Gram matrices and occupied overlaps for both complete 200-Ha preflight snapshots; orbital counts are capacity controls, not energy error bounds."))
    start=time.monotonic()
    blocks=[]
    for case,base,name in (("coarse_l8_e200","v68_default","v68_default_virtuals.npz"),
        ("fine_l6_e200","v68_refined_newton","v68_refined_200Ha_virtuals.npz")):
        core,r,levels,identity=load_reference(base)
        _,w=workspace_arrays(core._bsp_data)
        w=w.ravel()
        with np.load(ROOT/"preflight"/name) as data:
            for field in data.files:
                if not field.startswith("energy_"):continue
                k=int(field.removeprefix("energy_"));energies=data[field]
                ps,qs=[],[]
                for P,Q in zip(data[f"P_{k}"],data[f"Q_{k}"],strict=True):
                    p,q=eval_orbital_pair(P,Q,core._bsp_data);ps.append(p.ravel());qs.append(q.ravel())
                p,q=np.asarray(ps),np.asarray(qs)
                gram=(p*w)@p.T+(q*w)@q.T
                gram_error=float(np.max(abs(gram-np.eye(len(energies)))))
                core_overlap=0.
                for P,Q,occ in core.orbitals_raw.get(k,[]):
                    if occ<=0.01:continue
                    pc,qc=eval_orbital_pair(P,Q,core._bsp_data)
                    pc,qc=pc.ravel(),qc.ravel()
                    core_overlap=max(core_overlap,float(np.max(abs(p@(w*pc)+q@(w*qc)))))
                if gram_error>1e-7 or core_overlap>1e-7:
                    raise ValueError(f"Full virtual metric gate failed: {case} {k}")
                blocks.append(dict(case=case,base=base,kappa=k,
                    l=k if k>0 else -k-1,actual_virtuals=len(energies),
                    counts={str(e):int(np.count_nonzero(energies<e)) for e in (5.,20.,50.,100.,200.)},
                    virtual_Gram_max_error=gram_error,max_occupied_overlap=core_overlap))
    preserved={}
    for folder in ("yb_diagnosis_repair_2026_10_08","yb_energy_dependent_sigma_2026_10_08"):
        manifest=ROOT.parent/folder/"artifact_manifest.json"
        entries=json.loads(manifest.read_text())
        manifest_root=manifest.parent if folder=="yb_diagnosis_repair_2026_10_08" else REPO
        bad=[name for name,meta in entries.items() if sha(manifest_root/name)!=meta["sha256"]]
        if bad:raise ValueError(f"Sealed evidence changed: {bad}")
        preserved[folder]=dict(manifest_sha256=sha(manifest),files=len(entries),all_hashes_preserved=True)
    current=source_identity()
    if current!=json.loads((ROOT/"capture_source_sha256.json").read_text()):
        raise ValueError("Capture source changed")
    elapsed=time.monotonic()-start
    if elapsed>120:raise TimeoutError("Declared representation audit cap exceeded")
    save(out/"results.json",dict(blocks=blocks,preserved_previous_manifests=preserved,
        notes=["Initial refined preflight constructed only the <50-Ha spectrum, so its 200-Ha count is a count within that already filtered snapshot. This audit uses the separately rebuilt complete refined 200-Ha snapshot and supersedes that count for current 200-Ha comparisons.",
        "The initial candidate-workload estimate had a 1000-term arithmetic typo; workload_count_check.json records the correction without changing the running physical contract."],
        numerical_passed=True,source_unchanged=True,elapsed_seconds=elapsed))
    print("VIRTUAL METRIC PASS",max(x["virtual_Gram_max_error"] for x in blocks),
        "CORE OVERLAP",max(x["max_occupied_overlap"] for x in blocks),
        "PRESERVED",{k:v["files"] for k,v in preserved.items()},"SECONDS",elapsed)


if __name__=="__main__":main()
