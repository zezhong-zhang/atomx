"""Match every native capture's virtual identity and full diagram candidate counts."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import save,sha


def main():
    out=ROOT/"capture_coverage"
    out.mkdir(exist_ok=False)
    save(out/"contract.json",dict(script_sha256=sha(__file__),
        scope="All ten state captures must use identical full virtual manifests within each radial model and match independent preflight energies; both diagram loop counts must be complete.",
        energy_match_Ha=1e-12,integer_count_tolerance=0,
        captures_contract_sha256=sha(ROOT/"contract.json")))
    rows=[]
    for case,name,lmax in (("coarse_l8_e200","v68_default_virtuals.npz",8),
        ("fine_l6_e200","v68_refined_200Ha_virtuals.npz",6)):
        target=[]
        with np.load(ROOT/"preflight"/name) as data:
            for k in [-1]+[k for l in range(1,lmax+1) for k in (l,-l-1)]:
                target.extend((k,float(e)) for e in data[f"energy_{k}"])
        nv=len(target);nc=19
        expected={"2p1h":nc*nv*(nv+1)//2,"2h1p":nv*nc*(nc+1)//2}
        for folder in sorted((ROOT/"results"/case).iterdir()):
            r=json.loads((folder/"results.json").read_text())
            actual=[r["virtual_manifest"][str(i)] for i in range(nv)]
            if len(r["virtual_manifest"])!=nv or any(a[0]!=b[0] for a,b in zip(actual,target,strict=True)):
                raise ValueError("Virtual identity/count mismatch")
            err=max(abs(a[1]-b[1]) for a,b in zip(actual,target,strict=True))
            if err>1e-12 or r["candidate_counts"]!=expected:
                raise ValueError("Virtual energy or candidate coverage gate failed")
            rows.append(dict(case=case,label=r["label"],virtuals=nv,
                max_virtual_energy_match_error_Ha=err,candidate_counts=r["candidate_counts"],passed=True))
    if len(rows)!=10:raise ValueError("Incomplete ten-state coverage")
    save(out/"results.json",dict(passed=True,records=rows))
    print("FULL CAPTURE COVERAGE PASS",len(rows),"states; max energy mismatch",max(r["max_virtual_energy_match_error_Ha"] for r in rows))


if __name__=="__main__":main()
