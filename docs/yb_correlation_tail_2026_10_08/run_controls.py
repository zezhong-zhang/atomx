"""Complete five-state native sums in two radial spaces; bounded process pool."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
from array import array
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import resource
import signal
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"yb_diagnosis_repair_2026_10_08"
DYSON=ROOT.parent/"yb_energy_dependent_sigma_2026_10_08"
sys.path.insert(0,str(PRIOR))
from common import KEYS,load_reference,save,sha,source_identity


def worker(job):
    from atomx.constants import HARTREE_CM1
    from atomx.methods.mbpt_transitions import DiagonalSelfEnergy,build_mbpt2_self_energy
    case,base,lmax,label,cap=job
    out=ROOT/"results"/case/label.replace("/","_")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def alarm(*_):raise TimeoutError(f"{cap}-second declared state cap reached")
    signal.signal(signal.SIGALRM,alarm);signal.alarm(cap)
    try:
        save(out/"contract.json",dict(parent_contract_sha256=sha(ROOT/"contract.json"),
            script_sha256=sha(__file__),cap_seconds=cap,load=os.getloadavg(),
            case=case,base=base,lmax=lmax,emax_Ha=200.,label=label))
        core,r,levels,identity=load_reference(base)
        level=next(x for x in levels if x["label"]==label)
        key=KEYS[label];P,Q=r.orbitals_dict[key]
        kappas=[-1]+[k for l in range(1,lmax+1) for k in (l,-l-1)]
        snapshot_identity=dict(reference=identity,label=label,key=key,
            epsilon_DF_Ha=level["valence_energy_hartree"],virtual_kappas=kappas,
            upper_ceiling_Ha=200.,root_cap=None,Hamiltonian="Dirac-Coulomb",
            metric="native radial quadrature",scope="frozen-orbital second-order attachment")
        resolved_id=hashlib.sha256(json.dumps(snapshot_identity,sort_keys=True).encode()).hexdigest()
        save(out/"identity.json",snapshot_identity)
        gl,ge,dg=array("B"),array("d"),array("B")
        counts={"2p1h":0,"2h1p":0};virtuals={}
        def record(term):
            counts[term["diagram"]]+=1
            virtuals.update(zip(term["virtual_indices"],term["virtual"],strict=True))
            if term["numerator"]!=0:
                gl.append(max(k if k>0 else -k-1 for k,e in term["virtual"]))
                ge.append(max(e for k,e in term["virtual"]))
                dg.append(0 if term["diagram"]=="2p1h" else 1)
        print("START",case,label,"pid",os.getpid(),flush=True)
        model=build_mbpt2_self_energy(core,label,reference_identity=resolved_id,
            valence_orbital=(P,Q,key[1],level["valence_energy_hartree"]),
            virtual_kappas=kappas,n_virt_max=None,energy_window_au=200.,term_recorder=record)
        gl,ge,dg=np.asarray(gl),np.asarray(ge),np.asarray(dg)
        np.savez_compressed(out/"poles.npz",poles_Ha=model.poles_Ha,weights_Ha2=model.weights_Ha2,
            virtual_lmax=gl,virtual_emax_Ha=ge,diagram=dg,reference_identity=resolved_id,
            epsilon_DF_Ha=model.reference_energy_Ha)
        variants=[]
        for lcut in ((4,6,7,8) if lmax==8 else (4,6)):
            for ecut in (5.,20.,50.,100.,200.):
                keep=(gl<=lcut)&(ge<ecut)
                sub=DiagonalSelfEnergy(model.reference_energy_Ha,model.poles_Ha[keep],
                    model.weights_Ha2[keep],resolved_id+f":l={lcut}:e={ecut}")
                eps=model.reference_energy_Ha
                onshell=sub.evaluate(eps)
                bracket=(eps-.1,eps) if onshell<0 else (eps,eps+.1)
                root=sub.solve_dyson(bracket)
                replay_error=None
                if base=="v68_default" and lcut in (4,6) and ecut==50.:
                    old=json.loads((PRIOR/f"results/l{lcut}_e50_all/results.json").read_text())
                    target=next(x for x in old["records"] if x["label"]==label)
                    replay_error=onshell-target["Sigma2_Ha"]
                    if abs(replay_error)>1e-10:raise ValueError("Retained native on-shell gate failed")
                variants.append(dict(lmax=lcut,emax_Ha=ecut,on_shell_Ha=onshell,
                    dyson=asdict(root),matched_old_on_shell_error_Ha=replay_error,
                    retained_nonzero_terms=int(np.count_nonzero(keep))))
                del sub,keep
        if source_identity()!=identity["source_sha256"]:raise ValueError("Source changed during capture")
        row=dict(case=case,base=base,label=label,key=key,
            epsilon_DF_Ha=model.reference_energy_Ha,variants=variants,
            virtual_manifest=virtuals,candidate_counts=counts,
            elapsed_seconds_uncontrolled=time.monotonic()-started,
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            source_unchanged=True)
        save(out/"results.json",row);save(out/"status.json",dict(completed=True,numerical_gates_passed=True))
        print("DONE",case,label,"seconds",row["elapsed_seconds_uncontrolled"],
            "shift_cm1",variants[-1]["dyson"]["shift_Ha"]*HARTREE_CM1,flush=True)
        return dict(case=case,label=label,completed=True,seconds=row["elapsed_seconds_uncontrolled"])
    except Exception as exc:
        failure=dict(exception=repr(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started)
        save(out/"failure.json",failure)
        print("FAILED",case,label,failure["exception"],flush=True)
        return dict(case=case,label=label,completed=False,failure=failure)
    finally:
        signal.alarm(0)
        gc.collect()


def main():
    root=ROOT/"results"
    root.mkdir(exist_ok=False)
    identity=source_identity()
    save(ROOT/"capture_source_sha256.json",identity)
    contract=json.loads((ROOT/"contract.json").read_text())
    save(ROOT/"launch.json",dict(parent_contract_sha256=sha(ROOT/"contract.json"),
        script_sha256=sha(__file__),load=os.getloadavg(),workers=6,
        expected_jobs=10,source_sha256=identity))
    labels=["6s_1/2","5d_3/2","6p_1/2","5d_5/2","6p_3/2"]
    jobs=[]
    # Start all coarse jobs first; the sixth worker starts the fine reference.
    for case in contract["captures"]:
        for label in labels:
            jobs.append((case["name"],case["reference"],case["lmax"],label,case["cap_seconds_per_state"]))
    status=[]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            status.append(future.result())
            save(ROOT/"progress.json",dict(jobs=status,elapsed_seconds=time.monotonic()-started))
    save(ROOT/"capture_status.json",dict(completed=all(x["completed"] for x in status),
        jobs=status,elapsed_seconds_uncontrolled=time.monotonic()-started,
        source_unchanged=source_identity()==identity))
    print("ALL CAPTURES",all(x["completed"] for x in status),time.monotonic()-started,flush=True)


if __name__=="__main__":main()
