"""Fixed-orbital factors: full static Breit, model QED, complete 4f-hole CI."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import pickle
import resource
import signal
import time
import traceback
import numpy as np
from run_yb import ROOT, CONFIGS, save, snapshot


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="v68_default")
    p.add_argument("--factor",choices=("corrections","4f_holes"),required=True)
    args=p.parse_args()
    base=ROOT/"results"/args.base
    out=ROOT/"results"/(args.base+"_"+args.factor)
    out.mkdir(exist_ok=False)
    if not json.loads((base/"status.json").read_text())["passed"]:
        raise ValueError("factor probe requires a passed baseline")
    contract={"hypothesis":"At identical converged baseline orbitals, Hamiltonian correction increments and added 4f-hole configuration mixing provide distinct measured shifts, not additive error estimates.",
        "base":args.base,"factor":args.factor,"fixed_conditions":"same Z70,69 electrons,24 orbitals,Fermi nucleus,radial metric and phase convention; no orbital reoptimization and no empirical offsets",
        "matching":"same J/parity and maximum squared overlap with embedded baseline state; minimum 0.5",
        "acceptance":{"full_CI_absolute_residual_Ha":1e-8,"base_replay_energy_Ha":1e-9,"even_4f_extension_drift_Ha":1e-9},
        "corrections":"full static Breit (Gaunt+retardation) in CI; local Uehling VP and diagonal Indelicato model SE as first-order state expectations; fixed orbitals; no orbital QED self-consistency or two-loop/screened many-body QED",
        "resource_estimate":"five-CSF corrections or46-CSF full declared 4f-hole space; max36 raw determinants per symmetry block, cap2000 unchanged; anticipated<3GiB RSS",
        "stopping_rule":"900 seconds; retain failure on exception, time cap, replay, eigenpair or matching gate; no selected-configuration truncation"}
    if args.factor=="4f_holes":
        contract["extension_contract"]=json.loads((ROOT/"extension_contract.json").read_text())
    save(out/"experiment_contract.json",contract);snapshot(out)
    t0=time.monotonic()
    def alarm(*_):raise TimeoutError("900-second fixed-orbital factor cap reached")
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.constants import HARTREE_CM1
        from atomx.methods.ci import ci_solve
        from atomx.physics.csf.matrix import build_ci_matrix
        from atomx.physics.csf.one_body import scalar_one_body_expectations
        from atomx.potentials.nuclear import make_nuclear_potential
        from atomx.corrections.qed import short_range_radial_matrix,self_energy_radial_matrix,uehling_potential
        from atomx.config.subshell import Subshell
        print("START",args.factor,flush=True)
        with (base/"result.pkl").open("rb") as f:r=pickle.load(f)
        r._nuclear_potential=make_nuclear_potential(70,"fermi")
        base_sha=hashlib.sha256((base/"orbitals_workspace.npz").read_bytes()).hexdigest()
        csfs=r.csfs
        if args.factor=="4f_holes":
            csfs=[c for cfg in contract["extension_contract"]["configurations"] for c in enumerate_jj_csfs(cfg)]
        kwargs=dict(csfs=csfs,orbitals_dict=r.orbitals_dict,orbital_basis=sorted(r.orbitals_dict),
            bsp_data=r.bsp_data,Z=70,ns=r.bsp_data["ns"],nuclear_potential=r._nuclear_potential)
        print("BUILD",len(csfs),"CSFs",flush=True)
        ci=ci_solve(**kwargs,include_breit=args.factor=="corrections")
        H=build_ci_matrix(**kwargs,include_breit=args.factor=="corrections")
        defect=np.linalg.norm(H@ci.mixing_coefficients-ci.mixing_coefficients*ci.state_energies,axis=0)
        base_replay=None
        if args.factor=="corrections":
            dc=ci_solve(**kwargs,include_breit=False)
            base_replay=float(max(abs(dc.state_energies-r.state_energies)))
            if base_replay>1e-9:raise ValueError("baseline DC replay failed")
        embedded=np.zeros((len(csfs),r.n_states));embedded[:len(r.csfs),:]=r.mixing_coefficients
        overlaps=abs(embedded.T@ci.mixing_coefficients)**2
        matches=np.argmax(overlaps,axis=1)
        weights=overlaps[np.arange(r.n_states),matches]
        if len(set(matches))!=r.n_states or min(weights)<.5:raise ValueError("ambiguous state matching")
        QED={}
        if args.factor=="corrections":
            keys=tuple(sorted(r.orbitals_dict))
            orbitals=[(*r.orbitals_dict[k],k[1]) for k in keys]
            vp=short_range_radial_matrix(orbitals,r.bsp_data,lambda radius:uehling_potential(radius,70))
            se=self_energy_radial_matrix([(*r.orbitals_dict[k],k[1],Subshell(k[0],k[1],1).label) for k in keys],70,r.bsp_data,method="indelicato")
            QED["VP"]=scalar_one_body_expectations(csfs,ci.mixing_coefficients,keys,vp)
            QED["model_SE"]=scalar_one_body_expectations(csfs,ci.mixing_coefficients,keys,se)
        qed_shift=sum(QED.values(),np.zeros(len(ci.state_energies)))
        levels=json.loads((base/"structure.json").read_text())["levels"]
        records=[]
        for old,idx in enumerate(matches):
            c=csfs[int(np.argmax(abs(ci.mixing_coefficients[:,idx])))]
            ref=levels[old]
            assert (c.total_J2,c.parity)==(ref["two_J"],ref["parity"])
            records.append({"label":ref["label"],"baseline_index":old,"new_index":int(idx),
                "reference_overlap_squared":float(weights[old]),"baseline_energy_Ha":float(r.state_energies[old]),
                "new_energy_Ha":float(ci.state_energies[idx]),"QED_components_Ha":{k:float(v[idx]) for k,v in QED.items()},
                "energy_shift_Ha":float(ci.state_energies[idx]-r.state_energies[old]),
                "excitation_cm1":float((ci.state_energies[idx]-ci.state_energies[matches[0]])*HARTREE_CM1),
                "excitation_shift_cm1":float((ci.state_energies[idx]-r.state_energies[old]-(ci.state_energies[matches[0]]-r.state_energies[0]))*HARTREE_CM1),
                "QED_excitation_shift_cm1":float((qed_shift[idx]-qed_shift[matches[0]])*HARTREE_CM1)})
        even=max(abs(x["energy_shift_Ha"]) for x in records if x["label"].startswith(("6s","5d")))
        passed=float(max(defect))<1e-8 and (args.factor!="4f_holes" or even<1e-9)
        save(out/"factors.json",{"base_orbitals_npz_sha256":base_sha,"n_CSFS":len(csfs),"records":records,
            "all_energies_Ha":ci.state_energies,"mixing":ci.mixing_coefficients,
            "CI_residuals_Ha":defect,"max_CI_residual_Ha":float(max(defect)),"base_replay_error_Ha":base_replay,
            "even_extension_max_drift_Ha":even if args.factor=="4f_holes" else None,
            "elapsed_seconds_uncontrolled":time.monotonic()-t0,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        save(out/"status.json",{"passed":passed,"scope":"fixed-orbital factor probe; not orbital stationarity in extended/corrected model"})
        print("DONE",args.factor,"passed",passed,"max_residual",max(defect),flush=True)
        print(json.dumps(records),flush=True)
    except Exception as exc:
        save(out/"failure.json",{"exception":repr(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-t0})
        print(traceback.format_exc(),flush=True);raise
    finally:signal.alarm(0)


if __name__=="__main__":main()
