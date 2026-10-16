"""Replay declared pole subsets and inject the solved diagonal shifts into native CI."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name]="1"
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"yb_diagnosis_repair_2026_10_08"
sys.path.insert(0,str(PRIOR))
from common import KEYS, EXPERIMENT, load_reference, save, sha, source_identity


def main():
    from atomx.constants import HARTREE_CM1
    from atomx.methods.mbpt_transitions import DiagonalSelfEnergy
    from atomx.methods.ci_mbpt import ci_mbpt_solve
    from atomx.physics.csf.matrix import build_ci_matrix
    parent=ROOT/"results/l6_e50_all"
    rows=json.loads((parent/"results.json").read_text())["records"]
    out=ROOT/"replay_and_ci"
    out.mkdir(exist_ok=False)
    save(out/"contract.json",dict(script_sha256=sha(__file__),
        primary_contract_sha256=sha(ROOT/"contract.json"),
        subset_policy="Exact virtual-kappa and maximum virtual energy filtering; no amplitude threshold",
        cutoffs=[dict(lmax=l,emax_Ha=e) for l in (4,5,6) for e in (5.,20.,50.)],
        bracket_policy="For these additional subsets, choose [eps-.1,eps] if Sigma(eps)<0, otherwise [eps,eps+.1]; reject intervening weighted poles or missing roots",
        root_residual_Ha=1e-10, cap_seconds=120, matched_CI_energy_Ha=1e-8,
        CI_residual_Ha=1e-8, minimum_CI_squared_overlap=.99))
    start=time.monotonic()
    core,r,levels,identity=load_reference()
    subsets=[]
    for lmax in (4,5,6):
        for emax in (5.,20.,50.):
            rr=[]
            for row in rows:
                data=np.load(parent/f"{row['label'].replace('/','_')}_poles.npz")
                keep=(data["virtual_lmax"]<=lmax)&(data["virtual_emax_Ha"]<emax)
                eps=float(data["epsilon_DF_Ha"])
                sub=DiagonalSelfEnergy(eps,data["poles_Ha"][keep],data["weights_Ha2"][keep],
                    str(data["reference_identity"])+f":l={lmax}:emax={emax}")
                onshell=sub.evaluate(eps)
                bracket=(eps-.1,eps) if onshell<0 else (eps,eps+.1)
                root=sub.solve_dyson(bracket)
                rr.append(dict(label=row["label"],epsilon_DF_Ha=eps,on_shell_Ha=onshell,
                    dyson_energy_Ha=root.energy_Ha,dyson_shift_Ha=root.shift_Ha,
                    residue=root.residue,residual_Ha=root.residual_Ha))
            for row in rr:
                row["excitation_Dyson_cm1"]=(row["dyson_energy_Ha"]-rr[0]["dyson_energy_Ha"])*HARTREE_CM1
                row["residual_experiment_cm1"]=row["excitation_Dyson_cm1"]-EXPERIMENT[row["label"]]
            subsets.append(dict(lmax=lmax,emax_Ha=emax,records=rr))
            print("REPLAY",lmax,emax,flush=True)
    save(out/"subsets.json",subsets)
    sigma1={KEYS[row["label"]]:row["variants"]["6"]["dyson"]["shift_Ha"] for row in rows}
    kwargs=dict(csfs=r.csfs,orbitals_dict=r.orbitals_dict,orbital_basis=sorted(r.orbitals_dict),
        bsp_data=r.bsp_data,Z=70,ns=r.bsp_data["ns"],nuclear_potential=r._nuclear_potential)
    ci=ci_mbpt_solve(**kwargs,sigma1=sigma1)
    H=build_ci_matrix(**kwargs,one_body_shifts=sigma1)
    residuals=np.linalg.norm(H@ci.mixing_coefficients-ci.mixing_coefficients*ci.state_energies,axis=0)
    overlaps=abs(r.mixing_coefficients.T@ci.mixing_coefficients)**2
    matches=np.argmax(overlaps,axis=1)
    weights=overlaps[np.arange(r.n_states),matches]
    expected=[lv["energy_hartree"]+sigma1[KEYS[lv["label"]]] for lv in levels]
    errors=[ci.state_energies[j]-expected[i] for i,j in enumerate(matches)]
    passed=max(residuals)<1e-8 and max(abs(np.array(errors)))<1e-8 and min(weights)>=.99 and len(set(matches))==5
    save(out/"ci_injection.json",dict(passed=passed,identity=identity, residuals_Ha=residuals,
        matching_squared_overlap=weights,matched_energy_error_Ha=errors,
        corrected_energies_Ha=ci.state_energies,baseline_to_new_indices=matches,
        one_body_Dyson_shifts_Ha={str(k):v for k,v in sigma1.items()},
        scope="One diagonal value per frozen valence state; no off-diagonal energy-dependent CI matrix or transition dressing"))
    if not passed:raise ValueError("Native CI injection gate failed")
    elapsed=time.monotonic()-start
    if elapsed>120:raise TimeoutError("120-second declared replay cap exceeded")
    save(out/"status.json",dict(completed=True,elapsed_seconds=elapsed,
        CI_injection_passed=passed,source_sha256=source_identity()))
    print("CI PASS",passed,"max_Ha",max(abs(np.array(errors))),"seconds",elapsed,flush=True)


if __name__=="__main__":main()
