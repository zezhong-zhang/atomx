"""Exercise the actual CI one-body-density injection on all five Yb states."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import numpy as np
from common import ROOT, KEYS, load_reference, save, sha


def main():
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci_mbpt import ci_mbpt_solve
    from atomx.physics.csf.matrix import build_ci_matrix
    p = argparse.ArgumentParser();p.add_argument("--run",required=True);args=p.parse_args()
    parent = ROOT/"results"/args.run
    out = parent/"ci_injection"
    out.mkdir(exist_ok=False)
    save(out/"contract.json",dict(script_sha256=sha(__file__),max_residual_Ha=1e-8,
        matched_energy_error_Ha=1e-8, minimum_squared_overlap=.99,
        scope="Five-dimensional frozen-core CI with computed diagonal one-body second-order self-energy. No two-valence screening or dressed radiative operator."))
    records = json.loads((parent/"results.json").read_text())["records"]
    core,r,levels,identity = load_reference()
    sigma1 = {KEYS[x["label"]]:x["Sigma2_Ha"] for x in records}
    kwargs = dict(csfs=r.csfs,orbitals_dict=r.orbitals_dict,orbital_basis=sorted(r.orbitals_dict),
                  bsp_data=r.bsp_data,Z=70,ns=r.bsp_data["ns"],nuclear_potential=r._nuclear_potential)
    ci = ci_mbpt_solve(**kwargs,sigma1=sigma1)
    H = build_ci_matrix(**kwargs,one_body_shifts=sigma1)
    residuals = np.linalg.norm(H@ci.mixing_coefficients-ci.mixing_coefficients*ci.state_energies,axis=0)
    overlaps = abs(r.mixing_coefficients.T@ci.mixing_coefficients)**2
    matches = np.argmax(overlaps,axis=1)
    weights = overlaps[np.arange(r.n_states),matches]
    expected = [lv["energy_hartree"]+sigma1[KEYS[lv["label"]]] for lv in levels]
    errors = [ci.state_energies[idx]-expected[i] for i,idx in enumerate(matches)]
    passed = max(residuals)<1e-8 and max(abs(np.array(errors)))<1e-8 and min(weights)>=.99 and len(set(matches))==5
    save(out/"results.json",dict(passed=passed,identity=identity,residuals_Ha=residuals,
        matching_squared_overlap=weights,matched_energy_error_Ha=errors,
        corrected_energies_Ha=ci.state_energies,baseline_to_new_indices=matches,
        mbpt_provenance={"sigma1_Ha":{str(k):v for k,v in sigma1.items()},"two_body_screening":"not_implemented"}))
    print("CI Sigma1 injection",args.run,passed,"max_error_Ha",max(abs(np.array(errors))),flush=True)


if __name__=="__main__":main()
