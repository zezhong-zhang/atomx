"""Current-source replay and independent published DF controls."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from dataclasses import asdict
import json
import signal
from types import SimpleNamespace
import numpy as np
from common import ROOT, OLD, KEYS, load_reference, save, sha


def main():
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci import ci_solve
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.properties.transitions.calculator import TransitionCalculator
    from atomx.properties.transitions.angular import angular_eL
    from atomx.properties.transitions.multipole import radial_eL
    out = ROOT/"baseline_audit_v3"
    out.mkdir(exist_ok=False)
    save(out/"contract.json", dict(script_sha256=sha(__file__),
        Hamiltonian_replay_Ha=1e-8, CSF_vs_single_orbital_RME_magnitude_au=1e-10,
        published_DF_excitation_cm1=1., published_DF_binding_cm1=1.,
        published_DF_static_dipole_au=5e-5, published_Breit_difference_cm1=1.,
        limitations="Publication rounding and different nuclear/radial definitions; printed agreement does not certify sub-cm precision. RME sign retained, only magnitudes compared across external phase conventions.",
        sources=["https://arxiv.org/html/1205.5895v2",
                 "https://mariannasafronova.com/wp-content/uploads/2022/12/YbII-2009.pdf"]))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second audit cap")))
    signal.alarm(120)
    rows = []
    published = {"6s_1/2": (-90789,0), "5d_3/2": (-66517,24272),
                 "5d_5/2": (-66037,24752), "6p_1/2": (-66087,24702), "6p_3/2": (-63276,27513)}
    dipoles = {("6s_1/2","6p_1/2"):3.2422, ("6s_1/2","6p_3/2"):4.5426,
               ("5d_3/2","6p_1/2"):3.8610, ("5d_3/2","6p_3/2"):1.6969,
               ("5d_5/2","6p_3/2"):5.2001}
    for base in ("v68_default", "v68_box90_newton"):
        core, r, levels, identity = load_reference(base)
        kwargs = dict(csfs=r.csfs, orbitals_dict=r.orbitals_dict, orbital_basis=sorted(r.orbitals_dict),
                      bsp_data=r.bsp_data, Z=70, ns=r.bsp_data["ns"], nuclear_potential=r._nuclear_potential)
        ci = ci_solve(**kwargs)
        H = build_ci_matrix(**kwargs)
        res = np.linalg.norm(H@ci.mixing_coefficients-ci.mixing_coefficients*ci.state_energies, axis=0)
        drift = max(abs(ci.state_energies-r.state_energies))
        holder = SimpleNamespace(converged=True, Z=70, _csf_result=r)
        lines = TransitionCalculator(holder).compute_all(multipoles=["E1","E2","M1"], gauge="both", line_resolution="ci")
        save(out/(base+"_lines.json"), [asdict(line) for line in lines])
        energies = []
        for lv in levels:
            bind, excite = published[lv["label"]]
            energies.append(dict(label=lv["label"], excitation_cm1=lv["excitation_cm1"],
                paper_DF_excitation_cm1=excite, excitation_difference_cm1=lv["excitation_cm1"]-excite,
                paper_DF_binding_cm1=bind, binding_difference_cm1=lv["valence_energy_hartree"]*HARTREE_CM1-bind))
        rmes = []
        for line in lines:
            pair = (levels[line.initial_state_index]["label"], levels[line.final_state_index]["label"])
            if line.multipole != "E1" or pair not in dipoles: continue
            lower, upper = [KEYS[x] for x in pair]
            Pi, Qi = r.orbitals_dict[lower]; Pf, Qf = r.orbitals_dict[upper]
            static = angular_eL(lower[1],upper[1],1)*radial_eL(Pi,Qi,Pf,Qf,r.bsp_data,1)
            dynamic = angular_eL(lower[1],upper[1],1)*radial_eL(Pi,Qi,Pf,Qf,r.bsp_data,1,
                kappa_i=lower[1], kappa_f=upper[1], omega=line.energy_diff_hartree)
            signed = line.reduced_matrix_element_au
            rmes.append(dict(lower=pair[0], upper=pair[1], paper_DF_dipole_au=dipoles[pair],
                native_static_RME_au=static, native_CSF_finite_frequency_RME_au=signed,
                CSF_vs_orbital_magnitude_error_au=abs(abs(signed)-abs(dynamic)),
                static_vs_paper_magnitude_error_au=abs(abs(static)-dipoles[pair])))
        rows.append(dict(base=base, identity=identity, CI_max_residual_Ha=max(res), replay_max_energy_error_Ha=drift,
                         energies=energies, dipoles=rmes))
    factors = json.loads((OLD/"results/v68_default_corrections/factors.json").read_text())["records"]
    pub_B1 = {"6s_1/2":128, "5d_3/2":156, "5d_5/2":114, "6p_1/2":91, "6p_3/2":63}
    breit = [dict(label=r["label"], native_static_Breit_transition_cm1=r["excitation_shift_cm1"],
                  published_first_order_Breit_transition_cm1=pub_B1[r["label"]]-128,
                  difference_cm1=r["excitation_shift_cm1"]-(pub_B1[r["label"]]-128)) for r in factors]
    passed = all(row["replay_max_energy_error_Ha"]<1e-8 and row["CI_max_residual_Ha"]<1e-8
        and all(abs(e["excitation_difference_cm1"])<1 and abs(e["binding_difference_cm1"])<1 for e in row["energies"])
        and all(d["CSF_vs_orbital_magnitude_error_au"]<1e-10 and d["static_vs_paper_magnitude_error_au"]<5e-5 for d in row["dipoles"]) for row in rows)
    passed = passed and all(abs(b["difference_cm1"])<1 for b in breit)
    save(out/"results.json", dict(passed=passed, baseline_replays=rows, retained_Breit_controls=breit,
        Breit_current_recomputed=False, Breit_scope="Retained original current Breit source; comparison to published first-order values, which include frequency dependence, at printed integer precision only"))
    print("BASELINE AUDIT", passed, flush=True)
    signal.alarm(0)


if __name__ == "__main__": main()
