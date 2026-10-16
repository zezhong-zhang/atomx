"""Independent projected-Fock commutator; diagnostic, not a dressed current."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import signal
import numpy as np
from scipy.linalg import block_diag, solve
from common import ROOT, KEYS, load_reference, save, sha


def main():
    from atomx.methods.mbpt import _build_closed_shell_fock
    from atomx.numerics.bspline.convol import ConvolCache
    from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, _get_keep_grid
    from atomx.numerics.radial_workspace import eval_orbital_pair
    from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity
    from atomx.properties.transitions.angular import angular_eL
    out = ROOT/"gauge_audit_v3"
    out.mkdir(exist_ok=False)
    save(out/"contract.json", dict(hypothesis="Bare velocity mismatch is supplied by the nonlocal Fock interaction commutator",
        scope="Static dipole and leading photon velocity at identical retained V68 orbitals; radial projection explicitly retained",
        closure_gate_radial_au=1e-6, local_commutator_gate_radial_au=1e-6,
        published_length_RME_magnitude_gate_au=5e-5,
        published_velocity_observable="omega*V, compared within the Table I printed energy interval (+/-0.5 cm-1 per orbital) and Table III RME rounding (+/-0.000005 au). Compare relative V/L sign; orbital/reduced-tensor phase conventions need not coincide.",
        previous_gate="v2 raw signed V comparison failed; retained, not relaxed. Current observable removes the small transition denominator before comparing independently printed orbital energies.",
        published_source="Safronova 2009 Table III: https://doi.org/10.1103/PhysRevA.79.022512",
        max_seconds=120, script_sha256=sha(__file__),
        not_claimed="No RPA or correlated-current implementation; commutator closure is not an independent gauge validation of the approximate physical wavefunction"))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second gauge cap")))
    signal.alarm(120)
    core, result, levels, identity = load_reference()
    data = core._bsp_data; ns = data["ns"]
    cache = ConvolCache(data)
    save(out/"identity.json", identity)
    keep, grid = _get_keep_grid(ns,data["nsp"],data["nsq"],1,2)
    eye = np.eye(ns); zero = np.zeros(ns)
    bp = np.column_stack([eval_orbital_pair(e,zero,data)[0].ravel() for e in eye])
    bq = np.column_stack([eval_orbital_pair(zero,e,data)[1].ravel() for e in eye])
    wr = (data["grw"]*data["gr"]).ravel()
    T = block_diag(bp.T@(wr[:,None]*bp),bq.T@(wr[:,None]*bq))[grid]
    operators = {}
    for _, kap in KEYS.values():
        H, S = _build_closed_shell_fock(core,kap,cache=cache,potential_mode="VN")
        H0, _ = build_dirac_matrices(ns,kap,70,data,core._nuclear_potential)
        sym = lambda M: np.tril(M)+np.tril(M,-1).T
        H, H0, S = sym(H[grid]), sym(H0[grid]), S[grid]
        operators[kap] = (H,H0,H-H0,S)
    eps = {r["label"]:r["valence_energy_hartree"] for r in levels}
    pairs = [("6s_1/2","6p_1/2"),("6s_1/2","6p_3/2"),
             ("5d_3/2","6p_1/2"),("5d_3/2","6p_3/2"),("5d_5/2","6p_3/2")]
    rows = []
    published = [(3.24219,3.09818),(4.54262,4.32196),(3.86104,-23.75603),
                 (1.69693,.01089),(5.20010,-1.16778)]
    published_binding_cm1 = {"6s_1/2":-90789,"5d_3/2":-66517,"5d_5/2":-66037,
                             "6p_1/2":-66087,"6p_3/2":-63276}
    from atomx.constants import HARTREE_CM1
    for lower, upper in pairs:
        ki, kf = KEYS[lower], KEYS[upper]
        Pi, Qi = result.orbitals_dict[ki]; Pf, Qf = result.orbitals_dict[kf]
        vi, vf = np.r_[Pi,Qi][keep], np.r_[Pf,Qf][keep]
        Hi,H0i,Ui,Si = operators[ki[1]]; Hf,H0f,Uf,Sf = operators[kf[1]]
        omega = eps[upper]-eps[lower]
        length = radial_eL(Pi,Qi,Pf,Qf,data,1)
        bare = radial_eL_velocity(Pi,Qi,Pf,Qf,data,1,kappa_i=ki[1],kappa_f=kf[1],omega=omega,long_wavelength=True)
        comm = lambda Af,Ai: float(vf@(Af@solve(Sf,T@vi,assume_a="pos")-T@solve(Si,Ai@vi,assume_a="pos"))/omega)
        local, interaction, full = comm(H0f,H0i), comm(Uf,Ui), comm(Hf,Hi)
        angular = angular_eL(ki[1],kf[1],1)
        pl, pv = published[len(rows)]
        aligned_V = bare*angular*np.sign(length*angular)
        paper_omega_cm1 = published_binding_cm1[upper]-published_binding_cm1[lower]
        current_difference = omega*HARTREE_CM1*aligned_V-paper_omega_cm1*pv
        current_interval = abs(pv)*1.+abs(paper_omega_cm1)*5e-6+5e-6
        rows.append(dict(lower=lower,upper=upper,omega_Ha=omega, length_radial_au=length,
            length_RME_au=length*angular, velocity_RME_au=bare*angular,
            paper_signed_length_RME_au=pl, paper_signed_velocity_RME_au=pv,
            length_vs_paper_au=length*angular-pl, velocity_vs_paper_au=bare*angular-pv,
            paper_printed_omega_cm1=paper_omega_cm1,
            phase_independent_length_magnitude_difference_au=abs(length*angular)-pl,
            phase_independent_current_difference_cm1_au=current_difference,
            paper_current_rounding_interval_cm1_au=current_interval,
            bare_velocity_radial_au=bare, local_Hamiltonian_commutator_radial_au=local,
            Fock_interaction_commutator_radial_au=interaction, full_Fock_commutator_radial_au=full,
            length_minus_bare_radial_au=length-bare,
            local_minus_bare_radial_au=local-bare,
            closure_radial_au=length-bare-interaction))
    passed = all(abs(r["closure_radial_au"])<1e-6 and abs(r["local_minus_bare_radial_au"])<1e-6 for r in rows)
    passed = passed and all(abs(r["phase_independent_length_magnitude_difference_au"])<5e-5
        and abs(r["phase_independent_current_difference_cm1_au"])<=r["paper_current_rounding_interval_cm1_au"] for r in rows)
    save(out/"results.json",dict(passed=passed,records=rows))
    print("GAUGE COMMUTATOR", passed, [(r["lower"],r["upper"],r["closure_radial_au"]) for r in rows],flush=True)
    signal.alarm(0)


if __name__=="__main__":main()
