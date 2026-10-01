import numpy as np
from atomx.scattering.r_matrix import compute_multichannel, heplus_target_states

def phase(S, k, labels, n_basis=60):
    all_t = heplus_target_states(Z=1.0, n_max=2)
    targets = [all_t[l] for l in labels]
    E = -0.5 + k*k/2
    res = compute_multichannel(targets, L_total=0, S_total=S, parity=+1,
                               energies_au=np.array([E]), Z=1.0, a_au=25.0,
                               n_basis=n_basis, n_gauss=12, l_c_max=2)
    return res.matchings[0].eigenphase_sum

for S, name, exact in [(0.0, "1S", 2.3960), (1.0, "3S", 2.9088)]:
    for labels in (["1s"], ["1s","2s"], ["1s","2p"], ["1s","2s","2p"]):
        d = phase(S, 0.1, labels)
        print(f"e+H {name} k=0.1 targets={str(labels):18s}: delta={d:+.4f} ({(d+np.pi)%np.pi:.4f} mod pi)   exact {exact}")
