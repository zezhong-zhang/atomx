import numpy as np, time
from atomx.scattering.r_matrix import compute_multichannel, heplus_target_states

targets = list(heplus_target_states(Z=1.0).values())  # hydrogen 1s,2s,2p
# static-exchange-ish (3-state CC incl. exchange) e-H s-wave singlet/triplet
# Schwartz 1961 exact singlet: k=0.1 -> 2.3960; k=0.2 -> 2.0646; triplet: 2.9088, 2.7171
for S, name in [(0.0, "1S"), (1.0, "3S")]:
    for k in [0.1, 0.2, 0.3]:
        E = -0.5 + k*k/2
        res = compute_multichannel(targets, L_total=0, S_total=S, parity=+1,
                                   energies_au=np.array([E]), Z=1.0, a_au=25.0,
                                   n_basis=60, n_gauss=12, l_c_max=2)
        delta = res.matchings[0].eigenphase_sum
        print(f"e+H {name} k={k}: eigenphase sum = {delta:.4f} rad")
