import numpy as np
from atomx.scattering.r_matrix import compute_multichannel, heplus_target_states

def phase(Z, S, k, n_max=1, a=25.0, n_basis=60):
    targets = list(heplus_target_states(Z=Z, n_max=n_max).values())
    E = -0.5*Z*Z + k*k/2
    res = compute_multichannel(targets, L_total=0, S_total=S, parity=+1,
                               energies_au=np.array([E]), Z=Z, a_au=a,
                               n_basis=n_basis, n_gauss=12, l_c_max=2)
    return res.matchings[0].eigenphase_sum

# 1s-only static+exchange
for S, name in [(0.0,"1S"), (1.0,"3S")]:
    d1 = phase(1.0, S, 0.1, n_max=1)
    d2 = phase(1.0, S, 0.1, n_max=1, n_basis=100)
    d3 = phase(1.0, S, 0.1, n_max=2)
    print(f"e+H {name} k=0.1: 1s-only nb=60: {d1:.4f} ({(d1+np.pi)%np.pi:.4f} mod pi); "
          f"nb=100: {d2:.4f}; 1s2s2p: {d3:.4f} ({(d3+np.pi)%np.pi:.4f} mod pi)")
print("Reference: exact Schwartz 1S k=0.1: 2.3960; 3S: 2.9088 (mod pi: -0.7456, -0.2328)")
print("Static-exchange literature (approx): 1S ~2.55, 3S ~2.94 rad")
