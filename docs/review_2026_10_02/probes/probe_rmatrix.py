import numpy as np
from atomx.scattering.r_matrix import compute_multichannel, heplus_target_states
from atomx.scattering.r_matrix.cc_outer import match_K_matrix, channel_R_matrix, S_matrix_from_K

targets = list(heplus_target_states(Z=2.0).values())
# Energy above the n=2 thresholds (He+ 2s/2p at -0.5 Ha): total E=-0.45 Ha => 2 open target states
res = compute_multichannel(targets, L_total=0, S_total=0, parity=+1,
                           energies_au=np.array([-0.45]), Z=2.0, a_au=25.0,
                           n_basis=40, n_gauss=10, l_c_max=2,
                           propagate_long_range=True)
m = res.matchings[0]
print("open channels:", m.open_indices, "k:", m.k_diag)
K = m.K_matrix
print("K=\n", K)
print("K symmetric defect:", np.max(np.abs(K-K.T)))
S = m.S_matrix
print("S unitarity defect:", np.max(np.abs(S.conj().T@S - np.eye(len(S)))))
print("S symmetric defect:", np.max(np.abs(S-S.T)))
OM = m.collision_strength_matrix
print("Omega=\n", OM)
print("Omega reciprocity defect |O12-O21|:", abs(OM[0,1]-OM[1,0]) if OM.shape[0]>1 else "n/a")
sig = m.cross_section_matrix_a0
print("sigma (a0^2)=\n", sig)
# Reciprocity of cross sections: g_i k_i^2 sigma_ij = g_j k_j^2 sigma_ji (targets have different statistical weights)
if OM.shape[0] > 1:
    ki, kj = m.k_diag
    print("ki^2 sigma_12 / (kj^2 sigma_21) =", ki*ki*sig[0,1]/(kj*kj*sig[1,0]))
