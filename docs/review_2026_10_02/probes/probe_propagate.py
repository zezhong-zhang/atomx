import numpy as np
from atomx.scattering.r_matrix import heplus_target_states
from atomx.scattering.r_matrix.channel_table import enumerate_channels
from atomx.scattering.r_matrix.cc_inner import assemble_cc_inner_region
from atomx.scattering.r_matrix.cc_outer import channel_R_matrix, match_K_matrix

targets = [heplus_target_states(Z=2.0)["1s"]]
k = 1.0
E = -2.0 + k*k/2
for S in (0.0, 1.0):
    channels = enumerate_channels(targets, 0, +1, l_c_max=2, S_total=S)
    inner = assemble_cc_inner_region(channels=channels, L_total=0, S_total=S, Z=2.0,
                                     a_au=25.0, n_basis=60, k_spline_order=6, n_gauss=12)
    R_full = channel_R_matrix(E, inner.eigenvalues_au, inner.surface_amplitudes, 25.0)
    m0 = match_K_matrix(E, channels, R_full, 25.0, L_total=0, S_total=S,
                        propagate_long_range=False)
    m1 = match_K_matrix(E, channels, R_full, 25.0, L_total=0, S_total=S,
                        propagate_long_range=True, R_asym=80.0)
    print(f"e+He+ 1s S={int(S)}: K direct={m0.K_matrix[0,0]:.8f}  K propagated={m1.K_matrix[0,0]:.8f}  rel diff={abs(m0.K_matrix[0,0]-m1.K_matrix[0,0])/abs(m0.K_matrix[0,0]):.2e}")
