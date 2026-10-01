import numpy as np
from atomx.scattering.r_matrix import heplus_target_states
from atomx.scattering.r_matrix.channel_table import enumerate_channels
from atomx.scattering.r_matrix.cc_inner import assemble_cc_inner_region
from atomx.scattering.r_matrix.cc_outer import channel_R_matrix, match_K_matrix

targets = list(heplus_target_states(Z=1.0, n_max=2).values())
k = 0.1
E = -0.5 + k*k/2
channels = enumerate_channels(targets, 0, +1, l_c_max=2, S_total=1.0)
print("channels:", [(c.target.label, c.l_c) for c in channels])
inner = assemble_cc_inner_region(channels=channels, L_total=0, S_total=1.0, Z=1.0,
                                 a_au=25.0, n_basis=60, k_spline_order=6, n_gauss=12)
R_full = channel_R_matrix(E, inner.eigenvalues_au, inner.surface_amplitudes, 25.0)
for treatment in ("drop", "log_derivative"):
    m = match_K_matrix(E, channels, R_full, 25.0, closed_channel_treatment=treatment,
                       L_total=0, S_total=1.0)
    print(f"3S k=0.1 closed={treatment}: eigenphase sum = {m.eigenphase_sum:.4f} "
          f"({(m.eigenphase_sum+np.pi)%np.pi:.4f} mod pi)  [exact 2.9088]")

# convergence of the log_derivative path
for nb, a in [(60, 25.0), (100, 25.0), (60, 40.0)]:
    inner = assemble_cc_inner_region(channels=channels, L_total=0, S_total=1.0, Z=1.0,
                                     a_au=a, n_basis=nb, k_spline_order=6, n_gauss=12)
    R_full = channel_R_matrix(E, inner.eigenvalues_au, inner.surface_amplitudes, a)
    m = match_K_matrix(E, channels, R_full, a, closed_channel_treatment="log_derivative",
                       L_total=0, S_total=1.0)
    print(f"   nb={nb} a={a}: delta={(m.eigenphase_sum+np.pi)%np.pi:.4f}")
