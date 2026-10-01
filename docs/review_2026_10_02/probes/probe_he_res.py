import numpy as np
from atomx.scattering.r_matrix import compute_multichannel, heplus_target_states, extract_resonances

targets = list(heplus_target_states(Z=2.0).values())
energies = np.linspace(-0.85, -0.60, 500)
res = compute_multichannel(targets, L_total=0, S_total=0, parity=+1,
                           energies_au=energies, Z=2.0, a_au=25.0, n_basis=60,
                           n_gauss=12, l_c_max=2, include_l2_pseudostates=False)
fits = extract_resonances(res)
for f in fits:
    print(f"1Se resonance: E_r = {f.E_r_au:.5f} Ha, Gamma = {f.Gamma_au:.5f} Ha")
print("Burke 1971 / accepted: E_r = -0.77816 Ha (i.e. 57.06 eV above He+ 1s... total), Gamma = 0.0050 Ha")
# eigenphase sum profile sanity
d = res.eigenphase_sum
print("eigenphase sum min/max over scan:", d.min(), d.max())
