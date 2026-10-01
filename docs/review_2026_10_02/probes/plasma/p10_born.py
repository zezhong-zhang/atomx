"""Adjacent slice check: born_excitation_dipole vs the Bethe cross section.

Bethe (Inokuti 1971): sigma = 4 pi a0^2 (R/T) M^2 ln(4 c T/R),
M^2 = (3/4)(R/dE) f.  vR matches Bethe at high u iff
sigma_vR = pi a0^2 (8 pi/sqrt3) f gaunt ln(u) / (dE_Ry * T_Ry).
born.py computes sigma = 8 pi^2 a0^2 f gaunt ln(u) / (sqrt3 * T_Ha),
i.e. it is missing 1/dE_Ry and carries an extra factor 2 (Ha vs Ry):
ratio born/vR_standard = 2 * dE_Ry.
"""
import numpy as np
from atomx.constants import BOHR_RADIUS_M, HARTREE_EV, RYDBERG_EV
from atomx.scattering.distorted_wave.born import born_excitation_dipole

a0 = BOHR_RADIUS_M * 100.0

for dE_eV, f in ((10.199, 0.4162), (100.0, 0.2), (5.0, 0.5)):
    for u in (10.0, 100.0):
        E = u * dE_eV
        s_code = born_excitation_dipole(np.array([E]), f_ab=f, delta_E_eV=dE_eV)[0]
        # vR/Bethe standard, energies in Rydbergs
        T_Ry, dE_Ry = E / RYDBERG_EV, dE_eV / RYDBERG_EV
        s_std = (np.pi * a0**2) * (8 * np.pi / np.sqrt(3)) * f * 0.2 * np.log(u) / (dE_Ry * T_Ry)
        print(f"dE={dE_eV:8.3f} eV u={u:6.0f}: born.py={s_code:.4e}  "
              f"vR-standard={s_std:.4e}  ratio={s_code/s_std:6.3f}  (2*dE_Ry={2*dE_Ry:.3f})")
