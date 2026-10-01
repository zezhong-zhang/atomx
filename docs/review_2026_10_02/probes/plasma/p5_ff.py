"""NEW: free-free emissivity constant + Gaunt factor + units."""
import numpy as np
from scipy import constants as sc

from atomx.constants import BOLTZMANN_EV
from atomx.spectrum.continuum import (
    _C_FF_ERG_CM3_HZ,
    free_free_emissivity,
    gaunt_ff_born,
)


def main():
    # Rybicki & Lightman 5.14b prefactor from CODATA (cgs; e in statC):
    # C_ff = 2^5 pi e^6 / (3 m c^3) * sqrt(2 pi / (3 k m))
    e_esu = sc.e * sc.c * 10.0  # e [C] * c [m/s] * 10 -> statC (exact relation)
    e_esu = 4.803204712570263e-10
    m = 9.1093837015e-28
    c = 2.99792458e10
    k = 1.380649e-16
    cff = (2.0**5 * np.pi * e_esu**6 / (3.0 * m * c**3)
           * np.sqrt(2.0 * np.pi / (3.0 * k * m)))
    print(f"C_ff from CODATA: {cff:.6e}   module: {_C_FF_ERG_CM3_HZ:.6e}   "
          f"ratio {_C_FF_ERG_CM3_HZ / cff:.6f}   (textbook 6.8e-38)")

    # Gaunt factor limits
    x = np.array([1e-4, 1.0, 30.0])
    g = gaunt_ff_born(x)
    print("gaunt_ff_born:", dict(zip(x.tolist(), np.round(g, 5).tolist())))
    # x->0 limit: (sqrt3/pi) ln(4/(1.781 x))
    print("small-x closed form:", np.sqrt(3)/np.pi*np.log(4/(1.781*x[0])))

    # total emissivity integral check at T = 100 eV, Z=1, ne=ni=1e20:
    # int eps_E dE should equal the RL integrated free-free power
    # 1.426e-27 Z^2 sqrt(T_K) ne ni <g_B> erg/s/cm3 (RL 5.15b)
    T_eV = 100.0
    E = np.geomspace(1e-3, 1e5, 200000)
    eps = free_free_emissivity(E, T_eV, 1.0, n_e_cm3=1e20, n_i_cm3=1e20)
    tot_eV = np.trapezoid(eps, E)                    # eV s-1 cm-3
    tot_erg = tot_eV * 1.602176634e-12
    T_K = T_eV / BOLTZMANN_EV
    rl = 1.426e-27 * np.sqrt(T_K) * 1e40             # erg s-1 cm-3
    print(f"integrated free-free: {tot_erg:.4e} erg/s/cm3   "
          f"RL 5.15b (g_B~1): {rl:.4e}   ratio {tot_erg/rl:.3f}")

    # per-steradian question: line emissivity n_u A h nu is a 4 pi-integrated
    # total; the ff constant is RL's volume emissivity (also 4 pi-integrated).
    # verified consistent by construction.


if __name__ == "__main__":
    main()
