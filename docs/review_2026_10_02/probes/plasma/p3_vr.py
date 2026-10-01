"""NEW: van Regemorter fallback rate in crm.py -- is 1/dE missing?

Standard vR (from Omega_Bethe = (8 pi/sqrt3) g_l f gaunt ln(u)/dE_Ry,
C = 8.629e-6 Upsilon e^-x/(g_l sqrt(T_K))):

    C_exc = 1.578e-5 * f * gaunt_eff * exp(-x) / (dE_eV * sqrt(T_eV))  cm3/s

The constant 1.578e-5 = 8.629e-6 * (8 pi/sqrt 3) * 13.6057 / sqrt(k_B in K/eV)
already contains R_H = 13.6057 eV, so it is only consistent WITH the 1/dE.
"""
import numpy as np
from scipy.special import exp1

from atomx.constants import BOLTZMANN_EV, RYDBERG_EV
from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    CRMLevel,
    CRMTransition,
    effective_collision_strength,
)

NE = 1.0e10


def code_vr_rate(dE, f, g_l, g_u, T):
    tr = CRMTransition(upper=1, lower=0, A_rate=1e8, gf=g_l * f)
    crm = CollisionalRadiativeModel(
        [CRMLevel("l", 0.0, g=g_l), CRMLevel("u", dE, g=g_u)], [tr])
    R = crm._build_rate_matrix(NE, T)
    return R[1, 0] / NE


def correct_vr_rate(dE, f, g_l, T):
    x = dE / T
    exE1 = np.exp(x) * exp1(x) if x < 700 else (x + 1) / (x * (x + 2))
    gaunt = max(0.2, 0.276 * exE1)
    return 1.578e-5 * f * gaunt * np.exp(-x) / (dE * np.sqrt(T))


def omega_path_rate(dE, f, g_l, T, gaunt0=0.276):
    """The module's own Omega path fed with the Bethe/vR Omega table."""
    E = np.geomspace(dE * 1.0001, dE + 500 * T, 4000)
    om = (8 * np.pi / np.sqrt(3)) * g_l * f * gaunt0 * np.log(E / dE) / (dE / RYDBERG_EV)
    ups = effective_collision_strength(E, om, dE, T)
    T_K = T / BOLTZMANN_EV
    return 8.6291e-6 * ups / (g_l * np.sqrt(T_K)) * np.exp(-dE / T)


def main():
    print(f"{'dE':>7} {'T':>7} {'code':>11} {'correct vR':>11} {'code/corr':>10}")
    for dE, f, g_l, g_u, T in ((10.0, 0.5, 1, 3, 20.0),
                               (100.0, 0.3, 2, 4, 100.0),
                               (1.0, 0.5, 1, 1, 10.0),
                               (13.6, 0.416, 2, 6, 5.0)):
        c_code = code_vr_rate(dE, f, g_l, g_u, T)
        c_corr = correct_vr_rate(dE, f, g_l, T)
        print(f"{dE:7.1f} {T:7.1f} {c_code:11.4e} {c_corr:11.4e} {c_code/c_corr:10.3f}")

    # cross-check the Omega path against the correct closed form
    dE, f, g_l, T = 10.0, 0.5, 1, 20.0
    c_om = omega_path_rate(dE, f, g_l, T)
    c_corr = correct_vr_rate(dE, f, g_l, T)
    print(f"Omega-path {c_om:.4e} vs closed vR {c_corr:.4e}  ratio {c_om/c_corr:.3f}")


if __name__ == "__main__":
    main()
