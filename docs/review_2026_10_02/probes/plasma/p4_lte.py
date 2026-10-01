"""NEW: thermodynamic relaxation of the CRM.

(a) closed 3-level system (g, e, continuum) with EII+TBR+RR, no field:
    high ne -> Boltzmann within the ion AND Saha across stages.
(b) coronal limit: n_c/n_g -> S_EII/alpha_RR at low ne.
(c) bound-bound radiative coupling to the field: with a Planck bath at
    T_rad = T_e and low density, does the bound-bound pair reach
    Boltzmann?  (Only spontaneous A + collisional terms exist for
    bound-bound; photoexcitation/stimulated emission are not in the
    rate matrix.)
"""
import numpy as np

from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    CRMBoundFreeChannel,
    CRMLevel,
    CRMTransition,
    radiative_recombination_rate_cm3_s,
    saha_ionization_density_cm3,
)

CHI = 20.0   # ionization potential of the bound stage
DE = 5.0     # bound-bound gap
TE = 10.0


def build():
    levels = [CRMLevel("g", 0.0, g=1),
              CRMLevel("e", DE, g=3),
              CRMLevel("c", CHI, g=1)]
    trs = [CRMTransition(upper=1, lower=0, A_rate=1e8, gf=1.0)]
    ch = CRMBoundFreeChannel(bound=0, continuum=2,
                             ionization_potential_eV=CHI,
                             g_bound=1, g_continuum=1,
                             eii_rate_cm3_s=1e-8)
    return CollisionalRadiativeModel(levels, trs, bound_free_channels=[ch])


def main():
    crm = build()
    saha = saha_ionization_density_cm3(CHI, TE, g_bound=1, g_continuum=1)
    alpha = radiative_recombination_rate_cm3_s(CHI, TE, g_bound=1, g_continuum=1)

    print("== (a) collisional LTE approach, no radiation ==")
    print(f"{'ne':>10} {'n_e/n_g / Boltz':>18} {'n_c*n_e/n_g / Saha':>20}")
    for ne in (1e14, 1e18, 1e22, 1e26):
        pop = crm.steady_state(ne, TE, include_radiative_recombination=True)
        b_rat = 3.0 * np.exp(-DE / TE)
        s_rat = saha / ne
        # note: ground of bound stage couples to continuum; excited 'e' is
        # bound-bound only.  Compare n_e/n_g to Boltzmann, and the
        # ionization ratio to Saha.
        print(f"{ne:10.1e} {pop[1]/pop[0]/b_rat:18.6f} {pop[2]/pop[0]/s_rat:20.6f}")

    print("== (b) coronal limit ==")
    for ne in (1e2, 1e6, 1e10):
        pop = crm.steady_state(ne, TE)
        pred = 1e-8 / alpha
        print(f"ne={ne:8.1e}  n_c/n_g={pop[2]/pop[0]:.6e}  S/alpha_RR={pred:.6e}")

    print("== (c) Planck bath at T_rad = T_e, bound-bound pair ==")
    for ne in (1e6, 1e10, 1e18):
        pop = crm.steady_state(ne, TE, radiation_temperature_eV=TE)
        b_rat = 3.0 * np.exp(-DE / TE)
        s_rat = saha / ne
        print(f"ne={ne:8.1e}  n_e/n_g={pop[1]/pop[0]:.4e}  Boltzmann={b_rat:.4e}"
              f"   n_c/n_g/(Saha/ne)={pop[2]/pop[0]/s_rat:.6f}")


if __name__ == "__main__":
    main()
