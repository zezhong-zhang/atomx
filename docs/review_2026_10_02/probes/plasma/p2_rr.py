"""F6 verdict + independent Milne/Maxwell integration of the Kramers RR."""
import numpy as np
from scipy.integrate import quad
from scipy.special import exp1

from atomx.plasma.crm import (
    radiative_recombination_rate_cm3_s,
    saha_ionization_density_cm3,
    _planck_photoionization_rate_s,
    _hydrogenic_sigma0_cm2,
)

# CODATA cgs
M_E = 9.1093837015e-28   # g
EV_ERG = 1.602176634e-12
C = 2.99792458e10        # cm/s
H = 6.62607015e-27       # erg s


def milne_rr_direct(I_eV, T_eV, g_b=1.0, g_c=1.0):
    """From-scratch spontaneous Maxwellian RR via the Milne relation.

    sigma_fb(v) = (g_b/g_c) (h nu)^2/(m^2 c^2 v^2) sigma_bf(nu)   [Milne]
    alpha = <v sigma_fb> with a 3-D Maxwellian; Kramers sigma_bf.
    """
    sigma0 = _hydrogenic_sigma0_cm2(I_eV)  # same cross-section the module integrates
    kT = T_eV * EV_ERG
    I = I_eV * EV_ERG

    def integrand(eps):  # eps = electron energy in erg
        v = np.sqrt(2.0 * eps / M_E)
        E_ph = I + eps
        sigma_bf = sigma0 * (I / E_ph) ** 3
        sigma_fb = (g_b / g_c) * E_ph ** 2 / (M_E ** 2 * C ** 2 * v ** 2) * sigma_bf
        fv = 4.0 * np.pi * v ** 2 * (M_E / (2.0 * np.pi * kT)) ** 1.5 * np.exp(-eps / kT)
        dv = 1.0 / (M_E * v)  # d eps = m v dv
        return v * sigma_fb * fv * dv

    val, _ = quad(integrand, 0.0, 200.0 * kT, epsabs=0.0, epsrel=1e-11)
    return val


def main():
    I = 13.6
    print(f"{'T(eV)':>10} {'module':>12} {'direct Milne':>13} {'ratio':>9}")
    for T in (0.01, 1.0, 13.6, 136.0, 1360.0):
        a_mod = radiative_recombination_rate_cm3_s(I, T, g_bound=1, g_continuum=1)
        a_dir = milne_rr_direct(I, T)
        print(f"{T:10.2f} {a_mod:12.5e} {a_dir:13.5e} {a_mod/a_dir:9.5f}")

    # old bug: zero at T=0.01 eV via Saha underflow
    a = radiative_recombination_rate_cm3_s(13.6, 0.01)
    print(f"T=0.01 eV: module alpha = {a:.5e} cm3/s (review's direct value 7.6332e-13)")
    assert a > 0

    # spontaneous-only must NOT include the 1/(1-exp(-(I+E)/T)) bath factor:
    # ratio module/spontaneous must be 1 at high T (was 24 at 1360 eV pre-fix)
    for T in (13.6, 136.0, 1360.0):
        a_mod = radiative_recombination_rate_cm3_s(I, T, g_bound=1, g_continuum=1)
        a_dir = milne_rr_direct(I, T)
        assert abs(a_mod / a_dir - 1) < 2e-3, (T, a_mod / a_dir)

    # Planck LTE consistency: alpha(spon+stim) * Saha = Gamma_PI
    for (Ii, Ti) in ((13.6, 5.0), (100.0, 100.0), (13.6, 13.6)):
        saha = saha_ionization_density_cm3(Ii, Ti, g_bound=1, g_continuum=1)
        alpha_tot = radiative_recombination_rate_cm3_s(
            Ii, Ti, g_bound=1, g_continuum=1, radiation_temperature_eV=Ti)
        pi = _planck_photoionization_rate_s(Ii, Ti)
        print(f"I={Ii} T={Ti}: alpha*Saha/PI = {alpha_tot * saha / pi:.12f}")

    print("P2-rr: F6 FIXED (spontaneous Milne kernel, no Saha underflow)")


if __name__ == "__main__":
    main()
