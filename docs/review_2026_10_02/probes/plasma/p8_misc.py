"""NEW assorted: cascade cycles, crm.spectrum() mass, screening_shift limits,
Saha constant vs CODATA."""
import warnings

import numpy as np

from atomx.plasma.cascade import (
    CascadeLevel,
    CascadeTransition,
    cascade_yields,
    fluorescence_yield,
)
from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel, CRMTransition, saha_ionization_density_cm3
from atomx.plasma.screening import (
    debye_length,
    ion_sphere_potential,
    ion_sphere_radius,
    screening_shift,
)


def main():
    print("== cascade with a feedback cycle ==")
    levels = [CascadeLevel("a", 3.0), CascadeLevel("b", 2.0), CascadeLevel("c", 0.0)]
    trs = [
        CascadeTransition(upper=0, lower=1, rate_per_s=1.0, photon_eV=1.0),
        CascadeTransition(upper=1, lower=0, rate_per_s=1.0, photon_eV=1.0),  # cycle!
        CascadeTransition(upper=1, lower=2, rate_per_s=1.0),
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = cascade_yields(levels, trs, 0)
        print("warnings:", [str(x.message) for x in w])
    print("terminal yields:", res.terminal_yields,
          " photon yield:", fluorescence_yield(res))
    print("(population 'lost' in the cycle:",
          1.0 - sum(res.terminal_yields.values()), ")")

    print("== crm.spectrum() Doppler width ==")
    levels = [CRMLevel("g", 0.0, g=1), CRMLevel("e", 10.0, g=3)]
    trs = [CRMTransition(upper=1, lower=0, A_rate=1e10, gf=0.5)]
    crm = CollisionalRadiativeModel(levels, trs)
    lam, inten = crm.spectrum(1e14, 100.0, 1000.0, 1500.0, n_points=20001)
    i0 = int(np.argmax(inten))
    lam0 = lam[i0]
    half = inten[i0] / 2
    above = lam[inten > half]
    fwhm = above.max() - above.min()
    want_1u = lam0 * np.sqrt(100.0 / 931.5e6) * 2.355
    want_fe = lam0 * np.sqrt(100.0 / (931.5e6 * 55.85)) * 2.355
    print(f"line at {lam0:.1f} A: FWHM {fwhm:.4f} A; "
          f"1-u prediction {want_1u:.4f}, Fe prediction {want_fe:.4f}")
    print("(no mass argument in signature -> hard-coded 1 u)")

    print("== screening_shift limits (Debye) ==")
    Z, ne, Te = 10, 1e22, 100.0
    lam_D = debye_length(ne, Te)
    r = np.geomspace(1e-6, 1e6, 4000)
    dV = screening_shift(Z, r, ne, Te, "debye")
    print(f"dV(r->0) = {dV[0]:.6f} Ha, Z/lam_D = {Z/lam_D:.6f} Ha")
    print(f"dV(r->inf) = {dV[-1]:.3e} Ha  (docstring says it tends to "
          "Z/lambda at LARGE r)")
    print(f"IPD (debye model) = {Z/lam_D*27.211386245988:.4f} eV")

    print("== ion-sphere potential continuity at R_IS ==")
    R = ion_sphere_radius(2, 1e23)
    V = ion_sphere_potential(10, 2, 1e23)
    eps = 1e-8 * R
    print(f"V(R-eps)={V(R-eps):.8f}  V(R+eps)={V(R+eps):.8f}")

    print("== Saha constant vs CODATA ==")
    from scipy import constants as sc
    s = saha_ionization_density_cm3(13.6, 10.0, g_bound=1, g_continuum=1)
    lam = sc.h / np.sqrt(2 * np.pi * sc.m_e * sc.k * (10.0 * sc.e / sc.k))
    s_ref = 2.0 / lam**3 * np.exp(-13.6 / 10.0) * 1e-6  # m^-3 -> cm^-3
    print(f"module {s:.6e}  CODATA {s_ref:.6e}  ratio {s/s_ref:.8f}")


if __name__ == "__main__":
    main()
