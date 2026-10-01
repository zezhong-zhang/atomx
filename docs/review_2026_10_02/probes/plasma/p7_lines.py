"""NEW: lineshape normalization + Spectrum.add_line area conservation,
incl. the logarithmic-grid edge case."""
import numpy as np

from atomx.lineshapes import doppler_width, gaussian, lorentzian, natural_width, voigt
from atomx.spectrum import Spectrum


def main():
    x = np.linspace(-100, 100, 4_000_001)
    for name, prof in (("gaussian", gaussian(x, 0.0, 2.0)),
                       ("lorentzian", lorentzian(x, 0.0, 3.0)),
                       ("voigt", voigt(x, 0.0, 2.0, 3.0))):
        print(f"{name}: integral = {np.trapezoid(prof, x):.8f}")

    # natural width: HWHM of lorentzian(x, 0, A/4pi) must be A/(4pi)
    A = 1.0e9
    g = natural_width(A)
    p = lorentzian(np.array([0.0, g]), 0.0, g)
    print(f"natural_width HWHM check: peak/half = {p[0]/p[1]:.6f} (want 2)")

    # doppler width numeric: H Lya at 1e4 K -> sigma/nu = sqrt(kT/mc2)
    s = doppler_width(1215.67, 1e4, 1.008)
    kb, mu, c = 1.380649e-23, 1.66053906892e-27, 299792458.0
    want = 1215.67 * np.sqrt(kb * 1e4 / (1.008 * mu * c**2))
    print(f"doppler_width: {s:.6e} vs closed {want:.6e}")

    # add_line area conservation on a linear grid
    spec = Spectrum.linear(100.0, 200.0, 20001)
    spec.add_line(150.0, 1.0, sigma_nm=0.05)          # narrow Gaussian
    spec.add_line(151.0, 1.0, gamma_nm=0.05)          # narrow Lorentzian
    spec.add_line(152.0, 1.0, sigma_nm=0.05, gamma_nm=0.05)  # Voigt
    spec.add_line(153.0, 1.0, sigma_nm=0.0, gamma_nm=0.0)    # delta
    spec.add_line(160.0, 1.0, sigma_nm=5.0)           # wide Gaussian
    print("linear-grid areas (want 5.0):", round(spec.integrate(), 8))

    # logarithmic grid: line at the COARSE end with width >> min step
    # but << local step
    spec_log = Spectrum.logarithmic(100.0, 10000.0, 200)
    lam0 = 9000.0
    dx_local = 9000.0 * ((10000 / 100) ** (1 / 199) - 1)
    print(f"log grid: min step {np.min(np.diff(spec_log.wavelength_nm)):.3f} nm, "
          f"local step at 9000 nm {dx_local:.1f} nm")
    spec_log.add_line(lam0, 1.0, sigma_nm=10.0)   # 10 nm >> min step, << 210 nm local
    print(f"log-grid line area (want 1.0): {spec_log.integrate():.4f}")

    # same on linear grid for reference
    spec_lin = Spectrum.linear(8900.0, 9100.0, 2001)
    spec_lin.add_line(lam0, 1.0, sigma_nm=10.0)
    print(f"linear-grid line area (want 1.0): {spec_lin.integrate():.6f}")

    # Voigt mid branch vs quad branch consistency on a linear grid
    s2 = Spectrum.linear(100.0, 200.0, 50001)
    s2.add_line(150.0, 2.0, sigma_nm=0.005, gamma_nm=0.003)
    print(f"sub-cell Voigt area (want 2.0): {s2.integrate():.8f}")


if __name__ == "__main__":
    main()
