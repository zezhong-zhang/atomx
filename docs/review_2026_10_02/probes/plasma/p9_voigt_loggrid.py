import numpy as np
from atomx.spectrum import Spectrum

# log grid 100..10000 nm, 200 pts: min step 2.34 nm (left edge),
# local step at 9000 nm is ~211 nm.
spec = Spectrum.logarithmic(100.0, 10000.0, 200)
# Voigt with widths > 2*min_step (4.7 nm) but << local step (211 nm):
# takes the pure point-sampled branch -> area NOT conserved.
spec.add_line(9000.0, 1.0, sigma_nm=5.0, gamma_nm=5.0)
print("log-grid narrow Voigt area (want 1.0):", round(spec.integrate(), 6))

# Gaussian with same width is analytic-CDF integrated (conserved):
spec2 = Spectrum.logarithmic(100.0, 10000.0, 200)
spec2.add_line(9000.0, 1.0, sigma_nm=7.07)
print("log-grid narrow Gaussian area (want 1.0):", round(spec2.integrate(), 6))

# Voigt wide enough to be resolved locally is fine:
spec3 = Spectrum.logarithmic(100.0, 10000.0, 200)
spec3.add_line(9000.0, 1.0, sigma_nm=300.0, gamma_nm=300.0)
print("log-grid wide Voigt area (want 1.0):", round(spec3.integrate(), 6))
