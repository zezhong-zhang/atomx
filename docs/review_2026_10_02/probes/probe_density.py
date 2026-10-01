import numpy as np
from atomx import Atom
from atomx.numerics.bspline.continuum import (_continuum_spectrum, _continuum_energy_normalization_from_spectrum,
                                              continuum_energy_normalization_factor)

he = Atom(Z=2, config="1s1", charge=1, engine="bspline", corrections="none")
he.solve()
bsp = he._bsp_data
eps, vecs, kp, kq = _continuum_spectrum(1, 2, bsp, None, None)
E = 2.02 - 2.0  # 0.02 Ha above threshold... actually photon 2.02 => E=0.02 Ha
E = 0.02
idx = int(np.argmin(np.abs(eps - E)))
fb = continuum_energy_normalization_factor(eps[idx], bsp)**2
loc = _continuum_energy_normalization_from_spectrum(eps, idx, fallback_density=fb)
k = np.sqrt(2*E*(1+0.5*E/137.036**2))
print(f"box R = {float(bsp['t'][-1]):.1f} a0; target E=0.02 Ha, nearest pseudo E = {eps[idx]:.5f}")
print(f"local spacing density = {loc**2:.3f}; free-box fallback = {fb:.3f}; ratio = {loc**2/fb:.3f}")
print(f"free density R/(pi k) = {float(bsp['t'][-1])/(np.pi*k):.3f}")
# what would the uncapped local density be?
import atomx.numerics.bspline.continuum as C
loc2 = C._continuum_energy_normalization_from_spectrum(eps, idx, fallback_density=None)
print(f"uncapped local density = {loc2**2:.3f}  (true Coulomb near-threshold should exceed free)")
