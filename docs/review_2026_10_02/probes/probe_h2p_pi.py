import numpy as np
import mpmath as mp
mp.mp.dps = 40
alpha = 1/137.035999084

def R_2p(r):  # R_21
    return (1.0/(2.0*mp.sqrt(6.0))) * r * mp.e**(-r/2.0)

def u_coulomb(E, l, r):
    k = mp.sqrt(2*E); eta = -1.0/k
    return mp.sqrt(2.0/(mp.pi*k)) * mp.coulombf(l, eta, k*r)

def sigma_2p(E):
    # 2p -> eps s + eps d. Reduced NR: sigma = (4 pi^2 alpha omega/3) * (1/(2l_i+1)) * ( l_> |<r>|^2 ...)
    # use full m-decomposed: sigma = (4 pi^2 alpha omega/3) * sum_{m_i avg} ... Standard result:
    # sigma(2p) = (4 pi^2 alpha omega / 3) * [ (1/3)|R_{s}|^2 + (2/3)|R_{d}|^2 ]  (length gauge, l_i=1 => /3 averaged into coefficients)
    omega = E + 0.125
    Rs = mp.quad(lambda r: (r*R_2p(r)) * u_coulomb(E, 0, r) * r, [0, mp.inf])
    Rd = mp.quad(lambda r: (r*R_2p(r)) * u_coulomb(E, 2, r) * r, [0, mp.inf])
    return (4*mp.pi**2*alpha*omega/3.0) * ( (1.0/3.0)*Rs**2 + (2.0/3.0)*Rd**2 )

E = 1.0/27.211386245988
sig2p = float(sigma_2p(E))
print(f"exact sigma_2p_PI at 1 eV: {sig2p:.6e} a0^2")
print(f"exact n=2 shell PI total: {sig2p + 0.3050089:.6e} a0^2")

# atomx rydberg path n=2 breakdown
from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering.radiative_recombination import rr_cross_section_rydberg
h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()
rr = rr_cross_section_rydberg(h._result, np.array([1.0]), n_max_per_kappa=8, kappa_max=4)
print("\natomx n=2 per-state sigma_RR (a0^2):")
for lbl, s in sorted(rr.sigma_by_orbital.items()):
    if lbl.startswith("2"):
        print(f"  {lbl:8s} {s[0]:.6e}")
# exact sigma_RR per sublevel: g_n(2j+1) * milne * sigma_PI_subshell/ (2j+1 over l)...
# NR Milne per nl: sigma_RR = g_n * milne * sigma_PI_nl with g_n = 2(2l+1)
milne = (alpha**2) * (E + 0.125)**2 / (E*(2+alpha**2*E))
print("\nexact sigma_RR 2s (g=2):", 2*milne*0.3050089)
print("exact sigma_RR 2p (g=6):", 6*milne*sig2p)
