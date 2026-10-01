import numpy as np
from atomx.scattering.two_electron.coulomb import term_energy_1s_nl, term_energy_1s2

def P_1s(r, Z=2.0):
    r = np.asarray(r); return 2.0*Z**1.5*r*np.exp(-Z*r)
def P_2p(r, Z=2.0):
    r = np.asarray(r); return Z**2.5/np.sqrt(24.0) * r**2 * np.exp(-Z*r/2)

# hydrogenic Z=2 expectation (Bethe-Salpeter): 1P: -2.00351, 3P: -2.02661 Ha
e1 = term_energy_1s_nl(P_1s, P_2p, -2.0, -0.5, 1, 0)
e3 = term_energy_1s_nl(P_1s, P_2p, -2.0, -0.5, 1, 1)
print(f"He 1s2p 1P: {e1:.6f} (hydrogenic-limit -2.00351)")
print(f"He 1s2p 3P: {e3:.6f} (hydrogenic-limit -2.02661)")
e0 = term_energy_1s2(P_1s, -2.0)
print(f"He 1s2 1S: {e0:.6f} (hydrogenic-limit -2.25+5/8*2 = -1.375? no: 2*(-2)+F0(1s1s), F0=5Z/8=1.25 -> -2.75)")
