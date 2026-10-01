import numpy as np, mpmath as mp
mp.mp.dps = 30

print("=== coulomb_FG vs mpmath ===")
from atomx.scattering.r_matrix.coulomb_fn import coulomb_FG
for (l, eta, rho) in [(0, -1.0, 5.0), (1, -2.0, 8.0), (2, -0.5, 12.0), (0, -5.0, 3.0), (3, -1.5, 20.0)]:
    F, dF, G, dG = coulomb_FG(l, eta, rho)
    Fe = mp.coulombf(l, eta, rho); Ge = mp.coulombg(l, eta, rho)
    print(f"l={l} eta={eta} rho={rho}: F={F:.10f} (exact {float(Fe):.10f}) relerr {abs(F-float(Fe))/abs(float(Fe)):.2e}; "
          f"G={G:.10f} (exact {float(Ge):.10f}) relerr {abs(G-float(Ge))/abs(float(Ge)):.2e}; "
          f"Wronskian F G'-G F' = {F*dG-G*dF:.12f} (exact -1)")

print("\n=== DR Burgess prefactor ===")
from atomx.scattering.dielectronic_recombination.solver import _DR_PREFACTOR_CM3_S_EV_32
# direct SI check: (h^2/(2 pi m_e k_B T))^{3/2} at T=1 eV, in cm^3
h_Js = 6.62607015e-34; me = 9.1093837015e-31; eV_J = 1.602176634e-19
direct = (h_Js**2/(2*np.pi*me*eV_J))**1.5 * 1e6
print(f"module: {_DR_PREFACTOR_CM3_S_EV_32:.6e}, direct SI: {direct:.6e}, ratio: {_DR_PREFACTOR_CM3_S_EV_32/direct:.8f}")

print("\n=== polarisation: J=1 -> J_l=0 E1 ===")
from atomx.scattering.polarisation import alignment_from_msublevel_cross_sections, linear_polarisation
# pure M=0 population
A0 = alignment_from_msublevel_cross_sections([1.0, 0.0], 1.0)
P0 = linear_polarisation(A0, 1.0, 0.0)
# pure M=±1 population
A1 = alignment_from_msublevel_cross_sections([0.0, 1.0], 1.0)
P1 = linear_polarisation(A1, 1.0, 0.0)
# isotropic
Ai = alignment_from_msublevel_cross_sections([1.0, 2.0], 1.0)
Pi = linear_polarisation(Ai, 1.0, 0.0)
print(f"pure M=0: A={A0:.6f}, P(90deg)={P0:.6f}  (expect P=+1, pure pi light)")
print(f"pure M=+-1: A={A1:.6f}, P(90deg)={P1:.6f}  (expect P=-1, pure sigma at 90deg)")
print(f"isotropic: A={Ai:.6f}, P={Pi:.6f} (expect 0)")
