import numpy as np
import mpmath as mp
mp.mp.dps = 40

# Exact nonrelativistic H 2s -> eps p photoionization cross section at electron energy E (a.u.)
# sigma = (4 pi^2 alpha omega / 3) * |<2s|r|E p>|^2   (l_i=0 => /(2l+1)=1, sum over m_l absorbed in reduced form)
# with energy-normalized radial R_E: int R_E R_E' r^2 dr = delta(E-E'); u_E = r R_E, u ~ sqrt(2/(pi k)) sin(k r + ...)

alpha = 1/137.035999084
def R_2s(r):
    return (1.0/(2.0*mp.sqrt(2.0)))*(2.0 - r)*mp.e**(-r/2.0)

def u_coulomb_p(E, r):
    # l=1 Coulomb wave, eta = -1/k; F_l(eta, rho) regular; energy-normalized u = sqrt(2/(pi k)) F
    k = mp.sqrt(2*E)
    eta = -1.0/k
    rho = k*r
    F = mp.coulombf(1, eta, rho)
    return mp.sqrt(2.0/(mp.pi*k)) * F

def sigma_2s(E):
    omega = E + 0.125
    f = lambda r: R_2s(r) * u_coulomb_p(E, r) * r**2 * r  # int R_2s u_E r^3 dr? careful: <2s|r|Ep> radial: int P_2s(r) u_E(r) r dr
    # P_2s = r R_2s; dipole radial integral = int P_2s u_E r dr
    f = lambda r: (r*R_2s(r)) * u_coulomb_p(E, r) * r
    R = mp.quad(f, [0, mp.inf])
    return (4*mp.pi**2*alpha*omega/3.0) * R**2

for E in [1.0/27.211386245988, 0.1, 0.5]:
    print(f"E={E:.6f} Ha  sigma_2s_PI = {float(sigma_2s(E)):.6e} a0^2")

# ---- atomx side: reproduce the Rydberg-path 2s PI
from atomx import Atom
from atomx.constants import HARTREE_EV, BOHR_RADIUS_M
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.scattering.photoionization.direct import photoionization_cross_section
from atomx.scattering.radiative_recombination import _rr_potential_matrices

h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()
res = h._result
bsp = res._bsp_data
occupied = list(res.subshells)
outer = occupied[0]
pp, qq = _rr_potential_matrices(res, outer, None)
ns = bsp["ns"]
hmat, smat = build_dirac_matrices(ns, -1, 1, bsp)
hmat[:ns,:ns] += pp
hmat[ns:,ns:] += qq
basis = solve_dirac(hmat, smat, ns, -1, nsp=bsp.get("nsp"), nsq=bsp.get("nsq"))
bound = [(i, e) for i, e in enumerate(basis.eigenvalues) if e < 0.0]
print("s-channel bound states:", [(i, round(float(e),6)) for i, e in bound[:6]])
i2s, e2s = bound[1]
P, Q, _ = basis.get_orbital(i2s)
print("2s energy:", float(e2s), "exact -0.125")
# norm check
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
gr = bsp["gr"]; grw = bsp["grw"]
nrm = float(np.sum(grw*(eval_P(P,bsp)**2+eval_Q(Q,bsp)**2)))
print("2s norm:", nrm)
Ee_au = np.array([1.0/HARTREE_EV, 0.1, 0.5])
photon = Ee_au + 0.125
sig = photoionization_cross_section(P, Q, -1, float(e2s), photon, 1, bsp, pp, qq)
print("atomx sigma_2s_PI (a0^2):", sig)
print("ratio atomx/exact:", sig / np.array([float(sigma_2s(E)) for E in Ee_au]))
