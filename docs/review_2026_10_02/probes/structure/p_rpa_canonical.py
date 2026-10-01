import numpy as np
from atomx import Atom
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.rpa import solve_rpa
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.config.csf import enumerate_jj_csfs_many

# Canonical He 1s2 DHF (bspline, corrections off)
a = Atom(2, config="1s2", engine="bspline", corrections="none")
a.solve(tol=1e-10)
r = a.result
bsp = r._bsp_data
nuc = r.__dict__.get("_nuclear_potential")

# Canonical virtuals from the closed-shell V^N Fock operator
orb = {}
eps = {}
P1s, Q1s, _ = r.orbitals_raw[-1][0]
orb[(1, -1)] = (P1s, Q1s)
eps[(1, -1)] = r.orbital_energies["1s1/2"]
for kap, lab in ((1, "2p1/2"), (-2, "2p3/2")):
    occ, virt = build_virtual_basis(r, kap, potential_mode="VN")
    # lowest virtual = the 2p-like bound pseudostate
    P, Q, e = virt[0]
    orb[(2, kap)] = (P, Q)
    eps[(2, kap)] = e
    print(f"virtual {lab}: eps = {e:.8f}")

csfs = enumerate_jj_csfs_many(["1s2", "1s1 2p1"])
basis = {(sh.n, sh.kappa) for c in csfs for sh in c.subshells}
H = build_ci_matrix(csfs, orb, basis, bsp, 2, bsp["ns"], nuclear_potential=nuc)
E_ref = None
for i, c in enumerate(csfs):
    if c.parity == 1 and c.total_J2 == 0:
        E_ref = H[i, i]
print("E_ref(1s2 on these orbitals) =", E_ref)
odd_J1 = [i for i, c in enumerate(csfs) if c.total_J2 == 2 and c.parity == -1]
order = [i for i in odd_J1]
# map CI csf order to (2p1/2, 2p3/2): check kappa of the 2p in each csf
def pkappa(i):
    for sh in csfs[i].subshells:
        if sh.n == 2: return sh.kappa
    return None
order.sort(key=lambda i: -pkappa(i))  # 2p1/2 (kappa=+1) first
A_exact = H[np.ix_(order, order)] - E_ref * np.eye(len(order))
print("TDA-exact A (canonical orbitals):\n", A_exact)

res = solve_rpa(orb, bsp, eps, [(1, -1)], [(2, 1), (2, -2)], twoJ=2, exc_parity=-1)
print("RPA A:\n", res.A)
print("RPA B:\n", res.B)
print("poles RPA (eV):", res.omega_au * 27.2114)
w_ci = np.linalg.eigvalsh(A_exact)
print("poles TDA-exact (eV):", w_ci * 27.2114)
