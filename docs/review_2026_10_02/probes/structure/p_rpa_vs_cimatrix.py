import numpy as np
from atomx import Atom
from atomx.methods.ci import ci_solve
from atomx.methods.rpa import solve_rpa
from atomx.physics.csf.matrix import build_ci_matrix

a2 = Atom(2, config=['1s2', '1s1 2p1'], engine='bspline', corrections='none')
r2 = a2.solve(tol=1e-10).result
bsp = r2._bsp_data
orb = r2.orbitals_dict
nuc = r2.__dict__.get("_nuclear_potential")
csfs = r2.csfs
basis = {(sh.n, sh.kappa) for c in csfs for sh in c.subshells}

H = build_ci_matrix(csfs, orb, basis, bsp, 2, bsp["ns"], nuclear_potential=nuc)
labels = [c.label.split("[")[0].strip() + f" J={c.total_J2}/2" for c in csfs]
# Identify blocks
for i, c in enumerate(csfs):
    print(i, c.total_J2, c.parity, labels[i])
E_ref = H[0, 0]  # 1s2 J=0 even
print("E_ref(1s2) =", E_ref)
# odd J=1 block indices
odd_J1 = [i for i, c in enumerate(csfs) if c.total_J2 == 2 and c.parity == -1]
print("odd J=1 CI block - E_ref (TDA-exact A):")
A_exact = H[np.ix_(odd_J1, odd_J1)] - E_ref * np.eye(len(odd_J1))
print(A_exact)

# Current RPA A on same orbitals
eps_map = {(1,-1): r2.orbital_energies['1s1/2'],
           (2,1): r2.orbital_energies['2p1/2'],
           (2,-2): r2.orbital_energies['2p3/2']}
res = solve_rpa(orb, bsp, eps_map, [(1,-1)], [(2,1),(2,-2)], twoJ=2, exc_parity=-1)
print("RPA A:")
print(res.A)
print("difference (RPA - exact):")
print(res.A - A_exact)
