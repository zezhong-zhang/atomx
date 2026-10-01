import numpy as np
from atomx import Atom
from atomx.methods.ci import ci_solve
from atomx.methods.rpa import solve_rpa

# He on converged DHF orbitals (bspline engine), corrections off
a = Atom(2, config="1s2", engine="bspline", corrections="none")
a.solve(tol=1e-10)
r = a.result
bsp = r._bsp_data

# Same orbitals: exact CI for ['1s2','1s1 2p1'] gives the 1P1/3P1 gaps
a2 = Atom(2, config=['1s2', '1s1 2p1'], engine='bspline', corrections='none')
r2 = a2.solve(tol=1e-10).result
# use the MCDHF result's own orbitals and CSFs for a consistent-orbital CI
from atomx.config.csf import enumerate_jj_csfs_many
csfs = r2.csfs
orb = r2.orbitals_dict
basis = {(sh.n, sh.kappa) for c in csfs for sh in c.subshells}
ci = ci_solve(csfs, orb, basis, bsp, 2, bsp["ns"], nuclear_potential=r2.__dict__.get("_nuclear_potential"))
print("CI states (Ha above ground):")
for k in range(ci.state_energies.size):
    dom = int(np.argmax(np.abs(ci.mixing_coefficients[:, k])))
    print(f"  +{(ci.state_energies[k]-ci.state_energies[0])*27.2114:9.4f} eV  {csfs[dom].label[:44]}")

# RPA/TDA on the same orbitals: J=1 odd channel, occupied 1s, virtuals 2p1/2 2p3/2
eps = {}
for lab, e in r2.orbital_energies.items():
    print("  orb eps:", lab, e)
# orbital_energies keys are labels like '1s1/2','2p1/2','2p3/2'
lab2key = {}
for (n, k) in orb:
    l = k if k > 0 else -k - 1
    j2 = 2*abs(k)-1
    lab = f"{n}{'spdfg'[l]}{j2}/2"
    lab2key[lab] = (n, k)
eps_map = {lab2key[lab]: e for lab, e in r2.orbital_energies.items() if lab in lab2key}
print("eps map:", eps_map)
occ_keys = [(1, -1)]
virt_keys = [(2, 1), (2, -2)]
res = solve_rpa(orb, bsp, eps_map, occ_keys, virt_keys, twoJ=2, exc_parity=-1)
print("RPA omega (eV):", res.omega_au * 27.2114)
print("RPA A matrix:\n", res.A)
print("RPA B matrix:\n", res.B)
print("de:", [eps_map[p]-eps_map[h] for p, h in res.ph_pairs])
