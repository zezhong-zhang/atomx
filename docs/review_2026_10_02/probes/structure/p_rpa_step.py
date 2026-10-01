import numpy as np
from atomx import Atom
from atomx.methods.mbpt import _pair_v_J
from atomx.methods.rpa import ph_pairs_for, solve_rpa
from atomx.numerics.bspline.convol import ConvolCache

a2 = Atom(2, config=['1s2', '1s1 2p1'], engine='bspline', corrections='none')
r2 = a2.solve(tol=1e-10).result
bsp = r2._bsp_data
orb = r2.orbitals_dict
eps_map = {(1,-1): r2.orbital_energies['1s1/2'],
           (2,1): r2.orbital_energies['2p1/2'],
           (2,-2): r2.orbital_energies['2p3/2']}
occ = [(1,-1)]; virt = [(2,1),(2,-2)]
pairs = ph_pairs_for(occ, virt, 2, -1)
print("pairs:", pairs)
res = solve_rpa(orb, bsp, eps_map, occ, virt, twoJ=2, exc_parity=-1)
print("solve_rpa A diag:", res.A.diagonal(), " de:", [eps_map[p]-eps_map[h] for p,h in pairs])

cache = ConvolCache(bsp)
for i, (p, h) in enumerate(pairs):
    Pp, Qp = orb[p]; Ph, Qh = orb[h]
    k = _pair_v_J(Pp, Qp, p[1], Ph, Qh, h[1], Pp, Qp, p[1], Ph, Qh, h[1],
                  2, bsp, cache)
    print(f"pair {i} p={p} h={h}: manual kernel = {k:+.8f}   A[i,i]-de = {res.A[i,i]-(eps_map[p]-eps_map[h]):+.8f}")
