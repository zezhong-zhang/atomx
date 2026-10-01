import numpy as np
from atomx import Atom
from atomx.methods.mbpt import build_virtual_basis, _pair_v_J
from atomx.methods.rpa import solve_rpa
from atomx.numerics.bspline.convol import ConvolCache

a = Atom(2, config="1s2", engine="bspline", corrections="none")
a.solve(tol=1e-10)
r = a.result
bsp = r._bsp_data
orb = {(1, -1): r.orbitals_raw[-1][0][:2]}
eps = {(1, -1): r.orbital_energies["1s1/2"]}
for kap in (1, -2):
    _, virt = build_virtual_basis(r, kap, potential_mode="VN")
    orb[(2, kap)] = virt[0][:2]
    eps[(2, kap)] = virt[0][2]
res = solve_rpa(orb, bsp, eps, [(1, -1)], [(2, 1), (2, -2)], twoJ=2, exc_parity=-1)
print("B matrix:\n", res.B)

# pp-kernel reference for the diagonal B element: <(2p1/2)^2 J=1|v|(1s)^2 J=1>_A
cache = ConvolCache(bsp)
Pp, Qp = orb[(2, 1)]
Ph, Qh = orb[(1, -1)]
v_pp = _pair_v_J(Pp, Qp, 1, Pp, Qp, 1, Ph, Qh, -1, Ph, Qh, -1, 2, bsp, cache)
print("pp-kernel <(2p1/2)^2 J=1|v|(1s1/2)^2 J=1>_A =", v_pp, "(Pauli: must be 0)")
print("RPA B[0,0] =", res.B[0, 0])
