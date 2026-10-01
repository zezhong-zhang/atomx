import numpy as np
from atomx import Atom
from atomx.methods.mbpt import _pair_v_J
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell

a2 = Atom(2, config=['1s2', '1s1 2p1'], engine='bspline', corrections='none')
r2 = a2.solve(tol=1e-10).result
bsp = r2._bsp_data
orb = r2.orbitals_dict
cache = ConvolCache(bsp)
Pp, Qp = orb[(2, 1)]   # 2p1/2
Ph, Qh = orb[(1, -1)]  # 1s1/2

# Code's A diagonal kernel (as assembled in solve_rpa): <p h; J=1|v|p h; J=1>_A
A_code = _pair_v_J(Pp, Qp, 1, Ph, Qh, -1, Pp, Qp, 1, Ph, Qh, -1, 2, bsp, cache)
print("code A kernel (2p1/2,1s) J=1:", A_code)

# Radial anchors: F0(2p,1s) and G1(1s,2p)
F0 = compute_slater_rk_cell(Pp, Qp, Pp, Qp, Ph, Qh, Ph, Qh, bsp, 0, cache=cache)
G1 = compute_slater_rk_cell(Pp, Qp, Ph, Qh, Ph, Qh, Pp, Qp, bsp, 1, cache=cache)
print("F0(2p,1s) =", F0, " G1(1s,2p) =", G1)
# PP-coupled <p h; J|v|p h; J>_A for jp=1/2, jh=1/2, J=1:
# V_J(abab) built from F^k(a,a... via X^k(abab)=<a||C^k||a><b||C^k||b> R^k(abab)
# Correct Thouless ph kernel (singlet-ish J=1 channel) should be ~ -G1/3 or +2G1/3 scale:
print("G1/3 =", G1/3, " 2G1/3 =", 2*G1/3)
