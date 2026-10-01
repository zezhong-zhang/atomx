import numpy as np
from atomx import Atom
from atomx.methods.mbpt import _pair_v_J
from atomx.numerics.bspline.convol import ConvolCache

a2 = Atom(2, config=['1s2', '1s1 2p1'], engine='bspline', corrections='none')
r2 = a2.solve(tol=1e-10).result
bsp = r2._bsp_data
orb = r2.orbitals_dict
cache = ConvolCache(bsp)
Pp, Qp = orb[(2, 1)]
Ph, Qh = orb[(1, -1)]

# scalar path
v_scalar = _pair_v_J(Pp, Qp, 1, Ph, Qh, -1, Pp, Qp, 1, Ph, Qh, -1, 2, bsp, cache)
# bilinear path as in solve_rpa
bl = {}
v_bilin = _pair_v_J(Pp, Qp, 1, Ph, Qh, -1, Pp, Qp, 1, Ph, Qh, -1, 2, bsp, cache, rk_bilinear=bl)
print("scalar:", v_scalar, " bilinear:", v_bilin)

# and the B kernel path for the same pair: <p p; J|v|h h; J>_A / (1+1)... (p=p2, h=h2 -> norm 2)
bl2 = {}
vB = _pair_v_J(Pp, Qp, 1, Pp, Qp, 1, Ph, Qh, -1, Ph, Qh, -1, 2, bsp, cache)
print("B raw <pp|v|hh>_A:", vB, " -> /2 =", vB/2)
