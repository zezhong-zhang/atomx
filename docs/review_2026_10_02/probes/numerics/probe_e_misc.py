"""Probe E: bsplvd derivative columns > 1; find_span edges; shooting high-l/Rydberg."""
import numpy as np
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.basis import bsplvb, bsplvd, find_span

t, ns = make_physics_knots(ks=9, Z=10, nuclear='point')
ks = 9
x = 0.37
left = find_span(t, ns, ks, x)
db = bsplvd(t, ks, x, left, 3)
b = bsplvb(t, ks, x, left)
print("bsplvd col0 == bsplvb:", np.allclose(db[:, 0], b))
print("bsplvd col1 (1st deriv) nonzero:", np.abs(db[:, 1]).max())
print("bsplvd col2 (2nd deriv) max abs:", np.abs(db[:, 2]).max(), " <- zero = never computed")

# numeric check of 1st derivative column
h = 1e-6
bp = bsplvb(t, ks, x + h, find_span(t, ns, ks, x + h))
bm = bsplvb(t, ks, x - h, find_span(t, ns, ks, x - h))
fd = (bp - bm) / (2 * h)
print("1st deriv vs central FD max abs diff:", np.abs(db[:, 1] - fd).max())

# find_span at right boundary
print("find_span(tmax):", find_span(t, ns, ks, t[-1]), "expected", ns - 1)
print("find_span(0):", find_span(t, ns, ks, 0.0), "expected", ks - 1)

# Shooting: hydrogenic Rydberg and high-l states via dbound on a Coulomb spline
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.spline import vint
from atomx.numerics.shooting.dirac_bound import dbound
from atomx.constants import C
Z = 1.0
r, difr, np_ = sgrid(400.0, 1e-5, 0.5, 4000)
rv = -Z * np.ones(np_)  # r*V = -Z
rg, rvg, va, vb, vc, vd, nvt = vint(r, rv, np_)
def dirac_exact(n, kappa):
    gam = np.sqrt(kappa**2 - (Z / C) ** 2)
    return C**2 * (1.0 / np.sqrt(1.0 + (Z / C / (n - abs(kappa) + gam)) ** 2) - 1.0)
for (n, kappa, label) in [(1, -1, "1s"), (6, -1, "6s"), (4, 3, "4f7/2"), (5, -4, "5g7/2"), (8, -1, "8s")]:
    e0 = -Z**2 / (2 * n**2) * 1.05
    e, ier, p, q, ilast = dbound(e0, 1e-12, n, kappa, rg, rvg, va, vb, vc, vd, nvt, r, np_)
    ex = dirac_exact(n, kappa)
    print(f"{label:6s} ier={ier} E={e:.12f} exact={ex:.12f} rel={(e/ex-1):+.2e}")
