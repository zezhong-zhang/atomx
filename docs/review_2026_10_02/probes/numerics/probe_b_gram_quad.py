"""Probe B: Gram matrix conditioning vs ns; Galerkin quadrature accuracy."""
import numpy as np
from scipy import linalg
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.basis import prepare_bspline_data, setup_gauss_points, evaluate_bsplines_on_grid
from atomx.numerics.bspline.integrals import overlap_matrix, nuclear_matrix

print("== Gram (overlap) conditioning, default physics grid, ks=9 ==")
for Z in (1, 10, 36, 92):
    t, ns = make_physics_knots(ks=9, Z=Z, nuclear='point')
    data = prepare_bspline_data(t, ns, 9, kinetic_balance=True)
    bsp_p, grw = data["bsp_p"], data["grw"]
    S = overlap_matrix(data["nv"], data["ksp"], data["nsp"], bsp_p, grw)
    w = linalg.eigvalsh(S)
    print(f"Z={Z:3d} ns={ns:4d} cond(S_P)={w[-1]/w[0]:.3e}  min_eig={w[0]:.3e}")

print("\n== Gram conditioning on uniform-exponential grid, ns grows (ks=9) ==")
from atomx.numerics.bspline.knots import make_exponential_knots
for ns in (50, 113, 200, 400, 800):
    t = make_exponential_knots(ns, 9, 1e-3, 100.0)
    data = prepare_bspline_data(t, ns, 9, kinetic_balance=False)
    S = overlap_matrix(data["nv"], 9, ns, data["bsp_p"], data["grw"])
    w = linalg.eigvalsh(S)
    print(f"ns={ns:4d} cond(S)={w[-1]/w[0]:.3e} min_eig={w[0]:.3e}")

print("\n== Quadrature: nuclear matrix <Bi|1/r|Bj>, Gauss-ks vs Gauss-48 reference ==")
Z = 92
t, ns = make_physics_knots(ks=9, Z=Z, nuclear='point')
data = prepare_bspline_data(t, ns, 9, kinetic_balance=True)
V9 = nuclear_matrix(data["nv"], data["ksp"], data["nsp"], data["bsp_p"], data["gr"], data["grw"], Z, None)
# reference: recompute with 48-pt Gauss per interval
gr48, grw48 = setup_gauss_points(t, ns, 9, n_gauss=48)
bsp48 = evaluate_bsplines_on_grid(data["tp"], data["nsp"], data["ksp"], gr48, nderiv=1)[:, :, :, 0]
V48 = nuclear_matrix(data["nv"], data["ksp"], data["nsp"], bsp48, gr48, grw48, Z, None)
d = np.abs(V9 - V48)
scale = np.abs(V48).max()
i = np.unravel_index(np.argmax(d), d.shape)
print(f"Z=92: max|V9-V48|={d.max():.3e} at {i}, max|V48|={scale:.3e}, rel={d.max()/scale:.3e}")
# worst RELATIVE entry among large entries
mask = np.abs(V48) > 1e-3*scale
print(f"max relative entry diff (entries >1e-3 max): {(d[mask]/np.abs(V48[mask])).max():.3e}")

# same for overlap (polynomial: expect machine exact)
S48 = overlap_matrix(data["nv"], data["ksp"], data["nsp"], bsp48, grw48)
S9 = overlap_matrix(data["nv"], data["ksp"], data["nsp"], data["bsp_p"], data["grw"])
print(f"overlap: max|S9-S48|={np.abs(S9-S48).max():.3e} (machine eps scale)")
