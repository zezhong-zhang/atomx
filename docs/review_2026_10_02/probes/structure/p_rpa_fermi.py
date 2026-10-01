import numpy as np
from atomx.methods.rpa import solve_symmetric_rpa

# Analytic 1x1: A=[[a]], B=[[b]] -> omega = sqrt((a+b)(a-b)), (X+Y)^2 = (a-b)/omega
rng = np.random.default_rng(0)
for a, b in [(3.0, 0.4), (1.2, -0.3), (2.0, 0.9)]:
    A = np.array([[a]]); B = np.array([[b]])
    w, X, Y = solve_symmetric_rpa(A, B)
    exact = np.sqrt((a + b) * (a - b))
    xy2 = (X[0,0] + Y[0,0])**2
    print(f"a={a} b={b}: omega={w[0]:.12f} exact={exact:.12f} "
          f"(X+Y)^2={xy2:.10f} exact={(a-b)/exact:.10f} "
          f"XX-YY={X[0,0]**2 - Y[0,0]**2:.12f}")

# 2x2 against brute-force non-Hermitian diagonalisation cross-check via (A-B)^1/2 route
a1, a2, b1, b2, off = 2.5, 3.1, 0.3, 0.2, 0.05
A = np.array([[a1, off],[off, a2]]); B = np.array([[b1, 0.04],[0.04, b2]])
w, X, Y = solve_symmetric_rpa(A, B)
# verify (A+B)(A-B) eigenvalues = omega^2
M = (A + B) @ (A - B)
ev = np.sort(np.linalg.eigvals(M).real)
print("omega^2:", np.sort(w**2), " vs (A+B)(A-B) eig:", ev)
# residual check on R z = w z with z=(X,Y)
n = 2
R = np.block([[A, B], [-B, -A]])
z = np.vstack([X, Y])
res = R @ z - w[None, :] * z
print("max residual:", np.abs(res).max())

# Fermi potential normalization/continuity: V(r)*r -> -Z at large r, continuous at c
from atomx.potentials.nuclear import _make_fermi_potential, fermi_params
for Z in (2, 26, 82):
    V = _make_fermi_potential(Z)
    a, c = fermi_params(Z)
    r_far = 10.0
    print(f"Z={Z}: V(10)*10 = {V(r_far)*r_far:.12f} (should be -{Z}); "
          f"continuity at c={c:.3e}: {V(c*0.999999):.10f} vs {V(c*1.000001):.10f}; V(0)={V(0.0):.6f}")
