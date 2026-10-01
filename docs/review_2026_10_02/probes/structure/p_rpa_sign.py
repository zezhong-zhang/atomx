import re as _re
import numpy as np
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.rpa import solve_rpa, reduced_dipole

occ = [(1, -1), (2, -1), (2, 1), (2, -2)]
virt = [(3, -1), (3, 1), (3, -2), (3, 2), (3, -3), (4, -1),
        (4, 1), (4, -2), (4, 2), (4, -3), (4, 3), (4, -4),
        (5, -1), (5, 1), (5, -2)]
subs = [Subshell(n=n, kappa=k, occupation=float(2 * k if k > 0 else -2 * k)) for n, k in occ]
subs += [Subshell(n=n, kappa=k, occupation=1e-6) for n, k in virt]
dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7, starting_potential="tfda")
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
eps = {}
for lab, e in r.orbital_energies.items():
    m = _re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
    n = int(m.group(1)); ll = "spdf".index(m.group(2))
    j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
    eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
occ_keys = [k for k in occ if k in od]
virt_keys = [k for k in virt if k in od and k in eps]
res = solve_rpa(od, dhf.bsp_data, eps, occ_keys, virt_keys, twoJ=2, exc_parity=-1)
A, B = res.A, res.B
print("A symmetric:", np.allclose(A, A.T), " B symmetric:", np.allclose(B, B.T))
print("min eig A-B:", np.linalg.eigvalsh(A - B).min(), " min eig A+B:", np.linalg.eigvalsh(A + B).min())
print("min/max diag A:", A.diagonal().min(), A.diagonal().max())
print("B norm:", np.linalg.norm(B), " max|B|:", np.abs(B).max())
# TDA check (B=0): alpha_TDA vs alpha_df
d = np.array([reduced_dipole(od, dhf.bsp_data, p, h) for p, h in res.ph_pairs])
de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
alpha_df = float(np.sum(2.0/3.0 * d**2 / de))
w_tda, V_tda = np.linalg.eigh(A)
alpha_tda = float(np.sum(2.0/3.0 * (V_tda.T @ d)**2 / w_tda))
print(f"alpha_df = {alpha_df:.4f}  alpha_TDA = {alpha_tda:.4f}  alpha_RPA = ...")
from atomx.methods.rpa import dipole_polarizability
alpha_rpa = dipole_polarizability(res, od, dhf.bsp_data)
print(f"alpha_RPA = {alpha_rpa:.4f}")
# sign of dipole-channel interaction: d^T K d where K = A - diag(de)
K = A - np.diag(de)
print("d^T K d / d^T d:", float(d @ K @ d / (d @ d)))
print("d^T B d / d^T d:", float(d @ B @ d / (d @ d)))
