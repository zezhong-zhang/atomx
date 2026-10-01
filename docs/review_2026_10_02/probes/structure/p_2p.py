import numpy as np
from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.mcdhf.solver import MCDHFSolver
from atomx.constants import ALPHA

def dirac_h(n, kappa, Z=1):
    c2 = 1.0 / ALPHA ** 2
    gamma = float(np.sqrt(kappa * kappa - (Z * ALPHA) ** 2))
    ratio = Z * ALPHA / (n - abs(kappa) + gamma)
    return c2 * ((1.0 + ratio * ratio) ** -0.5 - 1.0)

csfs = []
for cfg in ["1s1", "2s1", "2p1"]:
    csfs.extend(enumerate_jj_csfs(cfg))

for label, kw in [("ns=30 ks=7 no-physics-grid", dict(ns=30, ks=7, use_physics_grid=False)),
                  ("default physics grid", dict())]:
    solver = MCDHFSolver(Z=1, csfs=csfs, **kw)
    res = solver.solve(max_iter=3, scf_tol=1e-4, ci_tol=1e-6)
    print(f"--- {label} ---")
    for k in range(res.n_states):
        dom = int(np.argmax(np.abs(res.mixing_coefficients[:, k])))
        lab = res.csfs[dom].label
        print(f"  state {k}: E={res.state_energies[k]:.10f}  dominant={lab}")
    print("  2p1/2 exact:", dirac_h(2, 1), " 2p3/2 exact:", dirac_h(2, -2))
    print("  orbital_energies:", res.orbital_energies)
