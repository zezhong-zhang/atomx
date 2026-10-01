import numpy as np
from atomx import Atom

# QED auto backend on string vs list route (Ne, Z=10 -> mohr)
a1 = Atom(10, config="1s2 2s2 2p6", corrections="breit+qed")
a1.solve()
c1 = a1.result.energy_components
print("Ne string QED:", {k: c1.get(k) for k in ("E_Breit", "E_SE", "E_VP", "E_QED_total", "se_method")})

a2 = Atom(10, config=["1s2 2s2 2p6"], corrections="breit+qed")
a2.solve()
c2 = a2.result.energy_components
print("Ne list   QED:", {k: c2.get(k) for k in ("E_Breit", "E_SE", "E_VP", "E_QED_total", "se_method")})
print("list qed_provenance:", a2.result.qed_provenance)

# Hylleraas-Undheim / superset: Be CSF-space nesting on the SAME orbitals
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.ci import ci_solve

a3 = Atom(4, config=["1s2 2s2", "1s2 2s1 2p1"], corrections="none", engine="bspline")
r3 = a3.solve().result
csfs_all = r3.csfs
orb = r3.orbitals_dict
basis = {(sh.n, sh.kappa) for c in csfs_all for sh in c.subshells}
res_all = ci_solve(csfs_all, orb, basis, r3._bsp_data, 4, r3._bsp_data["ns"],
                   nuclear_potential=r3.__dict__.get("_nuclear_potential"))
res_sub = ci_solve(csfs_all[:1], orb, basis, r3._bsp_data, 4, r3._bsp_data["ns"],
                   nuclear_potential=r3.__dict__.get("_nuclear_potential"))
print("CI superset check: E(single CSF) =", res_sub.ground_energy,
      " E(all) =", res_all.ground_energy,
      " superset lower:", res_all.ground_energy <= res_sub.ground_energy + 1e-12)
# H-U: k-th eigenvalue of the nested space >= k-th of the full space
print("H-U state1: sub has", res_sub.state_energies, "; full:", res_all.state_energies[:3])
