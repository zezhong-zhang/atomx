import numpy as np
from atomx import Atom
from atomx.methods.ci import ci_solve
from atomx.methods.ci_mbpt import ci_mbpt_solve

# --- sigma1 injection: He-like 1s2 single CSF; shift 1s by +0.1 -> E shifts by +0.2 exactly
a = Atom(2, config=["1s2"], corrections="none", engine="bspline")
r = a.solve().result
bsp = r._bsp_data
nuc = r.__dict__.get("_nuclear_potential")
basis = {(sh.n, sh.kappa) for c in r.csfs for sh in c.subshells}
e0 = ci_solve(r.csfs, r.orbitals_dict, basis, bsp, 2, bsp["ns"], nuclear_potential=nuc).ground_energy
e1 = ci_mbpt_solve(r.csfs, r.orbitals_dict, basis, bsp, 2, bsp["ns"],
                   nuclear_potential=nuc, sigma1={(1, -1): 0.1}).ground_energy
print("sigma1 injection: dE =", e1 - e0, " expected 0.2;  exact:", abs(e1 - e0 - 0.2) < 1e-12)

# sigma2 must raise
try:
    ci_mbpt_solve(r.csfs, r.orbitals_dict, basis, bsp, 2, bsp["ns"], sigma2={"x": 1})
    print("sigma2: NOT raised -- BUG")
except NotImplementedError:
    print("sigma2 raises NotImplementedError: OK")

# --- finite nucleus: Fe H-like 1s, point vs fermi
from atomx.constants import C, ALPHA
Z = 26
exact_pt = C**2 * (np.sqrt(1 - (Z*ALPHA)**2) - 1)
for nuc_model in ("point", "fermi"):
    aa = Atom(Z, config="1s1", charge=Z-1, nuclear=nuc_model, corrections="none", engine="bspline")
    rr = aa.solve(tol=1e-9).result
    print(f"Fe H-like 1s {nuc_model:6s}: E = {rr.total_energy:.6f} Ha (point exact {exact_pt:.6f})")

# --- default-tol MCDHF convergence on the F1 case
b = Atom(2, config=['1s2', '1s1 2s1'], engine='bspline', nuclear='point', corrections='none')
rb = b.solve().result
print("He 2-CSF default tol: converged =", rb.converged, " E =", rb.ground_energy)
diag = rb.orbital_optimization
print("  optimizer residual:", diag.get("orbital_residual"), " termination:", diag.get("termination_reason"))
