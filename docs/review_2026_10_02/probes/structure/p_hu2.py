import numpy as np
from atomx import Atom
from atomx.methods.ci import ci_solve

# Be ['1s2 2s2', '1s2 2p2'] — the J=0 even blocks genuinely mix
a = Atom(4, config=["1s2 2s2", "1s2 2p2"], corrections="none", engine="bspline")
r = a.solve().result
csfs = r.csfs
print("csfs:", [(c.label, c.total_J2, c.parity) for c in csfs])
orb = r.orbitals_dict
bsp = r._bsp_data
nuc = r.__dict__.get("_nuclear_potential")
basis = {(sh.n, sh.kappa) for c in csfs for sh in c.subshells}

res_full = ci_solve(csfs, orb, basis, bsp, 4, bsp["ns"], nuclear_potential=nuc)
print("full:", res_full.state_energies[:4])
# Nested: keep only J=0 ground-block CSFs subset
idx0 = [i for i, c in enumerate(csfs) if c.total_J2 == 0 and c.parity == 1]
sub = [csfs[i] for i in idx0]
for cut in range(1, len(sub) + 1):
    res_sub = ci_solve(sub[:cut], orb, basis, bsp, 4, bsp["ns"], nuclear_potential=nuc)
    print(f"nested J=0 first {cut} CSFs:", res_sub.state_energies)
# H-U: with cut growing, every eigenvalue must be non-increasing
