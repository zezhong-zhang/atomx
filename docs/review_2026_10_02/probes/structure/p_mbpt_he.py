import numpy as np
from atomx import Atom
from atomx.methods.mbpt import mbpt2_energy

# He closed-shell DHF (bspline engine), then MBPT(2)
a = Atom(2, config="1s2", engine="bspline", corrections="none")
a.solve(tol=1e-10)
r = a.result
print("DHF E:", r.total_energy, "converged:", r.converged)
res = mbpt2_energy(r, kappa_max=4)
print("E2 =", res.correlation_energy, "Ha   (lit. MP2/CBS ~ -0.0372; exact corr -0.042044)")
print("E_DHF + E2 =", r.total_energy + res.correlation_energy,
      " (exact nonrel ~ -2.903724)")
print("pair_contribs:", res.pair_contributions)
print("n_virtuals:", res.n_virtuals)
