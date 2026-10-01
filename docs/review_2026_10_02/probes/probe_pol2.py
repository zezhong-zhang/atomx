import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.properties.polarizability import dipole_polarizability

h = Atom(1, config=["1s1"], corrections="none")
h.solve()
for nv in (20, 40, 60):
    res = dipole_polarizability(h._result, kappa_max=4, n_virt_max=nv)
    print(f"n_virt_max={nv}: alpha={res.alpha_static:.6f}")
res = dipole_polarizability(h._result, kappa_max=4)   # no cap
print(f"uncapped: alpha={res.alpha_static:.6f}")
res = dipole_polarizability(h._result, kappa_max=4, e_virt_max=100.0)
print(f"e_virt_max=100: alpha={res.alpha_static:.6f}")

# how much do p1/2 vs p3/2 channels contribute?
import numpy as np
