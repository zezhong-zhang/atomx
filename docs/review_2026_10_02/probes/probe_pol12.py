import sys, inspect
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.properties.polarizability import dipole_polarizability

# plain ground-state Atom (no config list) -> which result type?
h = Atom(1, corrections="none")
h.solve()
print("plain Atom(1) result:", type(h._result).__name__)
res = dipole_polarizability(h._result, kappa_max=4)
print(f"  alpha_static = {res.alpha_static:.6f} (expect 4.5 if VN-1 works)")

he = Atom(2, corrections="none")
he.solve()
print("plain Atom(2) result:", type(he._result).__name__)
res2 = dipole_polarizability(he._result, kappa_max=4)
print(f"  He alpha_static = {res2.alpha_static:.6f} (DHF lit ~1.322-1.383)")
