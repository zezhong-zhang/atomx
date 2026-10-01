import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.properties.polarizability import dipole_polarizability

h = Atom(1, config="1s1", engine="bspline", corrections="none")
h.solve()
print("Atom(1, config='1s1', bspline):", type(h._result).__name__)
res = dipole_polarizability(h._result, kappa_max=4)
print(f"  alpha_static = {res.alpha_static:.6f} (exact 4.5)")

he = Atom(2, config="1s2", engine="bspline", corrections="none")
he.solve()
print("Atom(2, bspline):", type(he._result).__name__)
res2 = dipole_polarizability(he._result, kappa_max=4)
print(f"  He alpha_static = {res2.alpha_static:.6f} (DHF lit ~1.322)")

# And the list-config form again for contrast
h2 = Atom(1, config=["1s1"], engine="bspline", corrections="none")
h2.solve()
print("Atom(1, config=['1s1']):", type(h2._result).__name__)
res3 = dipole_polarizability(h2._result, kappa_max=4)
print(f"  alpha_static = {res3.alpha_static:.6f} (exact 4.5)")
