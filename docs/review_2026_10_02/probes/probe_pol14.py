import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.properties.polarizability import dipole_polarizability

he = Atom(2, config="1s2", engine="bspline", corrections="none")
he.solve()
for mode in ("VN-1", "VN"):
    r = dipole_polarizability(he._result, kappa_max=4, potential_mode=mode)
    print(f"He {mode}: alpha = {r.alpha_static:.6f}")
print("He DHF/SOS literature: ~1.322 (uncorrelated), exact 1.383")

# MCDHF-result He (list config) -- does VN-1 differ from VN there?
he2 = Atom(2, config=["1s2"], engine="bspline", corrections="none")
he2.solve()
print("He list-config result type:", type(he2._result).__name__)
for mode in ("VN-1", "VN"):
    r = dipole_polarizability(he2._result, kappa_max=4, potential_mode=mode)
    print(f"He(list) {mode}: alpha = {r.alpha_static:.6f}")
