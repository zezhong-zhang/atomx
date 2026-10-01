from atomx import Atom
# The default public path: shooting engine, Breit+QED on
a = Atom(2)  # He ground state, defaults
a.solve()
r = a.result
print("He default route: E =", r.total_energy, "converged:", r.converged, "engine:", r.provenance)
print("components:", r.energy_components)
# DBSR_HF DHF He: -2.86181332; +Breit 6.38e-5; +QED SE(1s)~2*1.115e-4 (Mohr Z=2) +VP
