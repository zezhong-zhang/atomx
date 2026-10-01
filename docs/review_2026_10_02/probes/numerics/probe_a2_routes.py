"""F7 wiring: scalar config (DFSolver) vs list config (MCDHF) grid honors."""
import numpy as np
from atomx import Atom, Basis

common = dict(Z=92, nuclear='point', corrections='none')
refined = Basis(grid_hi=0.005, grid_he=0.10, r_max=30.0)

a1 = Atom(config='1s1', **common).solve()
print("scalar default :", a1.orbital_energies)
a2 = Atom(config='1s1', basis=refined, **common).solve()
print("scalar refined :", a2.orbital_energies)
a3 = Atom(config=['1s1'], **common).solve()
print("list   default :", a3.orbital_energies)
a4 = Atom(config=['1s1'], basis=refined, **common).solve()
print("list   refined :", a4.orbital_energies)
