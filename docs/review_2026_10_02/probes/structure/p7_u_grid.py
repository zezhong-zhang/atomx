import numpy as np
from atomx import Atom, Basis
from atomx.constants import C, ALPHA

Z = 92
exact = C**2 * (np.sqrt(1 - (Z*ALPHA)**2) - 1)
print("exact point Dirac 1s: %.9f Ha" % exact)

for label, cfg, basis in [
    ("string default", '1s1', None),
    ("string refined", '1s1', Basis(grid_hi=0.005, grid_he=0.1, r_max=30.0)),
    ("list default", ['1s1'], None),
    ("list refined", ['1s1'], Basis(grid_hi=0.005, grid_he=0.1, r_max=30.0)),
]:
    a = Atom(Z, config=cfg, charge=Z-1, nuclear='point', corrections='none',
             engine='bspline', **({} if basis is None else {'basis': basis}))
    r = a.solve(tol=1e-9).result
    E = float(getattr(r, 'ground_energy', None) or r.total_energy)
    print(f"{label:16s} E = {E:.9f}  err = {(E-exact)*27.2114:9.4f} eV  converged={r.converged}")
