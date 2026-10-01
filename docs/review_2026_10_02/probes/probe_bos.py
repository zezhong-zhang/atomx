import numpy as np
from atomx import Atom
from atomx.scattering.electron_ionization import _bound_optical_strength
from atomx.numerics.bspline.convol import ConvolCache

for Z, cfg in [(1, "1s1"), (2, "1s2")]:
    a = Atom(Z=Z, config=cfg, engine="bspline", corrections="none")
    a.solve()
    res = a._result
    sh = res.subshells[0]
    P, Q, _ = res.orbitals_raw[sh.kappa][0]
    eps = res.orbital_energies[sh.label]
    cache = ConvolCache(res._bsp_data)
    f = _bound_optical_strength(P, Q, sh.kappa, eps, res, cache, {}, occupation=float(sh.occupation))
    print(f"Z={Z} {cfg}: f_a returned = {f:.4f}  (TRK: total = N_electrons = {float(sh.occupation)})")
