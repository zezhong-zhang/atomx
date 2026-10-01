import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.methods.mbpt import build_virtual_basis
from atomx.numerics.bspline.convol import ConvolCache

h = Atom(1, config=["1s1"], corrections="none")
h.solve()
res = h._result
cache = ConvolCache(res._bsp_data)
for mode in ("VN-1", "VN"):
    occ, virt = build_virtual_basis(res, 1, cache=cache, potential_mode=mode)
    ev = np.array([e for _, _, e in virt])
    print(f"{mode}: n_occ={len(occ)} n_virt={len(virt)}")
    print("   first 6 virtual eps:", np.round(ev[:6], 6))
print("hydrogenic np1/2 Dirac energies (approx): 2p -0.1250021, 3p -0.0555564, 4p -0.0312503")
print("1s eps:", res.orbital_energies)
