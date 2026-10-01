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
occ, virt = build_virtual_basis(res, 1, cache=cache, potential_mode="VN-1")
print("occ:", [(round(e,6),) for _,_,e in occ])
ev = np.array([e for _,_,e in virt])
print("n_virt:", len(virt), "min:", ev.min(), "negative count:", (ev<0).sum())
print("first 10:", np.round(ev[:10], 6))
