import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.methods.mbpt import _build_closed_shell_fock
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
from atomx.numerics.bspline.convol import ConvolCache

h = Atom(1, config=["1s1"], corrections="none")
h.solve()
res = h._result
bsp = res._bsp_data
ns_full = bsp["ns"]

print("subshells:", [(s.label, s.n, s.kappa, s.occupation) for s in res.subshells])
print("_nuclear_potential:", getattr(res, "_nuclear_potential", "MISSING"))
print("provenance keys:", list(getattr(res, "provenance", {}).keys())[:20])
npf = getattr(res, "_nuclear_potential", None) or res.provenance.get("nuclear_potential_func")
print("nuclear_potential_func resolved:", npf)

H3, S3 = build_dirac_matrices(ns_full, 1, 1, bsp, npf)
H2, _ = build_dirac_matrices(ns_full, 1, 1, bsp, None)
print("H0(npf) vs H0(None) diff:", np.max(np.abs(H3 - H2)))
H1, _ = _build_closed_shell_fock(res, 1, cache=ConvolCache(bsp), potential_mode="VN-1")
print("VN-1 vs H0(npf) diff:", np.max(np.abs(H1 - H3)))
print("orbitals_raw keys:", {k: len(v) for k, v in res.orbitals_raw.items()})
