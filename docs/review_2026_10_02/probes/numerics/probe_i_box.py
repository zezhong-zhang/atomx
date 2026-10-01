"""Box auto-sizing: shooting resolves 6s via _resolve_r_max; does bspline?"""
import numpy as np
from atomx import Atom, Basis
exact = -1.0 / (2 * 36)  # NR hydrogen 6s; relativistic corr ~ -1e-5
for eng in ("shooting", "bspline"):
    try:
        a = Atom(1, config='6s1', engine=eng, nuclear='point', corrections='none').solve()
        e = list(a.orbital_energies.values())[0]
        print(f"{eng:8s} E(6s) = {e:.9f}  (exact ~{exact:.9f})  rel err {abs(e/exact-1):.2e}  converged={a.converged}")
    except Exception as ex:
        print(f"{eng:8s} raised {type(ex).__name__}: {str(ex)[:140]}")
# with explicit r_max on bspline
a = Atom(1, config='6s1', engine='bspline', nuclear='point', corrections='none',
         basis=Basis(r_max=250.0)).solve()
e = list(a.orbital_energies.values())[0]
print(f"bspline+r_max=250: E(6s) = {e:.9f}  rel err {abs(e/exact-1):.2e}")
