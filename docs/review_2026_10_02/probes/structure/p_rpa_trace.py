import numpy as np
from atomx import Atom
import atomx.methods.rpa as rpa_mod

a2 = Atom(2, config=['1s2', '1s1 2p1'], engine='bspline', corrections='none')
r2 = a2.solve(tol=1e-10).result
bsp = r2._bsp_data
orb = r2.orbitals_dict
eps_map = {(1,-1): r2.orbital_energies['1s1/2'],
           (2,1): r2.orbital_energies['2p1/2'],
           (2,-2): r2.orbital_energies['2p3/2']}

# wrap _pair_v_J to record calls made INSIDE solve_rpa
orig = rpa_mod._pair_v_J
calls = []
def wrapped(*args, **kw):
    v = orig(*args, **kw)
    calls.append((args[2], args[5], args[8], args[11], args[12], v))
    return v
rpa_mod._pair_v_J = wrapped
res = rpa_mod.solve_rpa(orb, bsp, eps_map, [(1,-1)], [(2,1),(2,-2)], twoJ=2, exc_parity=-1)
rpa_mod._pair_v_J = orig
for c in calls:
    print("ka,kb,kr,ks,J -> value:", c)
print("A:\n", res.A)
print("B:\n", res.B)
print("de row:", [eps_map[p]-eps_map[h] for p,h in res.ph_pairs])
