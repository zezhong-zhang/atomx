import numpy as np
from atomx import Atom, Basis
common = dict(Z=92, nuclear='point', corrections='none', engine='bspline')
refined = Basis(grid_hi=0.005, grid_he=0.10, r_max=30.0)
for label, kw in [("scalar default ", dict(config='1s1')),
                  ("scalar refined ", dict(config='1s1', basis=refined)),
                  ("list   default ", dict(config=['1s1'])),
                  ("list   refined ", dict(config=['1s1'], basis=refined))]:
    a = Atom(**kw, **common).solve()
    oe = a.orbital_energies
    key = sorted(oe)[0]
    ns = a.result._bsp_data['ns'] if hasattr(a.result, '_bsp_data') else '?'
    print(f"{label}: E(1s)={oe[key]:.9f}  total={a.total_energy:.9f} ns={ns} converged={a.converged}")
