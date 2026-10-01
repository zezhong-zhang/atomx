import numpy as np
from atomx import Atom
from atomx.constants import HARTREE_CM1

# He-like C V: 1s2 1S0 ground, 1s2p 3P1/3P2 fine structure; NIST 3P1-3P2 ~ 123.29 cm-1
# (breit_ci docstring cites the scalar route leaving DC value 13.74 cm-1)
for breit in ("none", "breit"):
    a = Atom(6, charge=4, config=['1s2', '1s1 2p1'], corrections=breit, engine='bspline')
    r = a.solve().result
    lt = r.level_table
    # find the two odd-parity J=1 and J=2 states (2p fine structure)
    odd = [row for row in lt if row["parity"] == -1]
    for row in odd:
        print(f"breit={breit:6s} 2J={row['twoJ']} dE = {row['dE_eV']*8065.54:.3f} cm-1  cfg={row['config'][:40]}")
    j1 = [row for row in odd if row["twoJ"] == 2]
    j2 = [row for row in odd if row["twoJ"] == 4]
    if j1 and j2:
        fs = (j2[0]["E_hartree"] - j1[0]["E_hartree"]) * HARTREE_CM1
        print(f"breit={breit:6s} 3P2-3P1 fine structure = {fs:.2f} cm-1 (NIST ~123.29)")
