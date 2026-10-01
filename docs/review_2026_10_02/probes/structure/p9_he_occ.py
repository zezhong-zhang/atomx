import numpy as np
from atomx import Atom

a = Atom(2, config=['1s2', '1s1 2p1'], engine='bspline', corrections='none')
r = a.solve().result
print("converged:", r.converged)
print("occupations:", r.occupations)
print("sum:", sum(r.occupations.values()))
