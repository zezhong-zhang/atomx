import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
h = Atom(1, config=["1s1"], corrections="none")
h.solve()
res = h._result
a = res.subshells
b = res.subshells
print("subshells same object across access?", a is b)
print("element identity:", a[0] is b[0])
outer = max(res.subshells, key=lambda s: (s.n, 0, 0))
print("max() obj is loop obj?", any(id(sh) == id(outer) for sh in res.subshells))
