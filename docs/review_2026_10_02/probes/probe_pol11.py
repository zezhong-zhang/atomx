import sys, inspect
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
h = Atom(1, config=["1s1"], corrections="none")
h.solve()
res = h._result
print(type(res), type(res).__mro__[:3])
attr = inspect.getattr_static(type(res), "subshells", None)
print("subshells static attr:", attr)
