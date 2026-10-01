import time
from atomx import Atom
def run(label, **kw):
    t0 = time.perf_counter()
    a = Atom(**kw); a.solve()
    print(f"{label}: {time.perf_counter()-t0:.3f} s, E={a.total_energy:.6f}")
run("Fe XVII bspline #1 (cold)", Z=26, charge=16, config="1s2 2s2 2p6")
run("Fe XVII bspline #2 (warm)", Z=26, charge=16, config="1s2 2s2 2p6")
