import time, resource
t0 = time.perf_counter()
import atomx
from atomx import Atom
t_import = time.perf_counter() - t0
print(f"import atomx: {t_import:.2f} s")

def run(label, **kw):
    t0 = time.perf_counter()
    a = Atom(**kw)
    t_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    a.solve()
    t_solve = time.perf_counter() - t0
    print(f"{label}: init {t_init:.2f} s, solve {t_solve:.3f} s, E={a.total_energy:.8f}")
    return t_solve

# cold-ish first solve (numba cache=True should make JIT mostly warm from disk cache)
run("Ne default #1 (cold)", Z=10)
run("Ne default #2 (warm same process)", Z=10)
try:
    run("Ne shooting+gu #1", Z=10, engine="shooting", exchange="gu")
    run("Ne shooting+gu #2", Z=10, engine="shooting", exchange="gu")
except Exception as e:
    print("shooting+gu failed:", type(e).__name__, str(e)[:200])
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MiB")
