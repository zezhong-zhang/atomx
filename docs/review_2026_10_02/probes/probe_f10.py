import time
import numpy as np
from scipy import linalg
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline import continuum as C

ns, ks = 200, 9
data = prepare_bspline_data(make_knots(ns, ks, 0.01, 50.0), ns, ks)

calls = {"eigh": 0}
orig_eigh = linalg.eigh
def counting_eigh(*a, **kw):
    calls["eigh"] += 1
    return orig_eigh(*a, **kw)
linalg.eigh = counting_eigh

energies = np.linspace(0.05, 18.0, 80)  # ~1..500 eV
t0 = time.perf_counter()
out1 = [C.solve_continuum(kappa=-1, energy=e, Z=1, bsp_data=data) for e in energies]
t_first = time.perf_counter() - t0
print(f"80 energies, kappa=-1: {t_first:.3f} s, eigh calls = {calls['eigh']} (old review: 160 for 80 pts on 2 kappas)")

# second kappa channel -> one more eigh
t0 = time.perf_counter()
out2 = [C.solve_continuum(kappa=1, energy=e, Z=1, bsp_data=data) for e in energies]
t_second = time.perf_counter() - t0
print(f"80 energies, kappa=1:  {t_second:.3f} s, cumulative eigh calls = {calls['eigh']}")

# per-hit overhead on warm cache (digest + argmin + scatter)
t0 = time.perf_counter()
for _ in range(200):
    C.solve_continuum(kappa=-1, energy=1.0, Z=1, bsp_data=data)
t_hit = (time.perf_counter() - t0) / 200
print(f"warm cache-hit cost per solve_continuum: {t_hit*1e6:.0f} us")

# how much of that is the blake2b digest? time hashing alone
import hashlib
t0 = time.perf_counter()
for _ in range(200):
    d = hashlib.blake2b(digest_size=20)
    for key in ("ns","nv","ks","ksp","ksq","nsp","nsq","t","gr","grw","bsp_p","bsp_q","bsp_p_d","bsp_q_d"):
        v = data.get(key)
        d.update(key.encode())
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v); d.update(str((a.shape, a.dtype.str)).encode()); d.update(a.tobytes())
        else:
            d.update(repr(v).encode())
t_dig = (time.perf_counter() - t0) / 200
nbytes = sum(data.get(k).nbytes for k in ("t","gr","grw","bsp_p","bsp_q","bsp_p_d","bsp_q_d") if isinstance(data.get(k), np.ndarray))
print(f"digest alone: {t_dig*1e6:.0f} us/call over {nbytes/1e6:.2f} MB of hashed arrays")
print(f"bit-consistency spot: P identical across repeat calls: {np.array_equal(out1[0][0], C.solve_continuum(kappa=-1, energy=energies[0], Z=1, bsp_data=data)[0])}")
