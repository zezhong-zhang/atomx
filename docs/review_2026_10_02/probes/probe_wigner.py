"""Cross-validate atomx wigner 3j/6j/9j + clebsch vs sympy (sampled)."""
import random, sys
from sympy import Rational
from sympy.physics.wigner import wigner_3j as s3j, wigner_6j as s6j, wigner_9j as s9j, clebsch_gordan as scg

sys.path.insert(0, "/data/code/atomx/src")
from atomx.angular.wigner import wigner_3j, wigner_6j, wigner_9j
from atomx.angular.clebsch import clebsch_gordan

# warm up JIT
wigner_3j(1, 1, 2, 1, -1, 0); wigner_6j(1, 1, 2, 1, 1, 2); wigner_9j(1,1,2,1,1,2,2,2,4)
clebsch_gordan(1, 1, 1, 1, 2, 2)

rng = random.Random(20261002)
H = lambda v: Rational(v, 2)

def sample_js(nmax):
    return [H(v) for v in range(0, 2 * nmax + 1)]

maxerr = {k: 0.0 for k in ("3j", "6j", "9j", "cg", "3jL", "6jL")}
worst = {}; ncase = {k: 0 for k in maxerr}

def upd(key, err, args):
    ncase[key] += 1
    if err > maxerr[key]:
        maxerr[key] = err; worst[key] = args

# 3j: dense on j <= 3, sampled m
js = sample_js(3)
for _ in range(40000):
    j1, j2 = rng.choice(js), rng.choice(js)
    lo = abs(j1 - j2); hi = j1 + j2
    if lo > hi: continue
    j3 = lo + Rational(rng.randrange(int(2 * (hi - lo)) + 1), 2)
    m1 = -j1 + Rational(rng.randrange(int(2 * j1) + 1), 1)
    m2 = -j2 + Rational(rng.randrange(int(2 * j2) + 1), 1)
    m3 = -m1 - m2
    if abs(m3) > j3: continue
    ref = float(s3j(j1, j2, j3, m1, m2, m3))
    got = wigner_3j(int(2*j1), int(2*j2), int(2*j3), int(2*m1), int(2*m2), int(2*m3))
    upd("3j", abs(ref - got), (j1, j2, j3, m1, m2, m3))

# also large-j 3j cases to hit the exact path (t_max - t_min > 4)
for _ in range(2000):
    j1 = H(rng.randrange(12, 40)); j2 = H(rng.randrange(12, 40))
    lo, hi = abs(j1 - j2), j1 + j2
    j3 = lo + Rational(rng.randrange(min(int(2*(hi-lo)) + 1, 30)), 2)
    m1 = -j1 + Rational(rng.randrange(int(2*j1) + 1), 1)
    m2 = -j2 + Rational(rng.randrange(int(2*j2) + 1), 1)
    m3 = -m1 - m2
    if abs(m3) > j3: continue
    ref = float(s3j(j1, j2, j3, m1, m2, m3))
    got = wigner_3j(int(2*j1), int(2*j2), int(2*j3), int(2*m1), int(2*m2), int(2*m3))
    upd("3jL", abs(ref - got), (j1, j2, j3, m1, m2, m3))

# 6j sampled
for _ in range(15000):
    args = [rng.choice(sample_js(3)) for _ in range(6)]
    try:
        ref = float(s6j(*args))
    except Exception:
        continue
    got = wigner_6j(*[int(2 * x) for x in args])
    upd("6j", abs(ref - got), tuple(args))

# 6j large-j
for _ in range(1500):
    args = [H(rng.randrange(8, 30)) for _ in range(6)]
    try:
        ref = float(s6j(*args))
    except Exception:
        continue
    got = wigner_6j(*[int(2 * x) for x in args])
    upd("6jL", abs(ref - got), tuple(args))

# 9j sampled (valid-ish rows/cols)
for _ in range(4000):
    args = [rng.choice(sample_js(2)) for _ in range(9)]
    try:
        ref = float(s9j(*args))
    except Exception:
        continue
    got = wigner_9j(*[int(2 * x) for x in args])
    upd("9j", abs(ref - got), tuple(args))

# CG sampled
for _ in range(20000):
    j1, j2 = rng.choice(js), rng.choice(js)
    lo, hi = abs(j1 - j2), j1 + j2
    if lo > hi: continue
    j3 = lo + Rational(rng.randrange(int(2 * (hi - lo)) + 1), 2)
    m1 = -j1 + Rational(rng.randrange(int(2 * j1) + 1), 1)
    m2 = -j2 + Rational(rng.randrange(int(2 * j2) + 1), 1)
    m3 = m1 + m2
    if abs(m3) > j3: continue
    ref = float(scg(j1, j2, j3, m1, m2, m3))
    got = clebsch_gordan(int(2*j1), int(2*m1), int(2*j2), int(2*m2), int(2*j3), int(2*m3))
    upd("cg", abs(ref - got), (j1, j2, j3, m1, m2, m3))

for k in sorted(maxerr):
    print(f"{k:4s}: n={ncase[k]:6d}  max|err|={maxerr[k]:.3e}  worst={worst.get(k)}")
