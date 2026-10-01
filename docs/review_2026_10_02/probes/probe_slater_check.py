import mpmath as mp
mp.mp.dps = 40
Z = mp.mpf(2)

def P1s(r): return 2*Z**mp.mpf('1.5')*r*mp.e**(-Z*r)
def P2p(r): return Z**mp.mpf('2.5')/(2*mp.sqrt(6))*r**2*mp.e**(-Z*r/2)

def y_k(f, k, r):
    # y^k(r) = int_0^inf f(s) r_<^k/r_>^{k+1} ds
    i1 = mp.quad(lambda s: f(s)*s**k, [0, r]) / r**(k+1)
    i2 = mp.quad(lambda s: f(s)/s**(k+1), [r, mp.inf]) * r**k
    return i1 + i2

def slater(Pa, Pb, Pc, Pd, k):
    fab = lambda s: Pa(s)*Pb(s)
    return mp.quad(lambda r: Pc(r)*Pd(r)*y_k(fab, k, r), [0, mp.inf])

F0 = slater(P1s, P2p, P1s, P2p, 0)
G1 = slater(P1s, P2p, P2p, P1s, 1)
print(f"Z=2: F0(1s,2p) = {float(F0):.6f}, G1(1s,2p) = {float(G1):.6f}")
print(f"1P = -2.5 + F0 + G1/3 = {float(-2.5+F0+G1/3):.6f}")
print(f"3P = -2.5 + F0 - G1/3 = {float(-2.5+F0-G1/3):.6f}")
F0_11 = slater(P1s, P1s, P1s, P1s, 0)
print(f"F0(1s,1s) = {float(F0_11):.6f} (exact 5Z/8 = 1.25)")
