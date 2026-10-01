import mpmath as mp
mp.mp.dps = 40
alpha = 1/137.035999084

def R_1s(r): return 2.0*mp.e**(-r)
def R_2s(r): return (1.0/(2.0*mp.sqrt(2.0)))*(2.0 - r)*mp.e**(-r/2.0)

def u_coulomb(E, l, r):
    k = mp.sqrt(2*E); eta = -1.0/k
    return mp.sqrt(2.0/(mp.pi*k)) * mp.coulombf(l, eta, k*r)

def sigma(R_nl, l_f, E, I_n):
    omega = E + I_n
    R = mp.quad(lambda r: (r*R_nl(r)) * u_coulomb(E, l_f, r) * r, [0, mp.inf])
    return (4*mp.pi**2*alpha*omega/3.0) * R**2

for E in [mp.mpf('1e-4'), mp.mpf('0.036749'), mp.mpf('0.5')]:
    print(f"E={float(E):.6f}: sigma_1s = {float(sigma(R_1s,1,E,0.5)):.6e} a0^2")
for E in [mp.mpf('1e-4'), mp.mpf('0.036749')]:
    print(f"E={float(E):.6f}: sigma_2s = {float(sigma(R_2s,1,E,0.125)):.6e} a0^2")
