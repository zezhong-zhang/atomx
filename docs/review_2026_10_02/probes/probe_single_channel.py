import numpy as np

# Direct Numerov for u'' + [k^2 - l(l+1)/r^2 - 2 V(r)] u = 0 with H 1s STATIC potential:
# V_stat(r) = -1/r + (1 - (1+r) e^{-2r})/r   (nuclear + direct Hartree of 1s)
# => net = -(1+r) e^{-2r} * 2/r ... compute exactly:
#  Y0(1s,1s; r)/r with P_1s = 2 r e^{-r}:  Y0(r) = int_0^r P^2 ds + r int_r^inf P^2/s ds
# Standard result: V_direct(r) = (1 - (1+r)e^{-2r})/r ; total V = -1/r + V_direct.
def V_total(r):
    r = np.asarray(r, float)
    out = np.empty_like(r)
    big = r > 1e-8
    out[big] = -1.0/r[big] + (1.0 - (1.0+r[big])*np.exp(-2*r[big]))/r[big]
    out[~big] = -1.0  # limit: -1/r + (1-(1+r)(1-2r+2r^2...))/r -> -1/r + (2 - 2r ...); use series -1+O(r)
    return out

def numerov_phase(E, l, r_max=200.0, h=0.005):
    k = np.sqrt(2*E)
    r = np.arange(1e-8, r_max, h)
    f = k*k - l*(l+1)/r**2 - 2.0*V_total(r)
    u = np.zeros_like(r)
    u[0] = 0.0
    u[1] = r[1]**(l+1)
    for i in range(1, len(r)-1):
        u[i+1] = ((2 - (10*h*h/12)*f[i])*u[i] - (1 + (h*h/12)*f[i-1])*u[i-1]) / (1 + (h*h/12)*f[i+1])
    # match to u = F + tan(delta) G at two points: F = rho j_l(rho), G = -rho y_l(rho) (eta=0)
    from scipy.special import spherical_jn, spherical_yn
    r1, r2 = r[-3], r[-2]
    u1, u2 = u[-3], u[-2]
    def FG(rr):
        rho = k*rr
        return rho*spherical_jn(l, rho), -rho*spherical_yn(l, rho)
    F1, G1 = FG(r1); F2, G2 = FG(r2)
    t = (u1*F2 - u2*F1)/(u2*G1 - u1*G2)
    return np.arctan(t)

print("Direct Numerov static(no exchange) e+H(1s) s-wave phase shifts:")
for k in [0.1, 0.2, 0.3]:
    d = numerov_phase(k*k/2, 0)
    print(f"  k={k}: delta_static = {d:.4f} rad")

from atomx.scattering.r_matrix import compute_single_channel
for k in [0.1, 0.2, 0.3]:
    E_eV = k*k/2 * 27.211386245988
    res = compute_single_channel(V_total, np.array([E_eV]), a_au=25.0, n_basis=80, l=0)
    print(f"  k={k}: compute_single_channel delta = {res.phase_shift_rad[0]:.4f} rad")
