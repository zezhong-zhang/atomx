import numpy as np, mpmath as mp
mp.mp.dps = 40
from atomx.scattering.r_matrix.cc_outer import _closed_channel_log_derivative

# decaying solution in -Z_res/r field: u(r) = W_{nu, l+1/2}(2 kappa r), nu = Z_res/kappa
for (l, kap, a, Z) in [(0, 0.5, 25.0, 1.0), (1, 0.3, 25.0, 1.0), (0, 1.2, 20.0, 1.0), (2, 0.8, 25.0, 2.0)]:
    beta = _closed_channel_log_derivative(l, kap, a, Z_res=Z)
    nu = Z/kap
    x = 2*kap*a
    W = mp.whitw(nu, l+0.5, x)
    dW = mp.diff(lambda t: mp.whitw(nu, l+0.5, t), x)
    beta_exact = 2*kap*dW/W
    print(f"l={l} kappa={kap} a={a} Z={Z}: beta={beta:.8f} exact={float(beta_exact):.8f} relerr={abs(beta-float(beta_exact))/abs(float(beta_exact)):.2e}")
