"""Direct hydrogenic-box SOS for alpha(H 1s) to isolate the 7% deficit."""
import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx.properties.transitions.two_photon import _hydrogenic_all_states
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL

for ns, rmax_note in ((60, "ns=60"), (80, "ns=80")):
    E_s, Ps_s, Qs_s, bsp = _hydrogenic_all_states(1, -1, ns=ns, ks=9)
    bound = np.where((E_s < 0))[0]
    bound = bound[np.argsort(E_s[bound])]
    E_1s, P1, Q1 = E_s[bound[0]], Ps_s[bound[0]], Qs_s[bound[0]]
    alpha = 0.0
    for kap in (1, -2):
        E_n, Ps_n, Qs_n, _ = _hydrogenic_all_states(1, kap, ns=ns, ks=9)
        ang = angular_eL(-1, kap, 1)
        mask = E_n > -1.5 * (1.0 / 0.0072973525693) ** 2
        for j in np.where(mask)[0]:
            om = E_n[j] - E_1s
            if om <= 1e-10:
                continue
            R = radial_eL(P1, Q1, Ps_n[j], Qs_n[j], bsp, 1)
            alpha += (2.0 / 3.0) * 0.5 * (ang * R) ** 2 / om
    print(f"{rmax_note}: alpha_box = {alpha:.6f}  (exact 4.5)")
