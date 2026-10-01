import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")

# --- H static polarizability = 4.5 a0^3 (Dalgarno) via the SOS engine ---
from atomx import Atom
h = Atom(1, config=["1s1"], corrections="none")
h.solve()
from atomx.properties.polarizability import dipole_polarizability
res = dipole_polarizability(h._result, kappa_max=4, n_virt_max=40)
print(f"H 1s alpha_static = {res.alpha_static:.6f} (exact 4.5)")

# --- two-photon SOS rate vs Drake fit ---
from atomx.properties.transitions.two_photon import (
    two_photon_rate_h_like, two_photon_rate_sos_h_like)
print("Drake fit Z=1:", two_photon_rate_h_like(1), "(lit 8.229 s^-1)")
sos = two_photon_rate_sos_h_like(1, ns=60, ks=9, n_omega=100)
print(f"SOS Z=1: {sos:.4f} s^-1 (expect ~8.23)")

# --- LS allowed terms ---
from atomx.angular.ls_coupling import allowed_terms
print("p2:", allowed_terms(1, 2), "(expect [(1,2),(2,0),(0,0)] = 3P,1D,1S)")
print("d3:", allowed_terms(2, 3), "(expect 2H,2G,4F,2F,2D2,4P,2P)")
print("f2:", allowed_terms(3, 2))
