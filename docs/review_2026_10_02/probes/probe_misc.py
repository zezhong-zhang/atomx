"""Anchors: recoil operator on H 1s, E2 H 3d->1s vs sympy closed forms."""
import sys
import numpy as np
from sympy import Rational, integrate, exp, oo, S, sqrt as ssqrt
from sympy.physics.wigner import wigner_3j as W3J
sys.path.insert(0, "/data/code/atomx/src")
from atomx.methods.hydrogenic_reference import hydrogenic_dirac_wavefunction
from atomx.corrections.recoil import recoil_radial_matrix
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL
from atomx.properties.transitions.rates import einstein_A_eL
from atomx.constants import ALPHA

# --- recoil operator on H 1s: expect NR limit p^2/2 = 0.5 Ha ---
P, Q, bsp, E = hydrogenic_dirac_wavefunction(1, -1, 1, ns=80, ks=9)
K = recoil_radial_matrix([(P, Q, -1)], bsp, 1, relativistic=False)
Kr = recoil_radial_matrix([(P, Q, -1)], bsp, 1, relativistic=True)
print(f"H 1s recoil NR={K[0,0]:.8f} (expect ~0.5)  rel={Kr[0,0]:.8f}")
# relativistic NMS expectation for H 1s: p^2/2 - Z/c*(...) ~ T + O((Za)^2)
# T_rel for 1s Dirac: <p^2>/2 nonrel = 0.5; Breit corrections add ~ (Za)^2*scale

# --- E2: H 3d_j -> 1s_1/2, compare against exact nonrel closed forms ---
# Radial: I = int R_10(r) r^2 R_32(r) r^2 dr ; R from sympy hydrogen
from sympy.physics.hydrogen import R_nl
r = S('r', positive=True)
I_e2 = float(integrate(R_nl(1, 0, r) * r**2 * R_nl(3, 2, r) * r**2, (r, 0, oo)))
print(f"nonrel E2 radial int <1s|r^2|3d> = {I_e2:.10f}")
# angular |<s1/2||C^2||d_j>|^2 = (2*1/2+1)(2j+1)*3j(1/2 2 j; -1/2 0 1/2)^2
for tj in (3, 5):
    w = float(W3J(Rational(1,2), S(2), Rational(tj,2), Rational(-1,2), S(0), Rational(1,2)))
    ang2 = 2 * (tj + 1) * w * w
    print(f"  |<s1/2||C^2||d{tj}/2>|^2 = {ang2:.6f}")
    S_nr = ang2 * I_e2**2
    A_nr = einstein_A_eL(0.5*(1-1/9.0), S_nr, 2, tj)
    print(f"  NR anchor: S={S_nr:.6f} A={A_nr:.6e} s^-1")
# code:
P1, Q1, bsp1, E1 = hydrogenic_dirac_wavefunction(1, -1, 1, ns=80, ks=9)
for n, kap in ((3, 2), (3, -3)):
    P3, Q3, _, E3 = hydrogenic_dirac_wavefunction(n, kap, 1, ns=80, ks=9)
    om = E3 - E1
    ang = angular_eL(-1, kap, 2)
    rad = radial_eL(P1, Q1, P3, Q3, bsp1, 2)
    S = (ang * rad) ** 2
    A = einstein_A_eL(om, S, 2, 2 * abs(kap) - 1)
    print(f"  code 3d (kap={kap:+d}): ang^2={ang**2:.6f} rad={rad:.10f} S={S:.6f} A={A:.6e}")
