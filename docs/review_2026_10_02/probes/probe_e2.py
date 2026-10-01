import sys
import numpy as np
from sympy import Rational, integrate, S, Symbol
from sympy.physics.wigner import wigner_3j as W3J
from sympy.physics.hydrogen import R_nl
sys.path.insert(0, "/data/code/atomx/src")
from atomx.methods.hydrogenic_reference import hydrogenic_dirac_wavefunction
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL
from atomx.properties.transitions.rates import einstein_A_eL

r = Symbol('r', positive=True)
I_e2 = float(integrate(R_nl(1, 0, r) * r**2 * R_nl(3, 2, r) * r**2, (r, 0, S.Infinity)))
print(f"nonrel E2 radial int <1s|r^2|3d> = {I_e2:.10f}")
omega_nr = 0.5 * (1 - 1 / 9.0)
for tj in (3, 5):
    w = float(W3J(Rational(1, 2), S(2), Rational(tj, 2), Rational(-1, 2), S(0), Rational(1, 2)))
    ang2 = 2 * (tj + 1) * w * w
    S_nr = ang2 * I_e2**2
    A_nr = einstein_A_eL(omega_nr, S_nr, 2, tj)
    print(f"NR anchor d{tj}/2: |ang|^2={ang2:.6f} S={S_nr:.6f} A={A_nr:.6e}")

P1, Q1, bsp1, E1 = hydrogenic_dirac_wavefunction(1, -1, 1, ns=80, ks=9)
for kap in (2, -3):
    P3, Q3, _, E3 = hydrogenic_dirac_wavefunction(3, kap, 1, ns=80, ks=9)
    om = E3 - E1
    ang = angular_eL(-1, kap, 2)
    rad = radial_eL(P1, Q1, P3, Q3, bsp1, 2)
    S = (ang * rad) ** 2
    A = einstein_A_eL(om, S, 2, 2 * abs(kap) - 1)
    print(f"code kap={kap:+d}: ang^2={ang**2:.6f} rad={rad:.10f} S={S:.6f} A={A:.6e}")
