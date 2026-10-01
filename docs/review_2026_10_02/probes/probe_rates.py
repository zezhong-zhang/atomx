"""Anchor tests: H 2p->1s E1 A-rate (exact), length/velocity gauge agreement
on exact Dirac eigenfunctions, H static polarizability (Dalgarno 4.5)."""
import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx.methods.hydrogenic_reference import hydrogenic_dirac_wavefunction
from atomx.properties.transitions.angular import angular_eL, angular_mL
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity, radial_mL
from atomx.properties.transitions.rates import einstein_A_eL, einstein_A_mL
from atomx.constants import ALPHA

# --- H 2p_j -> 1s_1/2 E1 ---
P1, Q1, bsp, E1s = hydrogenic_dirac_wavefunction(1, -1, 1, ns=80, ks=9)
for n, kap, tag in ((2, 1, "2p1/2"), (2, -2, "2p3/2")):
    P2, Q2, _, E2 = hydrogenic_dirac_wavefunction(n, kap, 1, ns=80, ks=9)
    omega = E2 - E1s
    tj_f = 2 * abs(kap) - 1          # upper state 2j
    ang = angular_eL(-1, kap, 1)
    rad = radial_eL(P1, Q1, P2, Q2, bsp, 1)   # static length gauge
    S = (ang * rad) ** 2
    A = einstein_A_eL(omega, S, 1, tj_f)
    # nonrel line strengths: S(2p1/2)=1.1097, S(2p3/2)=2.2194 (2/3, 4/3 of R^2), R^2=1.6646
    print(f"{tag}->1s1/2: omega={omega:.8f} Ha  S={S:.6f}  A={A:.6e} s^-1")
print("exact nonrel anchor: A(2p->1s) = 6.2649e8 s^-1; S(1/2)=1.1097, S(3/2)=2.2195")

# --- gauge agreement on exact eigenfunctions ---
for n, kap, tag in ((2, 1, "2p1/2"), (2, -2, "2p3/2"), (3, 1, "3p1/2")):
    P2, Q2, _, E2 = hydrogenic_dirac_wavefunction(n, kap, 1, ns=80, ks=9)
    omega = E2 - E1s
    rl = radial_eL(P1, Q1, P2, Q2, bsp, 1, kappa_i=-1, kappa_f=kap, omega=omega)
    rv = radial_eL_velocity(P1, Q1, P2, Q2, bsp, 1, kappa_i=-1, kappa_f=kap, omega=omega)
    print(f"1s->{tag}: length={rl:.10e}  velocity={rv:.10e}  ratio={rv/rl:.8f}")

# --- M1 2s->1s vs Drake 2.496e-6 * Z^10 ---
for Z in (1, 2, 5, 10):
    Pa, Qa, bspa, Ea = hydrogenic_dirac_wavefunction(1, -1, Z, ns=80, ks=9)
    Pb, Qb, _, Eb = hydrogenic_dirac_wavefunction(2, -1, Z, ns=80, ks=9)
    om = Eb - Ea
    ang = angular_mL(-1, -1, 1)
    rad = radial_mL(Pa, Qa, Pb, Qb, bspa, 1, omega=om)
    S = (ang * rad) ** 2
    A = einstein_A_mL(om, S, 1, 1)  # tj_upper=1 (2s1/2)
    print(f"M1 2s->1s Z={Z}: A={A:.6e} s^-1  Drake={2.496e-6*Z**10:.6e}  ratio={A/(2.496e-6*Z**10):.4f}")
