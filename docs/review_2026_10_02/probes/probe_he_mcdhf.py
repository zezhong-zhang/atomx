"""F2 verdict probe: He intercombination line via the current MCDHF path."""
import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom

he = Atom(2, config=["1s2", "1s1 2p1"])
he.solve()
print("state energies (Ha):", he.state_energies)
trs = he.transitions(multipoles=["E1"])
print(f"{'I':>3s} {'F':>3s} {'wl(A)':>10s} {'A(s^-1)':>12s} {'S':>10s} {'res':>4s}")
for tr in trs:
    print(f"{tr.initial_label:>3s} {tr.final_label:>3s} {tr.wavelength_angstrom:10.2f} {tr.A_rate_per_s:12.4e} {tr.line_strength:10.4e} {tr.resolution:>4s}")

# Anchor values:
# He I 2 1P1 -> 1 1S0 resonance line 584.3 A: A = 1.799e9 s^-1 (NIST)
# He I 2 3P1 -> 1 1S0 intercombination 591.4 A: A ~ 1.7e-4 s^-1 (NIST/Drake)
print("anchors: 2 1P1->1 1S0 A=1.799e9 s^-1; 2 3P1->1 1S0 A≈1.72e-4 s^-1 (spin-forbidden)")
