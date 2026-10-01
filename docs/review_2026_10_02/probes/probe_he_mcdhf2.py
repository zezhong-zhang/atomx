"""F2 verdict probe, matching the 2026-09-23 review setup: point nucleus,
corrections off, configs ['1s2', '1s1 2p1']."""
import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom

he = Atom(2, config=["1s2", "1s1 2p1"], corrections="none")
he.solve()
print("state energies:", he.state_energies)
mcdhf = he._csf_result
print("csfs:", [getattr(c, "label", "?") for c in mcdhf.csfs])
print("mixing:\n", np.array2string(mcdhf.mixing_coefficients, precision=4, suppress_small=True))
trs = he.transitions(multipoles=["E1"])
for tr in trs:
    print(f"{tr.initial_label:>3s} {tr.final_label:>3s} {tr.wavelength_angstrom:10.2f} A={tr.A_rate_per_s:12.4e} S={tr.line_strength:10.4e}")
print("review independent-operator refs: triplet A=10.79169 s^-1, singlet A=2.6512242e9 s^-1")
print("physical He 2^3P1 A ~ 50.85 s^-1 (Jacobs/Drake)")
