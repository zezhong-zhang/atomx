"""The Rydberg-vs-Hartree bug, live (Exercise 1.2).

The first block is the broken snippet printed in the Ablate section:
it converts a Hartree energy difference with the Rydberg wavenumber
constant.  The second block is the repaired version.  Both outputs are
pasted into the chapter verbatim.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/rydberg_bug.py
"""

# --- broken snippet, exactly as printed in the chapter -------------------
from atomx import Atom, HydrogenicSchrodinger

gs = Atom(Z=1, config="1s1").solve(HydrogenicSchrodinger())
ex = Atom(Z=1, config="2p1").solve(HydrogenicSchrodinger())
dE = ex.orbital_energies["2p"] - gs.orbital_energies["1s"]
sigma = dE * 109737.316           # Rydberg constant, cm^-1
print(f"Lyman-alpha: {1e8 / sigma:.2f} Angstrom")
# --------------------------------------------------------------------------

# --- repaired -------------------------------------------------------------
from atomx.constants import HARTREE_CM1

sigma = dE * HARTREE_CM1          # dE is in Hartree, not Rydberg
print(f"Lyman-alpha: {1e8 / sigma:.2f} Angstrom (repaired)")
print(f"dE = {dE!r} Eh")
