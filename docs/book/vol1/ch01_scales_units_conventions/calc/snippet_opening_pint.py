"""Opening-problem code block: one photon through the pint boundary.

Printed verbatim in the chapter opening; output pasted from this run.
"""
from atomx.units import ureg

line = 15.014 * ureg.angstrom          # Fe XVII 3C, Bernitt 2012
for unit in ("eV", "rydberg", "hartree", "cm^-1", "THz"):
    q = line.to(unit, "spectroscopy")
    print(f"{unit:>8}: {q.magnitude:.6f}")
