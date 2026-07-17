"""Generate the table of Hartree atomic-unit scales (Contract C1.1).

Every derived scale is computed from atomx.constants (CODATA 2022) plus
the exact SI defining constants in the same table.  Output is a MyST
markdown table pasted verbatim into the chapter.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/scales_table.py
"""

from math import pi

from atomx import constants as c

t = c.CODATA["2022"]
h, cc, e, kB = t["_hplanck"], t["_c"], t["_e"], t["_k"]  # exact SI
hbar = h / (2.0 * pi)
me = c.ELECTRON_MASS_KG
a0 = c.BOHR_RADIUS_M
Eh = c.HARTREE_J
tau0 = c.AU_TIME_S

rows = [
    ("Length", "$a_0 = \\hbar^2 (4\\pi\\varepsilon_0)/(m_e e^2)$",
     f"{a0:.6e} m = {a0 * 1e10:.9f} Å"),
    ("Energy", "$E_\\mathrm{h} = \\hbar^2/(m_e a_0^2)$",
     f"{Eh:.6e} J = {c.HARTREE_EV:.12f} eV"),
    ("Time", "$\\tau_0 = \\hbar/E_\\mathrm{h}$",
     f"{tau0:.6e} s"),
    ("Velocity", "$v_0 = a_0/\\tau_0 = \\alpha c$",
     f"{a0 / tau0:.6e} m/s = c/{1.0 / (a0 / tau0 / cc):.3f}"),
    ("Momentum", "$p_0 = \\hbar/a_0$",
     f"{hbar / a0:.6e} kg·m/s"),
    ("Frequency", "$E_\\mathrm{h}/h$",
     f"{Eh / h / 1e12:.6f} THz"),
    ("Wavenumber", "$E_\\mathrm{h}/(hc)$",
     f"{c.HARTREE_CM1:.5f} cm⁻¹"),
    ("Electric field", "$E_\\mathrm{h}/(e a_0)$",
     f"{Eh / (e * a0):.6e} V/m"),
    ("Electric dipole", "$e a_0$",
     f"{e * a0:.6e} C·m"),
    ("Magnetic field", "$\\hbar/(e a_0^2)$",
     f"{hbar / (e * a0**2):.6e} T"),
    ("Cross section", "$a_0^2$",
     f"{a0**2:.6e} m² = {a0**2 * 1e22:.4f} Mb"),
    ("Temperature", "$E_\\mathrm{h}/k_B$",
     f"{Eh / kB:.3f} K"),
    ("Rate", "$1/\\tau_0$",
     f"{1.0 / tau0:.6e} s⁻¹"),
]

print("| Quantity | Atomic unit | Value (SI / practical) |")
print("|---|---|---|")
for name, expr, val in rows:
    print(f"| {name} | {expr} | {val} |")

# Consistency checks quoted in the chapter text
print()
print(f"check v0/c = alpha : {(a0 / tau0) / cc!r}  (ALPHA = {c.ALPHA!r})")
print(f"check Eh/h in THz  : {Eh / h / 1e12!r}")
print(f"check a0^2 in Mb   : {a0**2 * 1e22!r}  (1 Mb = 1e-22 m^2)")
