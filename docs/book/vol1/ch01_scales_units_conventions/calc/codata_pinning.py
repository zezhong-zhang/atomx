"""Verify atomx constants against the published CODATA 2022 values.

Reference values below are quoted from the NIST CODATA pages
(physics.nist.gov/cuu, 2022 adjustment; Mohr et al., arXiv:2409.00899),
retrieved 2026-07-16, in concise form value(uncertainty).

The script prints the Verify table for the chapter: atomx value,
published value, relative difference, and the published relative
standard uncertainty u_r.  It also cross-checks the pint registry
(pint 0.25) that atomx.units uses at the API boundary.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/codata_pinning.py
"""

from math import pi

from atomx import constants as c
from atomx.units import ureg

# (name, atomx value, NIST CODATA 2022 value, published u_r, concise form)
NIST = [
    ("alpha", c.ALPHA, 7.2973525643e-3, 1.6e-10,
     "7.297 352 5643(11) e-3"),
    ("a_0 / m", c.BOHR_RADIUS_M, 5.29177210544e-11, 1.6e-10,
     "5.291 772 105 44(82) e-11 m"),
    ("E_h / J", c.HARTREE_J, 4.3597447222060e-18, 1.1e-12,
     "4.359 744 722 2060(48) e-18 J"),
    ("E_h / eV", c.HARTREE_EV, 27.211386245981, 1.1e-12,
     "27.211 386 245 981(30) eV"),
    ("E_h / (hc), m^-1", c.HARTREE_CM1 * 100.0, 2.1947463136314e7, 1.1e-12,
     "2.194 746 313 6314(24) e7 m^-1"),
    ("tau_0 / s", c.AU_TIME_S, 2.4188843265864e-17, 1.1e-12,
     "2.418 884 326 5864(26) e-17 s"),
    ("m_e / kg", c.ELECTRON_MASS_KG, 9.1093837139e-31, 3.1e-10,
     "9.109 383 7139(28) e-31 kg"),
    ("m_p/m_e", c.PROTON_ELECTRON_MASS_RATIO, 1836.152673426, 1.7e-11,
     "1836.152 673 426(32)"),
    ("R_inf / m^-1", c.HARTREE_CM1 * 100.0 / 2.0, 10973731.568157, 1.1e-12,
     "10 973 731.568 157(12) m^-1"),
    ("mu_0 / (N A^-2)", c.CODATA["2022"]["_mu0"], 1.25663706127e-6, 1.6e-10,
     "1.256 637 061 27(20) e-6 N/A^2"),
]

print(f"{'quantity':<18} {'atomx (CODATA 2022 table)':<26} "
      f"{'rel. diff':<10} {'u_r (pub.)':<10}")
worst = 0.0
for name, ours, ref, u_r, concise in NIST:
    rel = abs(ours / ref - 1.0)
    worst = max(worst, rel / u_r if u_r else 0.0)
    print(f"{name:<18} {ours:<26.13e} {rel:<10.1e} {u_r:<10.1e}")
print(f"\nlargest |rel diff| / u_r over the table: {worst:.3f}")

# mu_0 consistency: mu_0 = 2 alpha h / (c e^2) is exact in the 2019 SI
t = c.CODATA["2022"]
mu0_derived = 2 * t["_alpha"] * t["_hplanck"] / (t["_c"] * t["_e"] ** 2)
print(f"mu_0 from 2 alpha h/(c e^2): {mu0_derived!r}"
      f"  (rel vs table {abs(mu0_derived / t['_mu0'] - 1):.1e})")

print("\npint registry cross-check (pint carries its own CODATA table)")
pint_pairs = [
    ("1 hartree -> eV", (1 * ureg.hartree).to("eV").magnitude, c.HARTREE_EV),
    ("1 bohr -> m", (1 * ureg.bohr).to("m").magnitude, c.BOHR_RADIUS_M),
    ("R_inf -> 1/m", (1 * ureg.rydberg_constant).to("1/m").magnitude,
     c.HARTREE_CM1 * 100.0 / 2.0),
]
for name, pv, av in pint_pairs:
    print(f"  {name:<18} pint {pv!r:<24} rel vs atomx {abs(pv / av - 1):.1e}")
