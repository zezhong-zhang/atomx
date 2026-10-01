import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom

ref = Atom(2, config="1s2", corrections="none", exchange="gu")
ref.solve()
he = Atom(2, config=["1s2", "1s1 2p1"], corrections="none", exchange="gu")
he.solve()
trs = he.transitions(multipoles=["E1"], line_resolution="ci", ci_reference=ref)
for tr in trs:
    print(f"ci: {tr.initial_label:>22s} -> {tr.final_label:>22s} wl={tr.wavelength_angstrom:9.2f} A={tr.A_rate_per_s:.4e} S={tr.line_strength:.4e}")
