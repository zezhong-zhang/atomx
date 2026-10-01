"""ci_lines sanity: He 1s2-1s2p pair vs MCDHF CSF path."""
import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom

# closed-shell reference: He ground
ref = Atom(2, config="1s2", corrections="none")  # shooting default
ref.solve()
he = Atom(2, config=["1s2", "1s1 2p1"], corrections="none")
he.solve()
trs = he.transitions(multipoles=["E1"], line_resolution="ci", ci_reference=ref)
for tr in trs:
    print(f"ci: {tr.initial_label:>20s} -> {tr.final_label:>20s} wl={tr.wavelength_angstrom:9.2f} A={tr.A_rate_per_s:.4e} res={tr.resolution}")
