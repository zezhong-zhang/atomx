"""End-to-end B-spline DHF vs DBSR_HF references (exercises convol R^k bank)."""
from atomx import Atom
# DBSR_HF reference (tests/data/dbsr_ne_reference.json): -128.69192584304 (fermi?)
ne = Atom(10, engine='bspline', corrections='none').solve()
print(f"Ne DHF: {ne.total_energy:.11f}  (DBSR ref -128.69192584304)  conv={ne.converged} ns={ne.result._bsp_data['ns']}")
he = Atom(2, engine='bspline', corrections='none').solve()
print(f"He DHF: {he.total_energy:.11f}  (ref ~-2.86181332)  conv={he.converged}")
# Same-kappa orthonormality of the result orbitals
import numpy as np
from atomx.methods.dirac_fock import _max_same_kappa_overlap
print("Ne max same-kappa overlap:", _max_same_kappa_overlap(ne.result.orbitals_raw and {int(k): v for k, v in ne.result.orbitals_raw.items()} if isinstance(ne.result.orbitals_raw, dict) else {}, ne.result._bsp_data))
