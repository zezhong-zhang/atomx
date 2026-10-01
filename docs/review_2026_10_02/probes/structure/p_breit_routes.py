import numpy as np
from atomx import Atom

DBSR_NE = 0.016643630840490
DBSR_HE = 6.377743e-05

# Ne string (single-config DHF) vs list (MCDHF CI operator)
a1 = Atom(10, config="1s2 2s2 2p6", corrections="breit")
a1.solve()
r1 = a1.result
comps1 = r1.energy_components
print("Ne string:", {k: v for k, v in comps1.items() if isinstance(v, (int, float)) and 'reit' in k})
print("  E_Breit/DBSR =", comps1.get("E_Breit", float('nan')) / DBSR_NE)
print("  breit_provenance:", getattr(r1, '_breit_provenance', None))

a2 = Atom(10, config=["1s2 2s2 2p6"], corrections="breit")
a2.solve()
r2 = a2.result
comps2 = r2.energy_components
print("Ne list:", {k: v for k, v in comps2.items() if isinstance(v, (int, float)) and 'reit' in k})
print("  E_Breit/DBSR =", comps2.get("E_Breit", float('nan')) / DBSR_NE)
print("  breit_provenance:", r2.breit_provenance if hasattr(r2, 'breit_provenance') else None)
dE = (r1.total_energy - r2.ground_energy) * 27.2114
print(f"  string-vs-list total difference: {dE:.4f} eV")

# He list route (Gaunt operator) vs DBSR
a3 = Atom(2, config=["1s2"], corrections="breit")
a3.solve()
r3 = a3.result
comps3 = r3.energy_components
print("He list:", comps3.get("E_Breit"), " ratio to DBSR:", comps3.get("E_Breit", float('nan')) / DBSR_HE)
