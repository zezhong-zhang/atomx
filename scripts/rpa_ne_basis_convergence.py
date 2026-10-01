"""Sweep radial resolution and energy ceiling of the common-Fock Ne response.

TRK and polarizability must be examined together; experiment also includes
correlation beyond closed-shell TDHF. Each row rebuilds its radial workspace.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpa_ne_dense_continuum import main

if __name__ == "__main__":
    for ns in (40, 60):
        for e_cap in (4.0, 8.0, 16.0):
            print(main(e_cap=e_cap, ns=ns), flush=True)
