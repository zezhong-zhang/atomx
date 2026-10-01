"""Ne TDHF validation using the common-Fock, metric-orthogonal virtual basis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpa_ne_dense_continuum import main

if __name__ == "__main__":
    main()
