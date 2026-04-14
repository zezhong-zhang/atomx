"""Post-SCF physical corrections.

Breit interaction: magnetic + retardation corrections to Coulomb.
QED: vacuum polarization (Uehling) + self-energy (Lamb shift).

The self-energy is dispatched through :mod:`atomx.corrections.qedmod`,
which implements a pluggable-backend framework (tabulated Mohr,
Welton heuristic, Flambaum-Ginges and Shabaev model operators).
"""
from atomx.corrections.breit import compute_breit_energy  # noqa: F401
from atomx.corrections.qed import (  # noqa: F401
    compute_qed_corrections,
    compute_se_energy,
    compute_vp_energy,
    uehling_potential,
)
from atomx.corrections import qedmod  # noqa: F401
