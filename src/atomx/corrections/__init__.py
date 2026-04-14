"""Post-SCF physical corrections.

Breit interaction: magnetic + retardation corrections to Coulomb.
QED: vacuum polarization (Uehling) + self-energy (Lamb shift).
"""
from atomx.corrections.breit import compute_breit_energy  # noqa: F401
from atomx.corrections.qed import compute_qed_corrections  # noqa: F401
