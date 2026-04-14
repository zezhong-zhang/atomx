"""Atomic potentials: nuclear charge models and starting approximations.

Nuclear models (potentials.nuclear):
    point, Fermi distribution, uniform sphere

Starting potentials:
    TFDA (Thomas-Fermi-Dirac-Amaldi) — fast approximate, no SCF needed
    GSZ (Green-Sellin-Zachor) — parametric effective potential

Note: The self-consistent Hartree (Coulomb) and exchange potentials are
computed dynamically within the SCF loop via bspline.evaluate.compute_yk().
"""
