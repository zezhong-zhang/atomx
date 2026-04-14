"""Backward-compatible re-export. Moved to atomx.potentials.nuclear."""
from atomx.potentials.nuclear import (  # noqa: F401
    make_nuclear_potential,
    fermi_grid_step,
    fermi_params,
    point_nucleus_potential,
    fermi_nucleus_potential,
    uniform_sphere_potential,
)
