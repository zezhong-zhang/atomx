"""Compatibility exports; physical static fields live in atomx.potentials."""
from atomx.potentials.static_field import (
    bare_nuclear_potential, static_hartree_potential, tabulated_potential,
)

__all__ = ['bare_nuclear_potential', 'static_hartree_potential', 'tabulated_potential']
