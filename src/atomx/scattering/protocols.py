"""Compatibility imports for shared native state contracts.

Canonical ownership is :mod:`atomx.state_protocols`.
"""
from atomx.state_protocols import (
    AtomicResultLike, SubshellLike, get_orbital, grid_data, list_occupied_orbitals,
)

__all__ = ['AtomicResultLike', 'SubshellLike', 'get_orbital', 'grid_data',
           'list_occupied_orbitals']
