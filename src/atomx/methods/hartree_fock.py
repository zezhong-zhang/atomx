"""Legacy HartreeFock name for the local HF–Slater implementation."""
from atomx.methods.hartree_fock_slater import HartreeFockSlater

HartreeFock = HartreeFockSlater
__all__ = ["HartreeFock", "HartreeFockSlater"]
