"""A local-exchange orbital seed must relax to the ordinary DHF solution."""

import pytest

from atomx import Atom, Basis
from atomx.methods.dirac_fock import DiracFock
from atomx.types import Exchange


def test_slater_seeded_dhf_recovers_plain_lithium_energy():
    atom = Atom(Z=3, config="1s2 2s1", engine="bspline", basis=Basis())
    slater = atom.solve(DiracFock(exchange=Exchange.SLATER), max_iter=50)
    plain = atom.solve(DiracFock(), max_iter=50)
    seeded = atom.solve(DiracFock(seed_orbitals=slater.orbitals_raw),
                        max_iter=50)
    assert slater.converged and plain.converged and seeded.converged
    assert seeded.total_energy == pytest.approx(plain.total_energy, abs=1e-8)
