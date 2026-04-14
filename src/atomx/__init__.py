"""AtomX: Modern atomic structure calculations from Schrodinger to QED.

A pedagogical and research platform for relativistic atomic structure,
supporting progressive theory levels from non-relativistic Hartree-Fock
through Dirac-Fock to MCDHF with QED corrections.

Quick start:
    from atomx import Atom
    ne = Atom(Z=10)
    ne.solve()
    ne.summary()

Method-based API:
    from atomx import Atom, DiracFock, Exchange
    ne = Atom(Z=10)
    result = ne.solve(DiracFock(breit=True))
    print(result.total_energy)
"""

__version__ = "0.2.0"

from atomx.atom import Atom
from atomx.types import (
    NuclearModel,
    Exchange,
    StartingPotential,
    Correlation,
    Multipole,
    Gauge,
    EnergyUnit,
    CouplingScheme,
    RelativisticCorrection,
    TheoryLevel,
)
from atomx.results import (
    AtomicResult,
    RadialWavefunctions,
    RadialOrbital,
    # Backward-compatible aliases
    Wavefunctions,
    OrbitalWavefunction,
)
from atomx.methods import (
    Method,
    # Preferred names
    NonRelativisticHydrogenic,
    RelativisticHydrogenic,
    HartreeFock,
    BreitPauli,
    DiracFock,
    MCDHF,
    # Backward-compatible aliases
    HydrogenicSchrodinger,
    HydrogenicDirac,
    RelativisticHartreeFock,
)
from atomx.compare import compare, theory_ladder, what_changed
