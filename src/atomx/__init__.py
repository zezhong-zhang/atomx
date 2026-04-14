"""AtomX: Modern atomic structure calculations from Schrodinger to QED.

Quick start:
    from atomx import Atom
    ne = Atom(Z=10)
    ne.solve()
    ne.summary()

Method-based API:
    from atomx import Atom, DiracFock, Exchange
    result = Atom(Z=10).solve(DiracFock(breit=True))
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
from atomx.results import AtomicResult, Wavefunctions, OrbitalWavefunction
from atomx.methods import (
    Method,
    HydrogenicSchrodinger,
    HydrogenicDirac,
    HartreeFock,
    RelativisticHartreeFock,
    DiracFock,
    MCDHF,
)
from atomx.compare import compare, theory_ladder, what_changed
