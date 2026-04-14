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
from atomx.results import (
    AtomicResult,
    HyperfineResult,
    OrbitalWavefunction,
    Wavefunctions,
)
from atomx import autoionization  # noqa: F401
from atomx import fitting  # noqa: F401
from atomx import hyperfine  # noqa: F401
from atomx import isotope  # noqa: F401
from atomx import lineshapes  # noqa: F401
from atomx import photoionization  # noqa: F401
from atomx import spectrum  # noqa: F401
from atomx import uta  # noqa: F401
from atomx import viz  # noqa: F401
from atomx import zeeman  # noqa: F401
from atomx.methods import (
    Method,
    HydrogenicSchrodinger,
    HydrogenicDirac,
    HartreeFock,
    BreitPauli,
    DiracFock,
    MCDHF,
)
from atomx.compare import compare, theory_ladder, what_changed
