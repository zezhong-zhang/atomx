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

from atomx import (
    fitting,  # noqa: F401
    lineshapes,  # noqa: F401
    properties,  # noqa: F401  — observables (hyperfine, transitions, isotope_shift, zeeman, …)
    scattering,  # noqa: F401  — photoionization, autoionization, DR, distorted_wave
    spectrum,  # noqa: F401
    units,  # noqa: F401  always importable; pint-optional
    uta,  # noqa: F401
    viz,  # noqa: F401
)
from atomx.atom import Atom
from atomx.compare import compare, theory_ladder, what_changed
from atomx.ladder import Ladder, LadderRun, RungOutcome
from atomx.methods import (
    MCDHF,
    BreitPauli,
    DiracFock,
    HartreeFock,
    HydrogenicDirac,
    HydrogenicSchrodinger,
    Method,
)
from atomx.results import (
    AtomicResult,
    HyperfineResult,
    OrbitalWavefunction,
    Waves,
)
from atomx.sampling import Sampling
from atomx.scattering.models import ContinuumModel, PotentialModel
from atomx.types import (
    Correlation,
    CouplingScheme,
    EnergyUnit,
    Exchange,
    Gauge,
    Multipole,
    NuclearModel,
    RelativisticCorrection,
    StartingPotential,
    TheoryLevel,
)

# Conveniences re-exported at the top level
from atomx.units import codata_version, pint_available, ureg  # noqa: F401
