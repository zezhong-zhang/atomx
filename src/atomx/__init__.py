"""AtomX: Modern atomic structure calculations from Schrodinger to QED.

Quick start:
    from atomx import Atom
    ne = Atom(Z=10)
    ne.solve()
    ne.summary()

Method-based API:
    from atomx import Atom, DiracFock, Exchange
    result = Atom(Z=10).solve(DiracFock(breit=True)).result

Self-describing constructor inputs (same physics, grouped by concern):
    from atomx import Atom, Basis, Corrections
    ne = Atom(10, engine="bspline", basis=Basis(ns=200),
              corrections=Corrections(breit=False, qed=False))
"""

__version__ = "0.2.0"

# Public names are resolved on first use. Importing the package does not
# initialize solvers, scattering services, or optional rendering backends.
# ---------------------------------------------------------------------------
# macOS/Accelerate BLAS warning filter
# ---------------------------------------------------------------------------
# Apple's Accelerate ddot/gemv kernels set spurious FP status flags
# (divide-by-zero, overflow, invalid) *inside* the kernel even for clean,
# finite inputs; numpy then reports them at the calling matmul as
# "… encountered in matmul".  Verified 2026-10-01: the trio fires on plain
# random vectors with no zeros, the computed dot is unaffected, and every
# direct input/intermediate at the warned physics sites was finite.  On this
# platform the warning therefore carries zero information, so suppress
# exactly that message pattern — and only when the linked BLAS is
# Accelerate (elsewhere the same warning can be genuine).  Elementwise
# warnings ("in divide", "in sqrt", …) and explicit NaN tripwires are
# untouched.  Set ATOMX_STRICT_FP_WARNINGS=1 to keep even these.
import os as _os
import sys as _sys
import warnings as _warnings
from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

# Keep the public callable stable even when atomx.compare is imported directly.
from atomx.compare import compare, theory_ladder, what_changed

if (_sys.platform == 'darwin'
        and not _os.environ.get('ATOMX_STRICT_FP_WARNINGS')):
    try:
        from numpy import __config__ as _np_config
        _blas_name = str(_np_config.show_config(mode='dicts')[
            'Build Dependencies']['blas']['name']).lower()
    except Exception:
        _blas_name = ''
    if 'accelerate' in _blas_name:
        _warnings.filterwarnings(
            'ignore', message=r'.*encountered in matmul',
            category=RuntimeWarning)

if TYPE_CHECKING:
    # Statically bind the lazy names so type checkers and IDEs resolve the
    # public API (``from atomx import Atom`` etc.) to the real classes instead
    # of the ModuleType inferred from the ``import_module`` calls in
    # ``__getattr__`` below.  Never executed at runtime — the lazy loader is
    # untouched.
    from atomx.atom import Atom
    from atomx.atom_inputs import (
        Basis,
        Corrections,
        Correlation,
        Environment,
        Exchange,
    )
    from atomx.calculation_spec import CalculationSpec
    from atomx.ladder import Ladder, LadderRun, RungOutcome
    from atomx.methods import (
        MCDHF,
        BreitPauli,
        DiracFock,
        HartreeFockSlater,
        HydrogenicDirac,
        HydrogenicSchrodinger,
        Method,
    )
    from atomx.numerics.shooting.adaptive import RadialRefinement
    from atomx.numerics.shooting.electron_space import ShootingSpace
    from atomx.physics.csf.levels import CILevel, CILevelPopulation
    from atomx.physics.csf.state_average import CIChargeStateAverage, CIStateAverage
    from atomx.results import AtomicResult, HyperfineResult, OrbitalWavefunction, Waves
    from atomx.sampling import Sampling
    from atomx.scattering.models import ContinuumModel, PotentialModel
    from atomx.types import (
        CouplingScheme,
        EnergyUnit,
        Gauge,
        Multipole,
        NuclearModel,
        RelativisticCorrection,
        StartingPotential,
        TheoryLevel,
    )
    from atomx.units import codata_version, pint_available, ureg

_PUBLIC_MODULES = (
    'fitting', 'lineshapes', 'properties', 'scattering', 'spectrum',
    'units', 'uta', 'viz',
)
_PUBLIC_NAMES = {
    'atom': ('Atom',),
    'atom_inputs': ('Basis', 'Corrections', 'Correlation', 'Environment', 'Exchange'),
    'compare': ('compare', 'theory_ladder', 'what_changed'),
    'ladder': ('Ladder', 'LadderRun', 'RungOutcome'),
    'methods': ('MCDHF', 'BreitPauli', 'DiracFock',
                'HartreeFockSlater', 'HydrogenicDirac', 'HydrogenicSchrodinger', 'Method'),
    'results': ('AtomicResult', 'HyperfineResult', 'OrbitalWavefunction', 'Waves'),
    'sampling': ('Sampling',),
    'calculation_spec': ('CalculationSpec',),
    'physics.csf.state_average': ('CIStateAverage', 'CIChargeStateAverage'),
    'physics.csf.levels': ('CILevel', 'CILevelPopulation'),
    'numerics.shooting.electron_space': ('ShootingSpace',),
    'numerics.shooting.adaptive': ('RadialRefinement',),
    'scattering.models': ('ContinuumModel', 'PotentialModel'),
    'types': ('CouplingScheme', 'EnergyUnit', 'Gauge', 'Multipole', 'NuclearModel',
              'RelativisticCorrection', 'StartingPotential', 'TheoryLevel'),
    'units': ('codata_version', 'pint_available', 'ureg'),
}
_EXPORTS = {name: module for module, names in _PUBLIC_NAMES.items() for name in names}
__all__ = [*_PUBLIC_MODULES, *_EXPORTS]


def __getattr__(name: str) -> Any:
    # ``Any`` so names not covered by the static bindings above (the
    # subpackages in _PUBLIC_MODULES) don't inherit the ModuleType
    # inferred from ``import_module`` in this body.
    if name in _PUBLIC_MODULES:
        value = _import_module(f'atomx.{name}')
        globals()[name] = value
        return value
    if name in _EXPORTS:
        module_name = _EXPORTS[name]
        module = _import_module(f'atomx.{module_name}')
        # Populate siblings together, including names that collide with a
        # submodule name (notably the public compare function).
        for public_name in _PUBLIC_NAMES[module_name]:
            globals()[public_name] = getattr(module, public_name)
        return globals()[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
