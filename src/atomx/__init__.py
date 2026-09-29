"""AtomX: Modern atomic structure calculations from Schrodinger to QED.

Quick start:
    from atomx import Atom
    ne = Atom(Z=10)
    ne.solve()
    ne.summary()

Method-based API:
    from atomx import Atom, DiracFock, Exchange
    result = Atom(Z=10).solve(DiracFock(breit=True))

Self-describing constructor inputs (same physics, grouped by concern):
    from atomx import Atom, Basis, Corrections
    ne = Atom(10, engine="bspline", basis=Basis(ns=200),
              corrections=Corrections(breit=False, qed=False))
"""

__version__ = "0.2.0"

# Public names are resolved on first use. Importing the package does not
# initialize solvers, scattering services, or optional rendering backends.
from importlib import import_module as _import_module

# Keep the public callable stable even when atomx.compare is imported directly.
from atomx.compare import compare, theory_ladder, what_changed

_PUBLIC_MODULES = (
    'fitting', 'lineshapes', 'properties', 'scattering', 'spectrum',
    'units', 'uta', 'viz',
)
_PUBLIC_NAMES = {
    'atom': ('Atom',),
    'atom_inputs': ('Basis', 'Corrections', 'Correlation', 'Environment', 'Exchange'),
    'compare': ('compare', 'theory_ladder', 'what_changed'),
    'ladder': ('Ladder', 'LadderRun', 'RungOutcome'),
    'methods': ('MCDHF', 'BreitPauli', 'DiracFock', 'HartreeFock',
                'HartreeFockSlater', 'HydrogenicDirac', 'HydrogenicSchrodinger', 'Method'),
    'results': ('AtomicResult', 'HyperfineResult', 'OrbitalWavefunction', 'Waves'),
    'sampling': ('Sampling',),
    'calculation_spec': ('CalculationSpec',),
    'numerics.shooting.electron_space': ('ShootingSpace',),
    'scattering.models': ('ContinuumModel', 'PotentialModel'),
    'types': ('CouplingScheme', 'EnergyUnit', 'Gauge', 'Multipole', 'NuclearModel',
              'RelativisticCorrection', 'StartingPotential', 'TheoryLevel'),
    'units': ('codata_version', 'pint_available', 'ureg'),
}
_EXPORTS = {name: module for module, names in _PUBLIC_NAMES.items() for name in names}
__all__ = [*_PUBLIC_MODULES, *_EXPORTS]


def __getattr__(name):
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
