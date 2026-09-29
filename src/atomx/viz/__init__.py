"""Visualisation layer for atomx results.

Three independent backends, each with matching function names so
users can switch with a single import change:

* :mod:`atomx.viz.mpl`    — matplotlib static figures (no extra deps).
* :mod:`atomx.viz.plotly` — interactive HTML (requires ``plotly``).
* :mod:`atomx.viz.manim`  — offline video Scene classes
  (requires ``manim-community``).

The design principle is **result-object decoupling**: every renderer
takes an atomx result (``SCFResult``, ``Waves``, ``Spectrum``,
``HyperfineResult``, …) and produces the corresponding figure.  No
renderer imports solver internals, so a viz bug can never affect a
physics calculation.

Quick start
-----------
>>> from atomx import Atom
>>> from atomx.viz import mpl
>>> ne = Atom(Z=10).solve()
>>> fig = mpl.plot_radial_orbital(ne.waves, label="2p3/2")
>>> fig.savefig("ne_2p.png")

See the individual backend modules for the complete list of rendering
helpers.  The ``manim`` backend additionally exposes *Scene* classes
that subclass ``manim.Scene`` and are run with the manim CLI.
"""

from importlib import import_module as _import_module

__all__ = ['mpl', 'scattering', 'plotly', 'plotly3d', 'manim']


def __getattr__(name):
    if name in __all__:
        module = _import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
