"""Model-operator framework for QED self-energy corrections.

Multiple interchangeable self-energy backends are provided.  All of them
compute the one-electron QED self-energy contribution to the total energy
given a list of converged Dirac-Fock orbitals; they differ in the level of
approximation.

Available backends
------------------
``"mohr"``
    Tabulated Mohr (1983) / Mohr & Kim (1992) F(Zα) values with an
    orbital-overlap screening correction.  Covers 1s1/2, 2s1/2,
    2p1/2, 2p3/2 exactly; other orbitals fall back to hydrogenic
    n^-3 scaling.  Default backend (matches atomx ≤ 0.2 behaviour).
``"welton"``
    Welton (1948) heuristic: ΔE ∝ |ψ(0)|² ln(1/(Zα)²).  Closed-form,
    valid for every j=1/2 orbital (κ = ±1), zero for higher-l.
    Useful as a sanity estimator and for orbitals outside the Mohr table.
``"flambaum_ginges"``
    Flambaum–Ginges (2005) / Ginges–Berengut (2016) local radiative
    potential.  **Stub** — structure in place; formulas to be added.
``"shabaev"``
    Shabaev–Tupitsyn–Yerokhin (2013) model operator (as in the QEDMOD
    code).  Local + non-local pieces, the current state of the art.
    **Stub** — structure in place; parameter tables to be added.

Public API
----------
>>> from atomx.corrections.qedmod import available_backends, get_backend
>>> available_backends()
['mohr', 'welton', 'flambaum_ginges', 'shabaev']
>>> backend = get_backend("mohr")
>>> result = backend.compute(orbitals, bsp_data, Z=10)
>>> result["E_SE"]      # total SE in Hartree
>>> result["per_orbital"]  # dict {label: E_SE_a}
"""

from atomx.corrections.qedmod.base import (
    SE_BACKENDS,
    SEBackend,
    SEResult,
    available_backends,
    get_backend,
    register_backend,
)

# Import side-effects register backends
from atomx.corrections.qedmod import mohr  # noqa: F401
from atomx.corrections.qedmod import welton  # noqa: F401
from atomx.corrections.qedmod import flambaum_ginges  # noqa: F401
from atomx.corrections.qedmod import shabaev  # noqa: F401

__all__ = [
    "SE_BACKENDS",
    "SEBackend",
    "SEResult",
    "available_backends",
    "get_backend",
    "register_backend",
]
