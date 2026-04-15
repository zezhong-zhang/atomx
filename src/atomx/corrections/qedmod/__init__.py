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
    Flambaum–Ginges (2005) local radiative potential — electric
    pieces (V_l + V_f + simplified WK).  Magnetic form-factor Φ_g
    deferred; use ``"shabaev_local"`` for precision on 1s.
``"shabaev_local"``
    Local part of the Shabaev–Tupitsyn–Yerokhin (2013) model operator,
    ``A_κ · r · exp(−r·c)``, with ``A_κ(Z)`` calibrated against Mohr
    F(αZ).  Port of QEDMOD v7 ``se_pot.f:local_se_pot`` keeping
    atomx's Mohr tabulation as the authoritative reference data.
    Handles κ ∈ {−1, +1, −2}; closes the Ne 1s FG gap (97 % of Mohr
    vs 30 % for FG electric).
``"shabaev"``
    Full Shabaev–Tupitsyn–Yerokhin operator (local + non-local).
    **Stub** — non-local projector term needs Coulomb Green's
    functions and a port of ``fse_dat.f`` tables.

Public API
----------
>>> from atomx.corrections.qedmod import available_backends, get_backend
>>> available_backends()
['mohr', 'welton', 'flambaum_ginges', 'shabaev_local', 'shabaev']
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
