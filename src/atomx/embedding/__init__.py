"""Quantum embedding: multiplet-grade impurity solvers (design E1/E2/E3).

See docs/QUANTUM_EMBEDDING_DESIGN.md. E1 delivers the atomic-limit
(Hubbard-I) Lehmann solver with exact sum-rule anchors and the
``impurity-greens@1`` payload; the atomic eigenproblem backend is an
interface — the v1 reference backend is a self-contained d-shell assembly
(provisional until the CI-native backend lands, tracked as E1.5).
"""
from atomx.embedding.abacus_cf import (  # noqa: F401
    abacus_dblock_to_spherical,
    impurity_from_abacus_dblock,
)
from atomx.embedding.bath import (  # noqa: F401
    ClusterSpec,
    nio_hyperx_configuration,
    solve_cluster_ground,
)
from atomx.embedding.impurity import (  # noqa: F401
    ImpuritySpec,
    occupation,
    solve_atomic_limit,
    spectral_function,
)
from atomx.embedding.io import impurity_greens_payload  # noqa: F401
from atomx.embedding.native_params import dirac_fock_3d_params  # noqa: F401
from atomx.embedding.one_shot import one_shot_spectral  # noqa: F401
