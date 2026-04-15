"""Post-SCF physical corrections.

Breit interaction: magnetic + retardation corrections to Coulomb.
QED: vacuum polarization (Uehling) + self-energy (Lamb shift).

The self-energy is dispatched through :mod:`atomx.corrections.qedmod`,
which implements a pluggable-backend framework (tabulated Mohr,
Welton heuristic, Flambaum-Ginges and Shabaev model operators).

The flavour-aware :class:`CorrectionStack` composes any subset of these
into a single "apply to any AtomicResult" call; each backend declares
the flavours it supports, and unsupported ones are silently skipped
with provenance captured under ``"_skipped"``.
"""
from atomx.corrections.breit import compute_breit_energy  # noqa: F401
from atomx.corrections.qed import (  # noqa: F401
    compute_qed_corrections,
    compute_se_energy,
    compute_vp_energy,
    uehling_potential,
)
from atomx.corrections import qedmod  # noqa: F401
from atomx.corrections.protocol import Correction, CorrectionResult  # noqa: F401
from atomx.corrections.stack import CorrectionStack  # noqa: F401
from atomx.corrections.adapters import (  # noqa: F401
    BreitCorrection,
    NormalMassShiftCorrection,
    SelfEnergyCorrection,
    UehlingVPCorrection,
    WichmannKrollVPCorrection,
)
