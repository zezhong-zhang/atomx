"""Multi-channel close-coupling — placeholder.

The full Phase 2 close-coupling solver (target-state expansion, channel
coupling, Buttle correction, K -> T -> partial-wave excitation cross
sections) is not yet implemented. The Phase 1 deliverable closes the
single-channel scaffold gap and the import surface here; the multichannel
driver will land in a subsequent PR per the HYP-172 phasing.
"""

from __future__ import annotations

CHANNEL_COUPLING_IMPLEMENTED = False


def compute_multichannel(*_args, **_kwargs):
    """Close-coupling multi-channel solve — not implemented yet (Phase 2)."""
    raise NotImplementedError(
        "Multi-channel R-matrix close-coupling is not yet implemented "
        "(HYP-172 Phase 2). The single-channel driver "
        "atomx.scattering.r_matrix.compute_single_channel is available."
    )
