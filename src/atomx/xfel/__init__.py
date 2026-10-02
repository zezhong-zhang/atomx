"""XFEL-pulse physics: sequential-photoionization cascades and
transient-configuration observables (see docs/XFEL_GAP_ANALYSIS.md)."""

from atomx.xfel.cascade import Pulse, RateProvider, cascade

__all__ = ["Pulse", "RateProvider", "cascade"]
