"""Explicit radial bound inputs for shared spectator-ion DW ionization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EIIDWBoundChannel:
    """One coherent radial spinor and a separately supplied physical threshold.

    No normalization is imposed on P/Q: a Dyson spinor carries its complete
    signed radial strength. ``occupation`` is a separate prefactor, equal to
    one for that unpolarized CI-removal consumer. This input is neither an
    SCF result nor a solved state; the caller owns state/H/field identity.
    """

    label: str
    kappa: int
    threshold_Ha: float
    occupation: float
    P: np.ndarray
    Q: np.ndarray

    def __post_init__(self):
        if (not isinstance(self.label, str) or not self.label
                or isinstance(self.kappa, (bool, np.bool_))
                or not isinstance(self.kappa, (int, np.integer)) or self.kappa == 0
                or not np.isfinite(self.threshold_Ha) or self.threshold_Ha <= 0
                or not np.isfinite(self.occupation) or self.occupation < 0):
            raise ValueError('explicit channel label/kappa/positive threshold/nonnegative weight required')
        shape = None
        for name in ('P', 'Q'):
            value = np.asarray(getattr(self, name))
            if (value.ndim != 1 or not value.size or not np.isrealobj(value)
                    or not np.isfinite(value).all() or (shape is not None and value.shape != shape)):
                raise ValueError('matching finite real radial coefficient vectors required')
            value = np.asarray(value, dtype=np.float64)
            shape = value.shape
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=value.dtype))
