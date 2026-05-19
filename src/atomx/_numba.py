"""Small compatibility layer for optional Numba acceleration."""

from __future__ import annotations

import numpy as np


try:
    import numba as nb  # type: ignore
except ImportError:
    class _NumbaFallback:
        int64 = np.int64

        @staticmethod
        def njit(*args, **_kwargs):
            if args and callable(args[0]):
                return args[0]

            def decorator(func):
                return func

            return decorator

        @staticmethod
        def prange(*args):
            return range(*args)

    nb = _NumbaFallback()


__all__ = ["nb"]
