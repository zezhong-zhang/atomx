"""Unresolved Transition Arrays (Bauche-Arnoult formalism).

When an atomic transition array from one configuration to another has
thousands of individual lines (typical for open-d/-f subshell plasma
emission), the individual lines are rarely resolvable and one instead
characterises the *array* by its statistical moments: total strength,
centre-of-gravity wavelength, variance.

This module ships the statistical moments of an explicit line list
(:func:`uta_moments` — exact weighted centroid/variance), plus a
level-list convenience (:func:`uta_simple_from_levels`) that is exact
when the caller supplies per-pair line strengths via ``strengths=`` and
otherwise returns the *Unsöld-limit*, degeneracy-weighted estimate — see
its docstring (review 2026-09-19, finding M5).  Higher moments
(skewness, kurtosis) and the closed-form Bauche-Arnoult expressions in
terms of Slater integrals are not implemented here.

Reference
---------
J. Bauche, C. Bauche-Arnoult, M. Klapisch,
    *Adv. At. Mol. Phys.* **23**, 131 (1988).
C. Bauche-Arnoult, J. Bauche, M. Klapisch,
    *Phys. Rev. A* **20**, 2424 (1979).

Public API
----------
>>> from atomx.uta import uta_moments, uta_simple_from_levels
>>> moments = uta_moments([(800.0, 0.1), (820.0, 0.4)])   # (E_eV, S)
>>> moments.centroid_eV, moments.std_eV
>>> moments = uta_simple_from_levels(
...     [800.0, 820.0], [0.0], strengths={(0, 0): 0.1, (1, 0): 0.4},
... )

Both entry points return :class:`UTAArray`.
"""

from atomx.uta.moments import (
    UTAArray,
    uta_moments,
    uta_simple_from_levels,
)

__all__ = [
    "UTAArray",
    "uta_moments",
    "uta_simple_from_levels",
]
