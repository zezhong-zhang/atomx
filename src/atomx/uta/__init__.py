"""Unresolved Transition Arrays (Bauche-Arnoult formalism).

When an atomic transition array from one configuration to another has
thousands of individual lines (typical for open-d/-f subshell plasma
emission), the individual lines are rarely resolvable and one instead
characterises the *array* by its statistical moments: total strength,
centre-of-gravity wavelength, variance.

This module ships the leading-order Bauche-Arnoult expressions for

    <σ>                — centroid transition energy
    (Δσ)²              — variance (second central moment)

given the radial Slater integrals and the two configurations.  Higher
moments (third = skewness, fourth = kurtosis) follow but are less
commonly used.

Reference
---------
J. Bauche, C. Bauche-Arnoult, M. Klapisch,
    *Adv. At. Mol. Phys.* **23**, 131 (1988).
C. Bauche-Arnoult, J. Bauche, M. Klapisch,
    *Phys. Rev. A* **20**, 2424 (1979).

Public API
----------
>>> from atomx.uta import uta_moments
>>> moments = uta_moments(
...     config_upper="4d^9 5p",
...     config_lower="4d^10",
...     E_upper_Ha=..., E_lower_Ha=...,
...     slater_Fk=...,
... )
>>> moments.centroid_nm, moments.fwhm_nm

The heavy-lifting closed-form coefficients are implemented in
:mod:`atomx.uta.moments`; callers typically only see :class:`UTAArray`.
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
