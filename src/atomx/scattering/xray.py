"""Nonresonant atomic X-ray scattering in the independent-orbital model.

Momentum transfer is in inverse Bohr, radial coordinates in Bohr, and
differential cross sections in barn/sr. The incoherent function uses the
XATOM configuration-average Waller–Hartree prescription: within one n-l
shell the occupied-state projector has weight q/g, between shells the
weight is the product of occupation fractions. This is not a Compton
energy spectrum or an anomalous (f1/f2) dispersion calculation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import spherical_jn

from atomx.angular.coefficients import reduced_C
from atomx.constants import ALPHA, BOHR_RADIUS_M, HARTREE_EV

_RE2_BARN = (ALPHA**2 * BOHR_RADIUS_M) ** 2 / 1e-28


@dataclass(frozen=True)
class ScatteringOrbital:
    """Real radial Dirac functions on the common quadrature grid."""

    n: int
    kappa: int
    occupation: float
    P: np.ndarray
    Q: np.ndarray

    @property
    def l(self):  # noqa: E743 -- orbital angular momentum
        return self.kappa if self.kappa > 0 else -self.kappa - 1


class AtomicScattering:
    """Bound-orbital f0(q) and configuration-average incoherent S(q).

    Supply normalized orbitals that are mutually orthogonal within each
    kappa, including all occupied jj partners with the same q/g fraction
    within each n-l shell. No normalization or positivity clipping is
    applied to hide radial integration errors.
    """

    def __init__(self, r, weights, orbitals):
        if np.iscomplexobj(r) or np.iscomplexobj(weights):
            raise ValueError("The radial quadrature must be real")
        self.r = np.array(r, dtype=float, copy=True).ravel()
        self.weights = np.array(weights, dtype=float, copy=True).ravel()
        orbitals = tuple(orbitals)
        if any(np.iscomplexobj(a.P) or np.iscomplexobj(a.Q) for a in orbitals):
            raise ValueError("ScatteringOrbital requires real radial functions")
        # Cached overlaps must not become stale if the solver's arrays change.
        self.orbitals = tuple(
            ScatteringOrbital(
                a.n,
                a.kappa,
                a.occupation,
                np.array(a.P, dtype=float, copy=True),
                np.array(a.Q, dtype=float, copy=True),
            )
            for a in orbitals
        )
        if (
            self.r.size < 2
            or self.weights.shape != self.r.shape
            or not np.isfinite(self.r).all()
            or not np.isfinite(self.weights).all()
            or np.any(self.r < 0)
            or np.any(np.diff(self.r) <= 0)
            or np.any(self.weights <= 0)
        ):
            raise ValueError(
                "Require increasing nonnegative r and positive quadrature weights"
            )
        shells = {}
        keys = set()
        for a in self.orbitals:
            if (
                not isinstance(a.n, (int, np.integer))
                or not isinstance(a.kappa, (int, np.integer))
                or a.n < 1
                or a.kappa == 0
                or a.l >= a.n
                or not np.isfinite(a.occupation)
                or not 0 <= a.occupation <= 2 * abs(a.kappa)
            ):
                raise ValueError("Invalid orbital quantum numbers or occupation")
            if (a.n, a.kappa) in keys:
                raise ValueError("Duplicate orbital")
            keys.add((a.n, a.kappa))
            for values in (a.P, a.Q):
                if (
                    np.asarray(values).shape != self.r.shape
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Orbital values must match the finite radial grid")
            fraction = a.occupation / (2 * abs(a.kappa))
            group = shells.setdefault((a.n, a.l), [])
            if group and not np.isclose(fraction, group[0][1], atol=1e-12, rtol=0):
                raise ValueError(
                    "Waller–Hartree n-l average requires equal jj occupation fractions"
                )
            group.append((a.kappa, fraction))
        for (_, l), group in shells.items():
            expected = {-(l + 1), l} if l else {-1}
            if {k for k, _ in group} != expected:
                raise ValueError("Include both jj partners of each occupied n-l shell")
        self.electrons = sum(a.occupation for a in self.orbitals)
        for values in (
            self.r,
            self.weights,
            *(v for a in self.orbitals for v in (a.P, a.Q)),
        ):
            values.setflags(write=False)
        self._pairs = []
        for i, a in enumerate(self.orbitals):
            for b in self.orbitals[i:]:
                fa = a.occupation / (2 * abs(a.kappa))
                fb = b.occupation / (2 * abs(b.kappa))
                weight = fa if (a.n, a.l) == (b.n, b.l) else fa * fb
                weight *= 1 if a is b else 2
                ranks = [
                    (L, (2 * L + 1) * reduced_C(a.kappa, b.kappa, L) ** 2)
                    for L in range(a.l + b.l + 1)
                ]
                self._pairs.append(
                    (
                        weight,
                        np.asarray(a.P) * b.P + np.asarray(a.Q) * b.Q,
                        [(L, c) for L, c in ranks if c],
                    )
                )

    @classmethod
    def from_result(cls, result):
        """Consume a DF or MCDHF result with a B-spline quadrature grid.

        The n-l configuration-average occupation restrictions still apply;
        a resolved jj determinant is not an isotropic n-l ensemble.
        """
        from atomx.numerics.bspline.evaluate import eval_P, eval_Q

        bd = getattr(result, "_bsp_data", None) or getattr(result, "bsp_data", None)
        if bd is None:
            raise ValueError("Scattering requires a B-spline bound-state result")
        if hasattr(result, "orbitals_by_subshell"):
            radial = result.orbitals_by_subshell()
        elif hasattr(result, "orbitals_dict") and hasattr(result, "occupations"):
            radial = {
                key: (*pair, result.occupations.get(key, 0.0))
                for key, pair in result.orbitals_dict.items()
            }
        else:
            raise ValueError(
                "Bound result must expose radial coefficients and occupations"
            )
        orbitals = [
            ScatteringOrbital(
                n, k, occupation, eval_P(P, bd).ravel(), eval_Q(Q, bd).ravel()
            )
            for (n, k), (P, Q, occupation) in radial.items()
        ]
        return cls(bd["gr"].ravel(), bd["grw"].ravel(), orbitals)

    @staticmethod
    def _q(q):
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError("Momentum transfer must be finite and nonnegative")
        return q

    def form_factor(self, q):
        """Real nonresonant f0(q); f0(0)=N for normalized orbitals."""
        q = self._q(q)
        density = sum(
            (
                a.occupation * (np.asarray(a.P) ** 2 + np.asarray(a.Q) ** 2)
                for a in self.orbitals
            ),
            np.zeros_like(self.r),
        )
        return spherical_jn(0, q[..., None] * self.r) @ (self.weights * density)

    def incoherent_function(self, q):
        """Waller–Hartree S(q), with S(0)=0 and S(infinity)=N."""
        q = self._q(q)
        x = q[..., None] * self.r
        bessel = {}
        occupied = np.zeros(q.shape)
        for weight, overlap, ranks in self._pairs:
            for L, coefficient in ranks:
                if L not in bessel:
                    bessel[L] = spherical_jn(L, x)
                radial = bessel[L] @ (self.weights * overlap)
                occupied += weight * coefficient * radial**2
        return self.electrons - occupied

    def rayleigh(self, energy_eV, theta):
        """Unpolarized Thomson * f0(q)^2, in barn/sr."""
        q = momentum_transfer(energy_eV, theta)
        return thomson(theta) * self.form_factor(q) ** 2

    def compton(self, energy_eV, theta, *, kernel="klein_nishina"):
        """Angle-only scattering-function approximation in barn/sr.

        Uses elastic q=2 alpha omega sin(theta/2) for S. The default
        weights it with Klein–Nishina; ``kernel='thomson'`` reproduces
        XATOM's Waller–Hartree output. Neither computes the DDCS.
        """
        q = momentum_transfer(energy_eV, theta)
        if kernel == "klein_nishina":
            free = klein_nishina(energy_eV, theta)
        elif kernel == "thomson":
            free = thomson(theta)
        else:
            raise ValueError("kernel must be klein_nishina or thomson")
        return free * self.incoherent_function(q)


def _angle(theta):
    theta = np.asarray(theta, dtype=float)
    if not np.isfinite(theta).all() or np.any((theta < 0) | (theta > np.pi)):
        raise ValueError("Scattering angles must be in [0, pi] radians")
    return theta


def _energy(energy_eV):
    energy = np.asarray(energy_eV, dtype=float)
    if not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError("Photon energy must be positive and finite")
    return energy


def momentum_transfer(energy_eV, theta):
    """Elastic momentum transfer in inverse Bohr."""
    return 2 * ALPHA * _energy(energy_eV) / HARTREE_EV * np.sin(_angle(theta) / 2)


def thomson(theta):
    """Free-electron unpolarized Thomson cross section, barn/sr."""
    return _RE2_BARN * (1 + np.cos(_angle(theta)) ** 2) / 2


def klein_nishina(energy_eV, theta):
    """Free-electron unpolarized Klein–Nishina cross section, barn/sr."""
    theta = _angle(theta)
    ratio = 1 / (1 + _energy(energy_eV) / HARTREE_EV * ALPHA**2 * (1 - np.cos(theta)))
    return _RE2_BARN / 2 * ratio**2 * (ratio + 1 / ratio - np.sin(theta) ** 2)
