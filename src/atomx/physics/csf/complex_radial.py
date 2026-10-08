"""Owned point-Dirac positive-band snapshot with actual complex Coulomb R^k.

The projector is defined by the positive spectral band of the dilated
point-nuclear Dirac operator, followed by an EXPLICIT retained-root count.
This is a finite no-pair nuclear-projector DC radial model with an explicit
static Breit provider. It is not state-relaxed MCDHF, finite-nucleus/QED
or a certified resonance provider.
"""

from collections import OrderedDict
from types import MappingProxyType

import numpy as np

from atomx.constants import C_SQ
from atomx.numerics.bspline.complex_dirac import ComplexScaledDiracWorkspace
from atomx.numerics.bspline.slater import slater_rk_direct
from atomx.numerics.yk import compute_yk
from atomx.physics.csf.retardation import (
    distance_potential,
    transition_current_divergence,
)


class ComplexCIRadialSnapshot:
    """Single-theta owned common-grid radial provider for exact jj contraction.

    ``counts`` maps kappa to the number of lowest positive-band box roots.
    Orbital keys use consecutive labels starting at the allowed principal
    n; beyond bound states these are basis labels, not physical Rydberg n.
    No roots are silently omitted. ``workspaces`` all share exact knots,
    quadrature and Z. Every supplied space is included in the snapshot.
    """

    def __init__(self, workspaces, counts, theta, *, cache_bytes=64*1024**2,
                 with_derivatives=False):
        if not workspaces or set(workspaces) != set(counts):
            raise ValueError("explicit matching workspaces and retained counts required")
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError("nonnegative integer cache_bytes required")
        if not isinstance(with_derivatives, bool):
            raise ValueError("with_derivatives must be bool")
        first = next(iter(workspaces.values()))
        if not isinstance(first, ComplexScaledDiracWorkspace):
            raise ValueError("point-Dirac DKB workspaces required")
        waves, energies, indices, derivatives = {}, {}, {}, {}
        diagnostics = {}
        for kappa, ws in sorted(workspaces.items()):
            count = counts[kappa]
            if (not isinstance(ws, ComplexScaledDiracWorkspace) or kappa != ws.kappa
                    or ws.Z != first.Z or not np.array_equal(ws.knots, first.knots)
                    or not np.array_equal(ws.gr, first.gr) or not np.array_equal(ws.grw, first.grw)):
                raise ValueError("common point-nuclear Hamiltonian and exact grid required")
            if (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1):
                raise ValueError("positive integer retained-root count required")
            spectrum = ws.solve(theta)
            positive = np.flatnonzero(spectrum.energies.real > -C_SQ)
            if len(positive) < count:
                raise ValueError("requested positive-band count exceeds DKB spectrum")
            # Reject loss of spectral-band separation, do not silently
            # relabel a near-gap eigenvector as a positive-band orbital.
            if np.min(abs(spectrum.energies.real+C_SQ)) < 1:
                raise ValueError("Dirac spectral band is not separated from the gap boundary")
            selected = positive[:count]
            p, q = ws.sample_spinors(spectrum, selected)
            if with_derivatives:
                dp, dq = ws.sample_spinors(spectrum, selected, derivative=1)
            gram = (p*ws.grw.ravel()) @ p.T + (q*ws.grw.ravel()) @ q.T
            defect = float(np.max(abs(gram-np.eye(count))))
            if defect > 1e-6 or np.max(spectrum.residuals[selected]) > 1e-10:
                raise ValueError(f"selected DKB c-orthogonality/algebraic gate failed: {defect:g}")
            start = abs(kappa)+(kappa > 0)
            for offset in range(count):
                key = (start+offset, kappa)
                a, b = p[offset].copy(), q[offset].copy()
                a.setflags(write=False)
                b.setflags(write=False)
                waves[key] = (a, b)
                if with_derivatives:
                    da, db = dp[offset].copy(), dq[offset].copy()
                    da.setflags(write=False)
                    db.setflags(write=False)
                    derivatives[key] = (da, db)
                energies[key] = complex(spectrum.energies[selected[offset]])
                indices[key] = int(selected[offset])
            diagnostics[kappa] = dict(c_gram_defect=defect, retained_count=int(count),
                                     positive_band_size=int(len(positive)))
        self.theta, self.Z = float(theta), first.Z
        self.gr, self.grw = first.gr.copy(), first.grw.copy()
        self.gr.setflags(write=False)
        self.grw.setflags(write=False)
        self.keys = tuple(sorted(waves))
        self.orbitals = MappingProxyType(waves)
        self.derivatives = MappingProxyType(derivatives)
        self.energies = MappingProxyType(energies)
        self.indices = MappingProxyType(indices)
        self.diagnostics = MappingProxyType(diagnostics)
        self._cache = OrderedDict()
        self._cache_bytes, self._used_bytes = cache_bytes, 0
        self._integrals = {}

    def _density(self, pair):
        if len(pair[0]) == 3:
            return (self.orbitals[pair[0][1:]][pair[0][0] == 'Q']
                    * self.orbitals[pair[1][1:]][pair[1][0] == 'Q'])
        a, b = (self.orbitals[key] for key in pair)
        return a[0]*b[0] + a[1]*b[1]

    def _potential(self, rank, pair):
        key = (rank, pair)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        potential = compute_yk(self._density(pair).reshape(self.gr.shape), self.gr, self.grw,
                               *self.gr.shape, rank).ravel()
        return self._store_potential(key, potential)

    def _store_potential(self, key, potential):
        potential.setflags(write=False)
        if potential.nbytes <= self._cache_bytes:
            while self._used_bytes+potential.nbytes > self._cache_bytes:
                _, old = self._cache.popitem(last=False)
                self._used_bytes -= old.nbytes
            self._cache[key] = potential
            self._used_bytes += potential.nbytes
        return potential

    def coulomb(self, rank, a, c, b, d, *, method='poisson'):
        """C-product Slater integral including the e^(-i theta) dilation.

        Uses shared Y^k or independent split-triangle quadrature. Poisson
        reciprocity is checked before removing its permitted small roundoff.
        No conjugate density or empirical width enters the integral.
        """
        if (isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 0):
            raise ValueError("nonnegative integer multipole rank required")
        pair1, pair2 = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if any(key not in self.orbitals for key in pair1+pair2):
            raise ValueError("Coulomb integral requests an orbital outside the snapshot")
        return self._bilinear(rank, pair1, pair2, method)

    def _bilinear(self, rank, pair1, pair2, method):
        """Same shared Slater/Yk bilinear for Coulomb and Gaunt densities."""
        if method not in ('poisson', 'direct'):
            raise ValueError("method must be poisson or direct")
        pairs = tuple(sorted((pair1, pair2)))
        cache_key = (int(rank), pairs, method)
        if cache_key in self._integrals:
            return self._integrals[cache_key]
        rho1, rho2 = (self._density(pair) for pair in pairs)
        if method == 'direct':
            value = slater_rk_direct(rho1, rho2, self.gr, self.grw, rank)
        else:
            v1 = np.dot(self.grw.ravel()*rho1, self._potential(rank, pairs[1]))
            v2 = np.dot(self.grw.ravel()*rho2, self._potential(rank, pairs[0]))
            if abs(v1-v2) > 1e-8*max(1., abs(v1), abs(v2)):
                raise ValueError("Coulomb reciprocity gate failed; refine radial quadrature")
            value = (v1+v2)/2
        value = complex(np.exp(-1j*self.theta)*value)
        if not np.isfinite(value):
            raise ValueError("nonfinite complex radial Coulomb integral")
        # Scalar storage is bounded separately; no CI-build-global memo.
        if len(self._integrals) < 16384:
            self._integrals[cache_key] = value
        return value

    def gaunt(self, rank, a, c, b, d, *, method='poisson'):
        """Component Slater bilinear for shared signed Gaunt angular keys."""
        if (isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 0):
            raise ValueError("nonnegative integer multipole rank required")
        keys = (a, c, b, d)
        if any(len(key) != 3 or key[0] not in ('P', 'Q') or key[1:] not in self.orbitals
               for key in keys):
            raise ValueError("Gaunt key requires P/Q components of snapshot orbitals")
        return self._bilinear(rank, tuple(sorted((a,c))), tuple(sorted((b,d))), method)

    def _current(self, pair):
        if not self.derivatives:
            raise ValueError("static retardation requires analytic spinor derivatives")
        a, c = pair
        wa = self.orbitals[a]+self.derivatives[a]
        wc = self.orbitals[c]+self.derivatives[c]
        return transition_current_divergence(a, c, wa, wc, self.gr.ravel())

    def _distance_potential(self, rank, pair):
        key = ('t', rank, pair)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        potential = distance_potential(self._current(pair).reshape(self.gr.shape),
                                       self.gr, self.grw, rank).ravel()
        return self._store_potential(key, potential)

    def retardation(self, rank, a, c, b, d):
        """Static current-divergence bilinear, including exactly one e^-iθ.

        The shared distance kernel is degree +1 and the two coordinate
        derivatives degree -2 together. Their full Breit dilation is
        e^-iθ; no further rotation of sampled dP/dr,dQ/dr is applied.
        The angular factor 1/2 belongs to the shared retardation terms.
        """
        if (isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 0):
            raise ValueError("nonnegative integer multipole rank required")
        if any(key not in self.orbitals for key in (a,c,b,d)):
            raise ValueError("retardation key outside the snapshot")
        if not self.derivatives:
            raise ValueError("static retardation requires analytic spinor derivatives")
        if a == c or b == d:
            return 0j
        sign = (-1 if a > c else 1)*(-1 if b > d else 1)
        pairs = tuple(sorted((tuple(sorted((a,c))), tuple(sorted((b,d))))))
        key = ('t', int(rank), pairs)
        if key not in self._integrals:
            rho1, rho2 = (self._current(pair) for pair in pairs)
            v1 = np.dot(self.grw.ravel()*rho1, self._distance_potential(rank, pairs[1]))
            v2 = np.dot(self.grw.ravel()*rho2, self._distance_potential(rank, pairs[0]))
            if abs(v1-v2) > 1e-8*max(1.,abs(v1),abs(v2)):
                raise ValueError("retardation reciprocity gate failed; refine radial quadrature")
            value = complex(np.exp(-1j*self.theta)*(v1+v2)/2)
            if not np.isfinite(value):
                raise ValueError("nonfinite static retardation integral")
            if len(self._integrals) < 16384:
                self._integrals[key] = value
            return sign*value
        return sign*self._integrals[key]

    def evaluate_breit(self, key):
        """Complete shared static Breit key provider; no Gaunt-only fallback."""
        if key[0] == 'g':
            return self.gaunt(*key[1:])
        if key[0] == 't':
            return self.retardation(*key[1:])
        raise ValueError(f"unsupported static Breit radial key {key}")

    def evaluate(self, key):
        """Radial-key provider for exact_block_matrix(integral_dtype=complex)."""
        if key[0] == 'h':
            _, na, kappa, nc = key
            if (na, kappa) not in self.energies or (nc, kappa) not in self.energies:
                raise ValueError("one-body key outside the radial snapshot")
            return self.energies[na, kappa] if na == nc else 0j
        if key[0] != 'r':
            raise ValueError(f"unsupported radial key {key}")
        _, rank, a, c, b, d = key
        return self.coulomb(rank, a, c, b, d)


__all__ = ['ComplexCIRadialSnapshot']
