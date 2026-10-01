"""Positive-energy finite-box Dirac roots, found by radial shooting.

E excludes electron rest energy. The regular origin condition and P(R)=0
define a self-adjoint radial boundary problem. These are box-normalized
pseudostates, not delta-energy-normalized scattering states. No asymptotic
amplitude or phase approximation enters their normalization.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constants import SL
from .dirac_integrator import doutw
from .problem import ShootingRadialProblem


class BoxDiracContinuum:
    """Reuse potential interpolation and grid for positive box roots.

    Node counts bracket successive roots and detect skipped roots. A grid
    with inadequate oscillation resolution or quadrature orthogonality is
    rejected. Localized bound states are supplied separately; diffuse
    negative box roots can be filled with ``negative_roots_above``.
    """

    def __init__(self, radius, jacobian=None, rv_potential=None, *, ode_tolerance=1e-11):
        self.problem = ShootingRadialProblem.coerce(radius, jacobian, rv_potential)
        self.r, self.jacobian, rv = self.problem.r, self.problem.jacobian, self.problem.rv
        if not np.isfinite(ode_tolerance) or not 0 < ode_tolerance < 1e-4:
            raise ValueError('invalid ODE tolerance')
        self.tolerance = ode_tolerance
        self.weights = self.problem.weights
        self.potential = self.problem.spline
        self.indices = self.problem.indices
        self.rv = rv
        self.interval_width = np.diff(self.r)
        self.mid_potential = (rv[1:]+rv[:-1])/(self.r[1:]+self.r[:-1])

    def _shoot(self, energy, kappa):
        kinetic = energy-self.mid_potential
        phase_step = self.interval_width*np.sqrt(np.maximum(
            kinetic*(2+kinetic/SL**2), 0.))
        # Adaptive ODE substeps cannot repair an undersampled output basis
        # subsequently used in fixed-grid CI quadrature. Resolve its fastest
        # local wavelength even if the endpoint integration itself succeeds.
        if np.max(phase_step) > .5:
            raise ValueError('radial grid does not resolve continuum quadrature (phase step > 0.5)')
        wave = np.zeros((2, len(self.r)))
        nodes, _, _, max_nodes = doutw(energy, self.tolerance, float(kappa),
            10**9, len(self.r), 0, self.r, self.indices, len(self.r),
            *self.potential, *wave)
        boundary_norm = float(np.hypot(*wave[:, -1]))
        if not np.all(np.isfinite(wave)) or boundary_norm == 0:
            raise FloatingPointError('invalid box continuum integration')
        if max_nodes > 1:
            raise ValueError('radial grid does not resolve continuum oscillations')
        return float(wave[0, -1]/boundary_norm), int(nodes), wave

    def solve(self, kappa, count, *, orthogonality_tolerance=2e-6):
        """Return the first ``count`` strictly positive box roots and spinors.

        The zero-energy node count identifies the first positive root even
        when the attractive potential supports many negative-energy roots.
        A root exactly at threshold is rejected because its assignment to
        bound or continuum space needs an explicit caller policy.
        """
        if not isinstance(kappa, (int, np.integer)) or kappa == 0:
            raise ValueError('kappa must be a nonzero integer')
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError('count must be a positive integer')
        if not np.isfinite(orthogonality_tolerance) or not 0 < orthogonality_tolerance < 1:
            raise ValueError('invalid orthogonality tolerance')
        left, momentum = 0., 0.
        fleft, nodes_left, _ = self._shoot(left, kappa)
        if abs(fleft) < 1e-10:
            raise ValueError('box eigenvalue at zero-energy threshold')
        initial_nodes = nodes_left
        step = np.pi/(2*self.r[-1])
        energies, waves, residuals = [], [], []
        evaluations = 1
        # Momentum spacing bounds the work for ordinary attractive atomic
        # potentials; node-count refinement prevents skipping narrow roots.
        for _ in range(100000):
            next_momentum = momentum+step
            right = next_momentum**2/(np.sqrt(1+(next_momentum/SL)**2)+1)
            fright, nodes_right, _ = self._shoot(right, kappa)
            evaluations += 1
            jump = nodes_right-nodes_left
            if jump < 0:
                raise RuntimeError('nonmonotone Dirac node count')
            if jump > 1:
                step *= .5
                if step < np.finfo(float).eps*max(1., momentum):
                    raise RuntimeError('unresolved continuum root bracket')
                continue
            if jump == 1:
                if fleft*fright > 0:
                    raise RuntimeError('node count and boundary root disagree')
                def boundary(energy):
                    nonlocal evaluations
                    evaluations += 1
                    return self._shoot(energy, kappa)[0]
                root = brentq(boundary, left, right, xtol=1e-13, rtol=2e-14)
                _, _, wave = self._shoot(root, kappa)
                evaluations += 1
                norm = float(np.sum(wave*wave*self.weights))
                if not np.isfinite(norm) or norm <= 0:
                    raise FloatingPointError('invalid continuum normalization')
                wave /= np.sqrt(norm)
                residual = abs(wave[0, -1])/np.max(abs(wave))
                if residual > 1e-8:
                    raise RuntimeError('continuum box boundary failed to converge')
                energies.append(root)
                waves.append(wave)
                residuals.append(residual)
                if len(energies) == count:
                    break
            left, fleft, nodes_left, momentum = right, fright, nodes_right, next_momentum
        else:
            raise RuntimeError('continuum root search exhausted')
        waves = np.array(waves)
        flat = waves.reshape(count, -1)
        gram = (waves*self.weights).reshape(count, -1) @ flat.T
        defect = float(np.max(abs(gram-np.eye(count))))
        if defect > orthogonality_tolerance:
            raise ValueError(f'continuum quadrature underresolved: overlap defect {defect:.3g}')
        return np.array(energies), waves, {
            'boundary': 'regular origin; P(R)=0', 'normalization': 'finite-box L2',
            'initial_zero_energy_nodes': initial_nodes,
            'orthogonality_defect': defect, 'boundary_residual': float(max(residuals)),
            'shooting_evaluations': evaluations, 'ode_tolerance': self.tolerance,
            'scope': 'positive-energy box roots only; negative box roots not included'}

    def negative_roots_above(self, kappa, lower_energy, *, expected_below):
        """Fill diffuse negative box roots above a localized bound basis.

        Deep roots are supplied separately by the two-sided bound solver.
        Node counts must confirm exactly ``expected_below`` roots beneath
        the cutoff. Shallow roots are reconstructed by inward/outward
        matching: outward integration alone loses the decaying tail near
        a bound eigenvalue, even when its energy is accurately bracketed.
        """
        from .dirac_green import scaled_box_pair
        if (not isinstance(kappa, (int, np.integer)) or kappa == 0
                or not isinstance(expected_below, (int, np.integer)) or expected_below < 0
                or not np.isfinite(lower_energy) or not -SL**2 < lower_energy < 0):
            raise ValueError('invalid negative box root cutoff')
        _, below, _ = self._shoot(lower_energy, kappa)
        _, total, _ = self._shoot(0., kappa)
        if below != expected_below:
            raise ValueError('negative box cutoff does not follow the supplied bound roots')
        brackets, pending = [], [(lower_energy, 0., below, total)]
        while pending:
            lo, hi, nlo, nhi = pending.pop()
            if nhi == nlo:
                continue
            if nhi < nlo:
                raise RuntimeError('nonmonotone negative box node count')
            if nhi-nlo == 1:
                brackets.append((lo, hi))
                continue
            mid = .5*(lo+hi)
            if mid in (lo, hi):
                raise RuntimeError('unresolved negative box root bracket')
            _, nm, _ = self._shoot(mid, kappa)
            pending.extend([(lo, mid, nlo, nm), (mid, hi, nm, nhi)])
        roots, waves, defects = [], [], []
        for lo, hi in sorted(brackets):
            # Match near the classical outer turning point. The match index
            # stays fixed within each energy bracket to keep its determinant
            # smooth throughout the scalar root solve.
            allowed = np.flatnonzero(self.rv[1:]/self.r[1:] < .5*(lo+hi))+1
            index = int(np.clip(allowed[-1] if len(allowed) else len(self.r)//2,
                                2, len(self.r)-3))
            def match(energy, reconstruct=False, *, index=index):
                # Match directions, not globally scaled amplitudes. A heavy
                # nucleus in a large box can span thousands of exponential
                # log units: rescaling an entire shooting history erases the
                # regular solution at the match point before normalization.
                outward, inward, log_out, log_in, _, _ = scaled_box_pair(
                    energy, kappa, self.r, *self.potential)
                u, v = outward[:, index], inward[:, index]
                determinant = u[0]*v[1]-u[1]*v[0]
                if not reconstruct:
                    return determinant
                wave = outward.copy()
                wave[:, index+1:] = (u@v)*inward[:, index+1:]
                amplitude = np.concatenate((log_out[:index+1]-log_out[index],
                                            log_in[index+1:]-log_in[index]))
                # One common offset preserves matching and normalization;
                # only physically negligible tails can underflow to zero.
                wave *= np.exp(amplitude-np.max(amplitude))
                norm = float(np.sum(wave*wave*self.weights))
                if not np.isfinite(norm) or norm <= 0:
                    raise FloatingPointError('invalid matched box norm')
                return wave/np.sqrt(norm), abs(determinant)
            root = brentq(match, lo, hi, xtol=1e-13, rtol=2e-14)
            wave, defect = match(root, reconstruct=True)
            if defect > 1e-8:
                raise RuntimeError('negative box matching failed to converge')
            roots.append(root)
            waves.append(wave)
            defects.append(defect)
        return np.array(roots), np.array(waves).reshape((-1, 2, len(self.r))), {
            'negative_box_roots': total, 'localized_roots_below_cutoff': below,
            'matched_negative_roots': len(roots),
            'maximum_matching_defect': float(max(defects, default=0.))}
