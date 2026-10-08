"""Explicit magnetic-state XFEL coherence, with secular Markov loss channels.

Rotating-wave Hamiltonians use signed dipoles and a complex field envelope.
Configuration-average line strengths cannot be promoted to a coherent
multielectron Hamiltonian. The MCDHF adapter consumes signed CI amplitudes;
the native configuration-rate adapter requires one-electron states.
KNOWN GAP: no Fano bound-continuum coherence, non-Markovian continuum,
stochastic SASE ensemble, or coherent strong-field imaging response.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, diags, eye, issparse, kron

from atomx.angular.wigner import wigner_3j
from atomx.constants import ALPHA, AU_RATE_PER_S, BOHR_RADIUS_M, HARTREE_EV
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL
from atomx.scattering.photoionization.subshell import subshell_photoionization
from atomx.xfel.coherent_subspace import dipole_commutator, reachable_coordinates
from atomx.xfel.kinetics import GaussianPulse
from atomx.xfel.model import gaussian_spectral_quadrature, row_config
from atomx.xfel.radiative_reservoir import finite_time_error_bound
from atomx.xfel.reservoir_errors import real_residual_jacobian

_RATE_FS = AU_RATE_PER_S * 1e-15
_HBAR_EV_FS = HARTREE_EV / _RATE_FS
_AREA_CM2 = (BOHR_RADIUS_M * 100) ** 2


@dataclass(frozen=True)
class CoherentGaussianPulse:
    """Transform-limited Gaussian intensity, with one explicit carrier phase.

    E(t)=Re[envelope(t)*exp(-i*omega*t)]. Bandwidth is derived from the
    intensity duration, never added as incoherent broadening of the drive.
    """

    photon_energy_eV: float
    fluence_photons_per_cm2: float
    fwhm_fs: float
    center_fs: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self):
        GaussianPulse(
            self.photon_energy_eV,
            self.fluence_photons_per_cm2,
            self.fwhm_fs,
            self.center_fs,
        )
        if not np.isfinite(self.phase_rad):
            raise ValueError("finite carrier phase required")
        if self.bandwidth_fwhm_eV > 0.1 * self.photon_energy_eV:
            raise ValueError("pulse spectrum is too broad for the rotating-wave model")

    @property
    def bandwidth_fwhm_eV(self):
        return 4 * np.log(2) * _HBAR_EV_FS / self.fwhm_fs

    @property
    def support_fs(self):
        width = 8 * self.fwhm_fs / np.sqrt(8 * np.log(2))
        return self.center_fs - width, self.center_fs + width

    def flux(self, t_fs):
        sigma = self.fwhm_fs / np.sqrt(8 * np.log(2))
        return (
            self.fluence_photons_per_cm2
            / (sigma * 1e-15 * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((np.asarray(t_fs) - self.center_fs) / sigma) ** 2)
        )

    def electric_field_au(self, t_fs):
        # I_au=c*E_peak^2/(8*pi), I=omega*photon_flux in atomic units.
        return np.sqrt(
            8
            * np.pi
            * ALPHA
            * self.photon_energy_eV
            / HARTREE_EV
            * self.flux(t_fs)
            * _AREA_CM2
            / AU_RATE_PER_S
        ) * np.exp(1j * self.phase_rad)


@dataclass
class CoherentDynamics:
    times_fs: np.ndarray
    state_ids: tuple
    density_matrices: np.ndarray
    charges: np.ndarray
    provenance: dict

    @property
    def populations(self):
        return self.density_matrices.diagonal(axis1=1, axis2=2).real

    @property
    def mean_charge(self):
        return self.populations @ self.charges

    def expectation(self, operator):
        if issparse(operator):
            op = operator.tocoo()
            if op.shape != self.density_matrices.shape[1:] or not np.isfinite(op.data).all():
                raise ValueError("operator must match the explicit state basis")
            return self.density_matrices[:, op.col, op.row] @ op.data
        op = np.asarray(operator, dtype=complex)
        if op.shape != self.density_matrices.shape[1:] or not np.isfinite(op).all():
            raise ValueError("operator must match the explicit state basis")
        return np.einsum("tij,ji->t", self.density_matrices, op)


def _dissipator(operators, n):
    unit = eye(n, format="csc", dtype=complex)
    out = csc_matrix((n * n, n * n), dtype=complex)
    for op in operators:
        c = csc_matrix(op)
        cc = c.conj().T @ c
        # Column-major vec(C rho C†) = (C* tensor C) vec(rho).
        out += kron(c.conj(), c, format="csc") - 0.5 * (
            kron(unit, cc, format="csc") + kron(cc.T, unit, format="csc")
        )
    return out


class CoherentSystem:
    """Explicit level Hamiltonian plus Lindblad decay and PI jump operators.

    dipole_up_au contains only photon-ladder raising matrix elements,
    <upper|D.polarization|lower>, including their relative complex phases.
    Decay operators have units sqrt(s^-1); PI operators sqrt(cm^2).
    Pulse-driven population edges must not be included a second time.
    """

    def __init__(
        self,
        energies_eV,
        photon_numbers,
        dipole_up_au,
        *,
        charges,
        decay_operators=(),
        pi_operators=(),
        reference_decay_operators=None,
        reference_pi_operators=None,
        provenance,
    ):
        self.state_ids = tuple(energies_eV)
        n = len(self.state_ids)
        if (
            n < 1
            or set(photon_numbers) != set(self.state_ids)
            or set(charges) != set(self.state_ids)
            or not provenance
        ):
            raise ValueError(
                "explicit energies, photon ladders, charges and provenance required"
            )
        self.energies = np.array(list(energies_eV.values()), dtype=float)
        self.ladders = np.array(
            [photon_numbers[s] for s in self.state_ids], dtype=float
        )
        self.charges = np.array([charges[s] for s in self.state_ids], dtype=float)
        if (
            not np.isfinite([self.energies, self.ladders, self.charges]).all()
            or np.any(self.ladders != self.ladders.astype(int))
            or np.any(self.charges < 0)
            or np.any(self.charges != self.charges.astype(int))
        ):
            raise ValueError("finite energies and integer ladders/charges required")
        self.dipole = np.array(dipole_up_au, dtype=complex, copy=True)
        if self.dipole.shape != (n, n) or not np.isfinite(self.dipole).all():
            raise ValueError("dipole matrix must match the explicit state basis")
        for i, j in zip(*np.nonzero(self.dipole), strict=True):
            if (
                self.ladders[i] - self.ladders[j] != 1
                or self.charges[i] != self.charges[j]
                or self.energies[i] <= self.energies[j]
            ):
                raise ValueError(
                    "dipoles must raise one photon at fixed charge and positive energy"
                )
        self.decay_operators = self._operators(decay_operators, n)
        self.pi_operators = self._operators(pi_operators, n)
        if (reference_decay_operators is None) != (reference_pi_operators is None):
            raise ValueError("both reference decay and PI operator groups required")
        self.reference_decay_operators = (None if reference_decay_operators is None
                                         else self._operators(reference_decay_operators, n))
        self.reference_pi_operators = (None if reference_pi_operators is None
                                      else self._operators(reference_pi_operators, n))
        self._reference_liouvillian_cache = None
        for group in (self.decay_operators, self.pi_operators,
                      self.reference_decay_operators or (), self.reference_pi_operators or ()):

            for op in group:
                if np.any(op.diagonal()):
                    raise ValueError(
                        "loss channels must be jumps, not unexplained dephasing"
                    )
                for i, j in zip(*op.nonzero(), strict=True):
                    if self.charges[i] < self.charges[j]:
                        raise ValueError("loss jump cannot lower charge")
                differences = {
                    self.ladders[i] - self.ladders[j]
                    for i, j in zip(*op.nonzero(), strict=True)
                }
                if len(differences) > 1:
                    raise ValueError(
                        "shared jump paths must have one photon-ladder difference; "
                        "otherwise the rotating-frame reservoir is time dependent"
                    )
        self.decay_liouvillian = _dissipator(self.decay_operators, n) * 1e-15
        self.pi_liouvillian = _dissipator(self.pi_operators, n) * 1e-15
        self.provenance = dict(provenance)

    @staticmethod
    def _operators(operators, n):
        values = tuple(csc_matrix(c, dtype=complex, copy=True) if issparse(c)
                       else np.array(c, dtype=complex, copy=True) for c in operators)
        if any(c.shape != (n, n) or not np.isfinite(c.data if issparse(c) else c).all()
               for c in values):
            raise ValueError("jump operators must match the explicit state basis")
        return values

    def solve(
        self,
        pulse,
        initial_density,
        *,
        times_fs=None,
        after_pulse_fs=0,
        rtol=1e-9,
        atol=1e-12,
        max_step_fs=None,
        density_representation="auto",
        reservoir_certificate="uniform",
        integrator="DOP853",
    ):
        """Evolve the declared finite-space RWA model and check its density.

        ``reachable`` uses exact coordinate closure, including reference
        generators when requested; ``auto`` enables it above 128 states.
        Small ``auto`` systems retain the historical full-density DOP853 path.

        ``residual`` explicitly compares against supplied original-energy
        electron operators and the declared arithmetic radiative reference.
        It integrates an analytical residual bound twice with tighter time
        resolution. The observed convergence margin is not an interval bound
        on numerical error or a certificate of microscopic reservoir physics.
        The default ``uniform`` bound and all physical applicability gates
        remain in force. BDF is optional; its nonholomorphic residual norm is
        integrated in real coordinates with the complete sparse Jacobian.
        """
        if integrator not in ("DOP853", "BDF"):
            raise ValueError("integrator must be DOP853 or BDF")
        if density_representation not in ("auto", "full", "reachable"):
            raise ValueError("density_representation must be auto, full or reachable")
        if reservoir_certificate not in ("uniform", "residual"):
            raise ValueError("reservoir_certificate must be uniform or residual")
        residual_certificate = reservoir_certificate == "residual"
        if residual_certificate and self.reference_decay_operators is None:
            raise ValueError("residual certificate requires explicit original-energy reference operators")
        if not isinstance(pulse, CoherentGaussianPulse):
            raise ValueError(
                "coherent evolution requires an explicit complex-field pulse"
            )
        n = len(self.state_ids)
        rho = np.asarray(initial_density, dtype=complex)
        if (
            rho.shape != (n, n)
            or not np.isfinite(rho).all()
            or not np.allclose(rho, rho.conj().T, rtol=0, atol=1e-12)
            or abs(np.trace(rho) - 1) > 1e-12
            or np.linalg.eigvalsh(rho).min() < -1e-12
        ):
            raise ValueError(
                "initial density must be Hermitian, positive and trace one"
            )
        if not np.isfinite(after_pulse_fs) or after_pulse_fs < 0:
            raise ValueError("post-pulse duration must be nonnegative")
        radiative_error = None
        eta = self.provenance.get("radiative_arithmetic_generator_difference_bound_per_s")
        if eta is not None:
            start, stop = pulse.support_fs
            radiative_error = finite_time_error_bound(eta, stop - start + after_pulse_fs)
            budget = self.provenance["radiative_arithmetic_trace_norm_error_budget"]
            if not np.isfinite(budget) or budget <= 0 or radiative_error > budget:
                raise ValueError("radiative finite-time arithmetic-reference error exceeds trace-norm budget; shorten duration or use a different reservoir")
        reservoir_error = None
        uniform_reservoir_error = None
        electron_pi = self.provenance.get("electron_pi_generator_difference_bound_cm2")
        if electron_pi is not None:
            electron_auger = self.provenance["electron_auger_generator_difference_bound_per_s"]
            if not np.isfinite([electron_pi, electron_auger]).all() or min(electron_pi, electron_auger) < 0:
                raise ValueError("finite nonnegative electron generator bounds required")
            start, stop = pulse.support_fs
            # Full photon fluence conservatively includes the Gaussian tails
            # outside the finite pulse support. Recheck when fluence/time changes.
            exponent = ((eta or 0.0) + electron_auger) * (stop-start+after_pulse_fs)*1e-15
            exponent += electron_pi * pulse.fluence_photons_per_cm2
            reservoir_error = float(np.expm1(exponent)) if exponent < 700 else np.inf
            budget = self.provenance["reservoir_trace_norm_error_budget"]
            uniform_reservoir_error = reservoir_error
            if not np.isfinite(budget) or budget <= 0 or (reservoir_error > budget and not residual_certificate):
                raise ValueError(
                    f"combined reservoir arithmetic/original-energy reference exceeds trace-norm budget: "
                    f"{reservoir_error:.8g} > {budget:.8g}")
        if residual_certificate and electron_pi is None:
            raise ValueError("residual certificate requires explicit electron error bounds and budget")
        spectrum = self.provenance.get("continuum_bandwidth_fwhm_eV")
        if spectrum is not None and not np.isclose(
            spectrum, pulse.bandwidth_fwhm_eV, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "coherent pulse spectrum differs from continuum loss table"
            )
        energy = self.provenance.get("photon_energy_eV")
        if energy is not None and not np.isclose(
            energy, pulse.photon_energy_eV, rtol=0, atol=1e-9
        ):
            raise ValueError("carrier energy differs from native continuum table")
        peak = abs(pulse.electric_field_au(pulse.center_fs))
        rabi_eV = peak * np.linalg.norm(self.dipole, 2) * HARTREE_EV
        if rabi_eV > 0.1 * pulse.photon_energy_eV:
            raise ValueError("coupling exceeds rotating-wave applicability gate")
        for i, j in zip(*np.nonzero(self.dipole), strict=True):
            if (
                abs(self.energies[i] - self.energies[j] - pulse.photon_energy_eV)
                > 0.1 * pulse.photon_energy_eV
            ):
                raise ValueError(
                    "driven transition is outside the near-resonant rotating-wave model"
                )
        detunings = self.energies - self.ladders * pulse.photon_energy_eV
        start, stop = pulse.support_fs
        stop += after_pulse_fs
        times = (
            np.linspace(start, stop, 1001)
            if times_fs is None
            else np.asarray(times_fs, dtype=float)
        )
        if (
            times.ndim != 1
            or len(times) < 2
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or times[0] < start
            or times[-1] > stop
        ):
            raise ValueError(
                "output times must increase within pulse integration support"
            )
        diagonal = np.diag(detunings / _HBAR_EV_FS)
        large_basis = n > 128
        if large_basis:
            frequency_difference = (detunings[:, None] - detunings[None, :]) / _HBAR_EV_FS
            sparse_dipole = csc_matrix(self.dipole)
            sparse_adjoint = sparse_dipole.conj().T

        def rhs(t, y):
            density = y.reshape((n, n), order="F")
            field = -0.5 * pulse.electric_field_au(t) * _RATE_FS
            if large_basis:
                # Exact diagonal action plus signed sparse E1; no rotating terms
                # or coherences are discarded. Keep the small-basis path stable.
                drive = field * sparse_dipole + field.conjugate() * sparse_adjoint
                commutator = -1j * (frequency_difference * density
                    + drive @ density - (drive.T @ density.T).T)
            else:
                v = -0.5 * pulse.electric_field_au(t) * self.dipole * _RATE_FS
                h = diagonal + v + v.conj().T
                commutator = -1j * (h @ density - density @ h)
            return (
                commutator.ravel(order="F")
                + self.decay_liouvillian @ y
                + float(pulse.flux(t)) * (self.pi_liouvillian @ y)
            )

        max_step = (
            min(pulse.fwhm_fs / 40, 0.1 * _HBAR_EV_FS / rabi_eV)
            if rabi_eV
            else pulse.fwhm_fs / 40
        )
        if max_step_fs is not None:
            if not np.isfinite(max_step_fs) or max_step_fs <= 0:
                raise ValueError("maximum time step must be finite and positive")
            max_step = min(max_step, max_step_fs)
        compact = density_representation == "reachable" or (
            density_representation == "auto" and n > 128)
        initial = rho.ravel(order="F")
        reference_generators = []
        if residual_certificate:
            if self._reference_liouvillian_cache is None:
                self._reference_liouvillian_cache = (
                    _dissipator(self.reference_decay_operators, n)*1e-15,
                    _dissipator(self.reference_pi_operators, n)*1e-15)
            reference_decay, reference_pi = self._reference_liouvillian_cache
            reference_generators = [reference_decay, reference_pi]
        coords = np.arange(n*n)
        if compact or residual_certificate or integrator == "BDF":
            up = dipole_commutator(self.dipole)
            down = dipole_commutator(self.dipole.conj().T)
            if compact:
                coords = reachable_coordinates(initial, [self.decay_liouvillian,
                    self.pi_liouvillian, up, down, *reference_generators])
            def restrict(matrix):
                return matrix[coords][:, coords].tocsr()
            frequency = (detunings[:, None]-detunings[None, :])/_HBAR_EV_FS
            drift = diags((-1j*frequency).ravel(order="F"), format="csc")
            decay = restrict(self.decay_liouvillian+drift)
            photo = restrict(self.pi_liouvillian)
            up, down = restrict(up), restrict(down)
            def compact_rhs(t, vector):
                field = -.5*pulse.electric_field_au(t)*_RATE_FS
                return (decay@vector + float(pulse.flux(t))*(photo@vector)
                        + field*(up@vector) + field.conjugate()*(down@vector))
            def affine_generator(t):
                field = -.5*pulse.electric_field_au(t)*_RATE_FS
                return decay + float(pulse.flux(t))*photo + field*up + field.conjugate()*down
            physical_rhs = compact_rhs
        else:
            physical_rhs = rhs
        numerical = {}
        if residual_certificate:
            # Reference electron generators are CPTP. Their propagator is a
            # trace-norm contraction on Hermitian matrices. Only the omitted
            # arithmetic radiative correction contributes eta_rad to growth.
            delta_decay = restrict(reference_decay-self.decay_liouvillian)
            delta_pi = restrict(reference_pi-self.pi_liouvillian)
            eta_fs = (eta or 0.0)*1e-15
            def certified_rhs(t, vector):
                state = vector[:-1]
                residual = delta_decay@state + float(pulse.flux(t))*(delta_pi@state)
                # ||R||_1 <= sqrt(n)||R||_F, exact for the full embedded matrix.
                forcing = np.sqrt(n)*np.linalg.norm(residual)
                bound_rate = eta_fs*(vector[-1]+1) + forcing
                return np.append(physical_rhs(t, state), bound_rate)
            initial_certified = np.append(initial[coords], 0j)
            if integrator == "BDF":
                # The residual norm is not complex analytic. Split Re/Im before
                # a stiff implicit solve; supply the complete real Jacobian.
                count = len(coords)
                real_initial = np.r_[initial[coords].real, initial[coords].imag, 0.]
                def real_rhs(t, vector):
                    state = vector[:count]+1j*vector[count:2*count]
                    result = certified_rhs(t, np.append(state,vector[-1]))
                    return np.r_[result[:-1].real,result[:-1].imag,result[-1].real]
                def real_jacobian(t, vector):
                    state = vector[:count]+1j*vector[count:2*count]
                    matrix = affine_generator(t)
                    error = delta_decay + float(pulse.flux(t))*delta_pi
                    return real_residual_jacobian(matrix,error,state,dimension=n,eta_fs=eta_fs)
            def integrate(tolerance_scale, step):
                real = integrator == "BDF"
                # Translate the stiff solve's time origin. At tight tolerances,
                # subtracting successive negative absolute times can otherwise
                # lose precision during the initial short decay steps.
                fun = (lambda elapsed,y:real_rhs(start+elapsed,y)) if real else certified_rhs
                jac = (lambda elapsed,y:real_jacobian(start+elapsed,y)) if real else None
                value = solve_ivp(fun, (0,stop-start) if real else (start,stop),
                    real_initial if real else initial_certified, t_eval=times-start if real else times,
                    method=integrator, rtol=rtol*tolerance_scale,
                    atol=atol*tolerance_scale, max_step=step, dense_output=True,
                    **({"jac":jac} if real else {}))
                if not value.success:
                    raise RuntimeError(value.message)
                if real:
                    dense = value.sol
                    value.y = np.vstack((value.y[:count]+1j*value.y[count:2*count],value.y[-1]))
                    def complex_dense(t):
                        data = dense(np.asarray(t)-start)
                        return np.concatenate((data[:count]+1j*data[count:2*count],data[-1:]),axis=0)
                    value.sol = complex_dense
                return value
            coarse = integrate(1, max_step)
            sol = integrate(.1, max_step/2)
            end_coarse, end_fine = coarse.sol(stop), sol.sol(stop)
            observed = max(np.linalg.norm(sol.y[:-1]-coarse.y[:-1], axis=0).max(),
                           np.linalg.norm(end_fine[:-1]-end_coarse[:-1]))*np.sqrt(n)
            bound_change = abs(float(end_fine[-1].real-end_coarse[-1].real))
            resolution = 1e-8
            if max(observed, bound_change) > resolution:
                raise RuntimeError("residual certificate time refinement is unresolved at 1e-8")
            estimate = float(end_fine[-1].real)
            if not np.isfinite(estimate) or estimate < 0 or abs(end_fine[-1].imag) > resolution:
                raise RuntimeError("invalid residual certificate integral")
            # Record the numerical resolution independently of the analytical
            # bound. This is convergence validation, not interval arithmetic.
            reservoir_error = estimate+observed+bound_change
            if reservoir_error > budget:
                raise ValueError(f"residual reservoir reference exceeds trace-norm budget: {reservoir_error:.8g} > {budget:.8g}")
            numerical = dict(reservoir_residual_bound_estimate=estimate,
                reservoir_certificate_density_refinement_upper=float(observed),
                reservoir_certificate_integral_refinement_change=bound_change,
                reservoir_certificate_numerical_resolution=resolution,
                reservoir_certificate_interpretation="analytical residual bound evaluated with time refinement; no interval arithmetic or microscopic bath bound")
            vectors = sol.y[:-1].T
        else:
            sol = solve_ivp(physical_rhs, (start,stop), initial[coords], t_eval=times,
                method=integrator,rtol=rtol,atol=atol,max_step=max_step,
                **({"jac":lambda t,y:affine_generator(t)} if integrator == "BDF" else {}))
            if not sol.success:
                raise RuntimeError(sol.message)
            vectors = sol.y.T
        if len(coords) < n*n:
            full = np.zeros((len(vectors), n*n), complex)
            full[:, coords] = vectors
            vectors = full
        densities = np.array([v.reshape((n,n), order="F") for v in vectors])
        trace_error = np.max(abs(np.trace(densities, axis1=1, axis2=2) - 1))
        hermitian_error = np.max(abs(densities - densities.conj().transpose(0, 2, 1)))
        eigen_min = np.linalg.eigvalsh(densities).min()
        tolerance = max(1e-8, 100 * atol)
        if (
            not np.isfinite(densities).all()
            or trace_error > tolerance
            or hermitian_error > tolerance
            or eigen_min < -tolerance
        ):
            raise RuntimeError("density positivity, Hermiticity or trace gate failed")
        return CoherentDynamics(
            times.copy(),
            self.state_ids,
            densities,
            self.charges.copy(),
            dict(
                self.provenance,
                method=f"RWA Lindblad {integrator}",
                density_frame="photon-ladder rotating frame",
                hamiltonian_action="diagonal plus sparse E1" if large_basis else "dense",
                peak_rabi_energy_eV=float(rabi_eV),
                max_trace_error=float(trace_error),
                max_hermiticity_error=float(hermitian_error),
                minimum_density_eigenvalue=float(eigen_min),
                radiative_arithmetic_trace_norm_error_bound=radiative_error,
                reservoir_reference_trace_norm_error_bound=reservoir_error,
                reservoir_uniform_reference_trace_norm_error_bound=uniform_reservoir_error,
                reservoir_certificate=reservoir_certificate,
                density_representation="exact reachable" if compact else "full",
                density_active_coordinates=len(coords),
                ode_function_evaluations=sol.nfev,
                **numerical,
            ),
        )


def magnetic_e1_component(j_bra2, m_bra2, j_ket2, m_ket2, q, reduced):
    """Wigner-Eckart tensor component in the repo's doubled-J convention."""
    if (
        q not in (-1, 0, 1)
        or m_bra2 not in range(-j_bra2, j_bra2 + 1, 2)
        or m_ket2 not in range(-j_ket2, j_ket2 + 1, 2)
    ):
        raise ValueError("invalid magnetic E1 quantum numbers")
    return (
        (-1) ** ((j_bra2 - m_bra2) // 2)
        * wigner_3j(j_bra2, 2, j_ket2, -m_bra2, 2 * q, m_ket2)
        * reduced
    )


class NativeOneElectronCoherentModel:
    """Expand a native one-electron model into magnetic states, linear z drive.

    One common initial-state orbital basis supplies signed E1 elements;
    radiative rates are rebuilt from that same basis and compared with
    the native graph. Independently relaxed
    orbitals cannot be assumed orthogonal. PI resolves m with the same
    linear polarization using state-specific native continuum reservoirs.
    Radiative operators sum degenerate m paths for each emitted spherical
    photon component. Distinct lines and PI continua use secular jumps.
    """

    def __init__(self, native_model, *, photon_numbers):
        rows = native_model.table["configs"]
        if any(
            sum(row_config(r).values()) != 1 or r.get("auger") or r.get("double_pi")
            for r in rows.values()
        ):
            raise ValueError(
                "native coherence requires exclusively one-electron configurations"
            )
        if set(photon_numbers) != set(rows):
            raise ValueError(
                "assign a photon ladder to every native bound configuration"
            )
        self.native_model = native_model
        energy = native_model.graph.photon_energy_eV
        bandwidth = native_model.table["provenance"]["continuum_bandwidth_fwhm_eV"]
        if any(
            abs(threshold - energy) <= 9 * bandwidth / np.sqrt(8 * np.log(2))
            for row in rows.values()
            for threshold in row["thresholds_eV"].values()
        ):
            raise ValueError(
                "near-threshold coherent PI needs continuum memory, beyond Markov loss"
            )
        energies, ladders, charges, by_config = (
            {"": 0.0},
            {"": 0},
            {"": native_model.graph.charges[""]},
            {},
        )
        self.config_by_state = {"": ""}
        for sid, row in rows.items():
            k = next(iter(row_config(row)))[1]
            by_config[sid] = {}
            for m2 in range(-2 * abs(k) + 1, 2 * abs(k), 2):
                label = f"{sid}|m2={m2}"
                by_config[sid][m2] = label
                self.config_by_state[label] = sid
                energies[label] = row["energy_hartree"] * HARTREE_EV
                ladders[label] = photon_numbers[sid]
                charges[label] = native_model.graph.charges[sid]
        self.indices = {s: i for i, s in enumerate(energies)}
        n = len(energies)
        common_result = native_model.result_loader(
            row_config(rows[native_model.graph.initial_state])
        )
        bd = common_result._bsp_data
        common = common_result.orbitals_by_subshell()
        shells = [next(iter(row_config(row))) for row in rows.values()]
        if not set(shells).issubset(common):
            raise ValueError("common initial-state orbital basis is incomplete")
        waves = {
            s: (eval_P(common[s][0], bd), eval_Q(common[s][1], bd)) for s in shells
        }
        gram = np.array(
            [
                [
                    float(
                        np.sum(
                            bd["grw"]
                            * (waves[a][0] * waves[b][0] + waves[a][1] * waves[b][1])
                        )
                    )
                    if a[1] == b[1]
                    else 0
                    for b in shells
                ]
                for a in shells
            ]
        )
        orthogonality_error = float(abs(gram - np.eye(len(shells))).max())
        if orthogonality_error > 1e-7:
            raise ValueError("coherent common orbital basis must be orthonormal")
        dipole, decay, photo = np.zeros((n, n), complex), [], []
        for line in native_model.lines:
            upper_config, lower_config = (
                row_config(rows[line.upper]),
                row_config(rows[line.lower]),
            )
            a, h = next(iter(upper_config)), next(iter(lower_config))
            # Static E1 length-gauge dipole radial via the shared kernel
            # (eval_orbital_pair == eval_P/eval_Q on this workspace).
            radial = radial_eL(
                common[a][0], common[a][1], common[h][0], common[h][1], bd, 1)
            reduced = float(angular_eL(a[1], h[1], 1) * radial)
            ju, jl = 2 * abs(a[1]) - 1, 2 * abs(h[1]) - 1
            # Einstein E1 prefactor (4/3)*alpha^3*omega^3 * AU_RATE_PER_S —
            # the constant inside einstein_A_eL for L=1.  The shared
            # function's /(2j_upper+1) is absent INTENTIONALLY: rate_scale
            # multiplies m-resolved amplitudes (magnetic_e1_component), and
            # the statistical weight emerges from the squared 3j sum over
            # (q, m_lower) at each upper sublevel — the coherent model needs
            # the per-sublevel amplitudes, not the averaged rate.
            rate_scale = (
                4 / 3 * ALPHA**3 * (line.energy_eV / HARTREE_EV) ** 3 * AU_RATE_PER_S
            )
            for q in (-1, 0, 1):
                op = np.zeros((n, n), complex)
                for mu, upper in by_config[line.upper].items():
                    for ml, lower in by_config[line.lower].items():
                        value = magnetic_e1_component(jl, ml, ju, mu, q, reduced)
                        op[self.indices[lower], self.indices[upper]] = (
                            np.sqrt(rate_scale) * value
                        )
                        if (
                            q == 0
                            and photon_numbers[line.upper] - photon_numbers[line.lower]
                            == 1
                        ):
                            dipole[self.indices[upper], self.indices[lower]] = np.conj(
                                value
                            )
                if np.any(op):
                    decay.append(op)
        # Rebuild lifetimes from the same orthonormal basis as the drive.
        # State-specific relaxed orbitals define a different rate model;
        # record its differences rather than rescaling dipoles to match it.
        total_rates = sum((c.conj().T @ c for c in decay), np.zeros((n, n), complex))
        rate_comparison = {}
        for sid, magnetic in by_config.items():
            expected = sum(ch["rate"] for ch in rows[sid]["radiative"])
            resolved_rates = [
                total_rates[self.indices[label], self.indices[label]].real
                for label in magnetic.values()
            ]
            if not np.allclose(
                resolved_rates, resolved_rates[0], rtol=1e-12, atol=1e-15
            ):
                raise ValueError(
                    "magnetic radiative sum violates rotational invariance"
                )
            rate_comparison[sid] = dict(
                common_basis_rate_per_s=float(resolved_rates[0]),
                native_state_specific_rate_per_s=expected,
                relative_difference=float(resolved_rates[0] / expected - 1)
                if expected
                else None,
            )
        for sid, magnetic in by_config.items():
            row = rows[sid]
            result = native_model.result_loader(row_config(row))
            label = next(s.label for s in result.subshells if s.occupation)
            photons, weights = gaussian_spectral_quadrature(
                energy, bandwidth, row["thresholds_eV"].values()
            )
            for m2, state in magnetic.items():
                pi = subshell_photoionization(
                    result,
                    photons,
                    continuum_model=native_model.continuum_model,
                    thresholds_eV=row["thresholds_eV"],
                    magnetic_m2_by_orbital={label: m2},
                    polarization_q=0,
                )
                sigma = float(weights @ pi.sigma_total_cm2)
                op = np.zeros((n, n), complex)
                op[0, self.indices[state]] = np.sqrt(sigma)
                if sigma:
                    photo.append(op)
        self.system = CoherentSystem(
            energies,
            ladders,
            dipole,
            charges=charges,
            decay_operators=decay,
            pi_operators=photo,
            provenance=dict(
                photon_energy_eV=energy,
                continuum_bandwidth_fwhm_eV=bandwidth,
                atomic_model="native one-electron Dirac radials and Delta-SCF energies",
                coherent_orbital_basis="common initial-state orthonormal orbitals",
                common_basis_max_orthogonality_error=orthogonality_error,
                radiative_rate_comparison=rate_comparison,
                polarization="linear z, spherical q=0",
                continuum="magnetic-resolved PI; secular Markov loss",
                known_gap="chosen basis, no bound-continuum Fano coherence or coherent imaging",
            ),
        )
        self.initial_density = np.zeros((n, n), complex)
        initial = by_config[native_model.graph.initial_state]
        for state in initial.values():
            self.initial_density[self.indices[state], self.indices[state]] = 1 / len(
                initial
            )

    def solve(self, pulse, **kwargs):
        return self.system.solve(pulse, self.initial_density, **kwargs)

    def configuration_populations(self, dynamics):
        if dynamics.state_ids != self.system.state_ids:
            raise ValueError("density matrix belongs to a different magnetic basis")
        return {
            sid: sum(
                (
                    dynamics.populations[:, i]
                    for i, state in enumerate(dynamics.state_ids)
                    if self.config_by_state[state] == sid
                ),
                np.zeros(len(dynamics.times_fs)),
            )
            for sid in self.native_model.graph.state_ids
        }
