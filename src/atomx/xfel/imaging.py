"""Transient atomic imaging observables with population-fluctuation intensity."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import spherical_jn

from atomx.numerics.bspline.evaluate import eval_P, eval_Q


def native_form_factor(result, q_a0_inv):
    """Spherical f0 for actual jj occupations, without an LS fraction constraint.

    Waller-Hartree incoherent S(q) has additional occupation constraints;
    the linear spherical elastic density integral used here does not.
    """
    q = np.asarray(q_a0_inv, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("q must be a finite nonnegative inverse-bohr grid")
    bd = result._bsp_data
    r, w = bd["gr"].ravel(), bd["grw"].ravel()
    density = np.zeros_like(r)
    electrons = 0.0
    for (_n, k), (P, Q, occupation) in result.orbitals_by_subshell().items():
        if not np.isfinite(occupation) or not 0 <= occupation <= 2 * abs(k):
            raise ValueError("invalid native jj occupation")
        if not occupation:
            continue
        rho = eval_P(P, bd).ravel() ** 2 + eval_Q(Q, bd).ravel() ** 2
        if not np.isclose(w @ rho, 1, rtol=0, atol=1e-7):
            raise ValueError("orbital is not normalized in the native radial metric")
        density += occupation * rho
        electrons += occupation
    factors = spherical_jn(0, q[:, None] * r[None, :]) @ (w * density)
    if np.any(q == 0) and not np.allclose(
        factors[q == 0], electrons, rtol=0, atol=1e-6
    ):
        raise RuntimeError("f0(0) fails electron-count gate")
    return factors


@dataclass
class ImagingSignal:
    times_fs: np.ndarray
    q_a0_inv: np.ndarray
    mean_amplitude: np.ndarray
    mean_intensity: np.ndarray
    fluctuation_intensity: np.ndarray
    pulse_average_amplitude: np.ndarray
    pulse_average_intensity: np.ndarray
    covered_fluence_photons_per_cm2: float
    provenance: dict


class ScatteringFactorBank:
    def __init__(
        self,
        q_a0_inv,
        factors_by_state,
        *,
        photon_energy_eV,
        provenance,
        intensities_by_state=None,
        bandwidth_fwhm_eV=None,
    ):
        self.q = np.array(q_a0_inv, dtype=float, copy=True)
        if self.q.ndim != 1 or not np.isfinite(self.q).all() or np.any(self.q < 0):
            raise ValueError("q must be finite and nonnegative")
        if not np.isfinite(photon_energy_eV) or photon_energy_eV <= 0 or not provenance:
            raise ValueError("factor bank requires energy and provenance")
        self.photon_energy_eV = photon_energy_eV
        self.factors = {
            s: np.array(v, dtype=complex, copy=True)
            for s, v in factors_by_state.items()
        }
        for v in self.factors.values():
            if v.shape != self.q.shape or not np.isfinite(v).all():
                raise ValueError("state factors must match q and be finite")
            v.setflags(write=False)
        self.q.setflags(write=False)
        self.provenance = dict(provenance)
        self.bandwidth_fwhm_eV = bandwidth_fwhm_eV
        if bandwidth_fwhm_eV is not None and (
            not np.isfinite(bandwidth_fwhm_eV) or bandwidth_fwhm_eV < 0
        ):
            raise ValueError("invalid scattering-factor spectral bandwidth")
        if intensities_by_state is None:
            self.intensities = {sid: abs(f) ** 2 for sid, f in self.factors.items()}
            for v in self.intensities.values():
                v.setflags(write=False)
        else:
            if set(intensities_by_state) != set(self.factors):
                raise ValueError("spectral intensity requires every scattering state")
            self.intensities = {
                sid: np.array(v, dtype=float, copy=True)
                for sid, v in intensities_by_state.items()
            }
            for sid, v in self.intensities.items():
                if (
                    v.shape != self.q.shape
                    or not np.isfinite(v).all()
                    or np.any(
                        v < abs(self.factors[sid]) ** 2 - 1e-10 * np.maximum(1, v)
                    )
                ):
                    raise ValueError(
                        "spectral second moment violates amplitude/intensity inequality"
                    )
                v.setflags(write=False)

    @classmethod
    def from_native_results(
        cls,
        q_a0_inv,
        results_by_state,
        *,
        photon_energy_eV,
        f1_by_state,
        f2_by_state,
        provenance,
    ):
        """Compose f0+f1+i*f2; anomalous data must be supplied explicitly.

        Use explicit zero dictionaries for a documented f0-only control.
        No missing excited-state or anomalous factor falls back silently.
        """
        factors = {}
        for sid, result in results_by_state.items():
            if sid not in f1_by_state or sid not in f2_by_state:
                raise ValueError("missing anomalous factor for a transient state")
            f0 = (
                native_form_factor(result, q_a0_inv)
                if result is not None
                else np.zeros_like(q_a0_inv, dtype=float)
            )
            if result is None and sid != "":
                raise ValueError("only the bare state may lack an SCF result")
            factors[sid] = f0 + f1_by_state[sid] + 1j * f2_by_state[sid]
        return cls(
            q_a0_inv, factors, photon_energy_eV=photon_energy_eV, provenance=provenance
        )

    def observe(self, dynamics, pulse):
        if self.bandwidth_fwhm_eV is not None and not np.isclose(
            pulse.bandwidth_fwhm_eV, self.bandwidth_fwhm_eV, rtol=0, atol=1e-12
        ):
            raise ValueError("probe bandwidth differs from scattering-factor moments")
        dynamics_energy = dynamics.provenance.get("photon_energy_eV")
        if dynamics_energy is not None and not np.isclose(
            dynamics_energy, self.photon_energy_eV, rtol=0, atol=1e-9
        ):
            raise ValueError("dynamics energy differs from scattering-factor bank")
        if not np.isclose(
            pulse.photon_energy_eV, self.photon_energy_eV, rtol=0, atol=1e-9
        ):
            raise ValueError("pulse energy differs from scattering-factor bank")
        missing = set(dynamics.state_ids) - set(self.factors)
        if missing:
            raise ValueError(f"missing transient scattering states: {missing}")
        f = np.array([self.factors[s] for s in dynamics.state_ids])
        mean = dynamics.populations @ f
        intensity = dynamics.populations @ np.array(
            [self.intensities[s] for s in dynamics.state_ids]
        )
        variance = intensity - abs(mean) ** 2
        if np.min(variance) < -1e-8:
            raise RuntimeError("negative population-fluctuation intensity")
        weights = pulse.flux(dynamics.times_fs)
        fluence = float(trapezoid(weights, dynamics.times_fs) * 1e-15)
        if fluence <= 0 or not np.isclose(
            fluence, pulse.fluence_photons_per_cm2, rtol=5e-3, atol=0
        ):
            raise ValueError(
                "imaging time samples do not resolve the full photon fluence"
            )
        avg = (
            trapezoid(weights[:, None] * mean, dynamics.times_fs, axis=0)
            * 1e-15
            / fluence
        )
        avg_int = (
            trapezoid(weights[:, None] * intensity, dynamics.times_fs, axis=0)
            * 1e-15
            / fluence
        )
        return ImagingSignal(
            dynamics.times_fs.copy(),
            self.q.copy(),
            mean,
            intensity,
            variance,
            avg,
            avg_int,
            fluence,
            dict(
                self.provenance,
                convention="per-atom factor squared; mean(|f|^2), including population variance",
            ),
        )
