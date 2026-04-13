"""Basic Collisional-Radiative Model (CRM).

Solves rate equations for atomic level populations in a plasma:

    dn_i/dt = sum_j [R_{ji} * n_j] - n_i * sum_j [R_{ij}]

where R_{ij} includes:
- Radiative decay: A_{ij} (spontaneous emission, i > j)
- Collisional excitation: n_e * C_{ij}^exc (i < j)
- Collisional de-excitation: n_e * C_{ij}^deexc (i > j)

In steady state: R * n = 0, with normalization sum(n) = n_total.

Reference: Gu, Can. J. Phys. 86, 675 (2008) - FAC CRM module
"""

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Optional

from atomx.constants import HARTREE_EV, BOLTZMANN_EV


@dataclass
class CRMLevel:
    """Atomic energy level for CRM."""
    label: str
    energy_eV: float
    g: int = 1  # statistical weight (2J+1)


@dataclass
class CRMTransition:
    """Transition between CRM levels."""
    upper: int     # index of upper level
    lower: int     # index of lower level
    A_rate: float  # Einstein A coefficient (s^-1)
    gf: float = 0.0  # weighted oscillator strength


class CollisionalRadiativeModel:
    """Collisional-Radiative Model for plasma level populations.

    Supports steady-state and time-dependent solutions.

    Parameters
    ----------
    levels : list of CRMLevel
        Energy levels (sorted by energy).
    transitions : list of CRMTransition
        Radiative transitions with A-rates.
    """

    def __init__(self, levels, transitions):
        self.levels = levels
        self.transitions = transitions
        self.n_levels = len(levels)

    @classmethod
    def from_atom(cls, atom, multipoles=None):
        """Build CRM from a solved Atom with transition data.

        Parameters
        ----------
        atom : Atom
            Solved atom (preferably MCDHF with multiple states).
        multipoles : list of str
            Multipole types for transition calculation.

        Returns
        -------
        CollisionalRadiativeModel
        """
        if multipoles is None:
            multipoles = ['E1']

        tr_results = atom.transitions(multipoles)

        # Build levels from orbital energies or CI states
        levels = []
        if atom._mcdhf_result is not None:
            for i, E in enumerate(atom.state_energies):
                levels.append(CRMLevel(
                    label=f"State_{i}",
                    energy_eV=E * HARTREE_EV,
                    g=1,  # placeholder
                ))
        else:
            # Single-config: use orbital energies as levels
            for label, E in sorted(atom.orbital_energies.items(),
                                    key=lambda x: x[1]):
                levels.append(CRMLevel(
                    label=label,
                    energy_eV=E * HARTREE_EV,
                    g=1,
                ))

        # Build transitions
        transitions = []
        label_to_idx = {lev.label: i for i, lev in enumerate(levels)}
        for tr in tr_results:
            i_upper = label_to_idx.get(tr.initial_label)
            i_lower = label_to_idx.get(tr.final_label)
            if i_upper is not None and i_lower is not None:
                transitions.append(CRMTransition(
                    upper=i_upper,
                    lower=i_lower,
                    A_rate=tr.A_rate_per_s,
                    gf=tr.gf_value,
                ))

        return cls(levels, transitions)

    def _build_rate_matrix(self, ne_cm3, Te_eV):
        """Build the rate matrix R.

        R[j, i] = rate from level i to level j.
        Diagonal: R[i, i] = -sum_j R[j, i].

        Parameters
        ----------
        ne_cm3 : float
            Electron density in cm^-3.
        Te_eV : float
            Electron temperature in eV.

        Returns
        -------
        ndarray, shape (n_levels, n_levels)
        """
        n = self.n_levels
        R = np.zeros((n, n))
        kT = Te_eV

        for tr in self.transitions:
            u = tr.upper
            l = tr.lower

            # Radiative decay: upper -> lower
            R[l, u] += tr.A_rate

            # Collisional excitation: lower -> upper
            # Using van Regemorter formula: C_exc ~ 1.6e-5 * gf / (g_l * sqrt(T))
            #                                * exp(-dE/kT) * gaunt_factor
            dE_eV = self.levels[u].energy_eV - self.levels[l].energy_eV
            if dE_eV > 0 and kT > 0 and tr.gf > 0:
                g_l = max(self.levels[l].g, 1)
                # van Regemorter approximation for excitation rate
                gaunt = max(0.2, 0.276 * np.exp(dE_eV / kT)
                            * _E1_integral(dE_eV / kT))
                C_exc = (1.578e-5 * tr.gf * gaunt
                         / (g_l * np.sqrt(kT)) * np.exp(-dE_eV / kT))
                R[u, l] += ne_cm3 * C_exc

                # Collisional de-excitation (detailed balance)
                g_u = max(self.levels[u].g, 1)
                C_deexc = C_exc * (g_l / g_u) * np.exp(dE_eV / kT)
                R[l, u] += ne_cm3 * C_deexc

        # Diagonal: loss rate
        for i in range(n):
            R[i, i] = -np.sum(R[:, i]) + R[i, i]  # subtract off-diagonal sum

        # Fix diagonal
        for i in range(n):
            R[i, i] = 0.0
            R[i, i] = -np.sum(R[:, i])

        return R

    def steady_state(self, ne_cm3, Te_eV):
        """Solve for steady-state level populations.

        Solves R * n = 0 with normalization sum(n) = 1.

        Parameters
        ----------
        ne_cm3, Te_eV : float

        Returns
        -------
        populations : ndarray
            Fractional population of each level.
        """
        R = self._build_rate_matrix(ne_cm3, Te_eV)
        n = self.n_levels

        # Replace last equation with normalization: sum(n) = 1
        A = R.copy()
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0

        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", linalg.LinAlgWarning)
                populations = linalg.solve(A, b)
        except linalg.LinAlgError:
            # Fallback: use least squares for singular/near-singular matrices
            populations, _, _, _ = linalg.lstsq(A, b)

        # Ensure non-negative
        populations = np.maximum(populations, 0.0)
        total = populations.sum()
        if total > 0:
            populations /= total

        return populations

    def time_dependent(self, ne_cm3, Te_eV, t_span, n0=None):
        """Solve time-dependent rate equations.

        dn/dt = R * n

        Parameters
        ----------
        ne_cm3, Te_eV : float
        t_span : tuple of float
            (t_start, t_end) in seconds.
        n0 : ndarray, optional
            Initial populations. Default: all in ground state.

        Returns
        -------
        t : ndarray
            Time points.
        populations : ndarray, shape (n_levels, n_times)
        """
        R = self._build_rate_matrix(ne_cm3, Te_eV)

        if n0 is None:
            n0 = np.zeros(self.n_levels)
            n0[0] = 1.0  # ground state

        def rhs(t, y):
            return R @ y

        sol = solve_ivp(rhs, t_span, n0, method='Radau', dense_output=True,
                        rtol=1e-8, atol=1e-12)

        return sol.t, sol.y

    def spectrum(self, ne_cm3, Te_eV, wavelength_min=1.0, wavelength_max=10000.0,
                 n_points=10000):
        """Compute synthetic emission spectrum.

        Parameters
        ----------
        ne_cm3, Te_eV : float
        wavelength_min, wavelength_max : float
            Wavelength range in Angstroms.
        n_points : int

        Returns
        -------
        wavelengths : ndarray
            Wavelength grid in Angstroms.
        intensities : ndarray
            Emission intensities (arbitrary units).
        """
        from atomx.constants import HARTREE_INV_ANGSTROM

        populations = self.steady_state(ne_cm3, Te_eV)
        wavelengths = np.linspace(wavelength_min, wavelength_max, n_points)
        intensities = np.zeros_like(wavelengths)

        for tr in self.transitions:
            n_upper = populations[tr.upper]
            if n_upper <= 0 or tr.A_rate <= 0:
                continue

            dE_eV = self.levels[tr.upper].energy_eV - self.levels[tr.lower].energy_eV
            if dE_eV <= 0:
                continue

            # Wavelength of this transition
            dE_hartree = dE_eV / HARTREE_EV
            lam = 1.0 / (HARTREE_INV_ANGSTROM * dE_hartree)

            if lam < wavelength_min or lam > wavelength_max:
                continue

            # Gaussian line profile (thermal broadening)
            # FWHM ~ lambda * sqrt(kT / (m_ion * c^2))
            fwhm = lam * np.sqrt(Te_eV / 931.5e6) * 2.355  # approximate
            fwhm = max(fwhm, (wavelength_max - wavelength_min) / n_points * 2)
            sigma = fwhm / 2.355

            intensity = n_upper * tr.A_rate
            profile = intensity * np.exp(-0.5 * ((wavelengths - lam) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            intensities += profile

        return wavelengths, intensities

    def boltzmann_populations(self, Te_eV):
        """LTE Boltzmann distribution for validation.

        n_i / n_total = g_i * exp(-E_i/kT) / Z

        Parameters
        ----------
        Te_eV : float

        Returns
        -------
        ndarray
        """
        kT = Te_eV
        if kT <= 0:
            pop = np.zeros(self.n_levels)
            pop[0] = 1.0
            return pop

        E_ref = self.levels[0].energy_eV
        weights = np.array([
            lev.g * np.exp(-(lev.energy_eV - E_ref) / kT)
            for lev in self.levels
        ])
        return weights / weights.sum()


def _E1_integral(x):
    """Exponential integral E1(x) approximation for van Regemorter formula."""
    if x <= 0:
        return 0.0
    if x < 1:
        return -np.log(x) - 0.5772 + x - x ** 2 / 4
    else:
        return np.exp(-x) / x * (x + 1) / (x + 2)
