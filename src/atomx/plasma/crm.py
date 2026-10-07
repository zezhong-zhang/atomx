"""Basic Collisional-Radiative Model (CRM).

Solves rate equations for atomic level populations in a plasma:

    dn_i/dt = sum_j [R_{ji} * n_j] - n_i * sum_j [R_{ij}]

where R_{ij} includes:
- Radiative decay: A_{ij} (spontaneous emission, i > j)
- Collisional excitation: n_e * C_{ij}^exc (i < j)
- Collisional de-excitation: n_e * C_{ij}^deexc (i > j)
- Electron-impact ionisation: n_e * S_i
- Three-body recombination: n_e^2 * beta_i, detailed-balanced to EII
- Spontaneous radiative recombination: n_e * alpha_RR_i, the Maxwellian
  Milne inverse of this module's photoionisation cross-section
- Stimulated recombination when an explicit radiation field is present
- Photoionisation: gamma_i from a Planck or user-supplied radiation field

In steady state: R * n = 0, with normalization sum(n) = n_total.

Reference: Gu, Can. J. Phys. 86, 675 (2008) - FAC CRM module
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import linalg
from scipy.integrate import quad, solve_ivp
from scipy.special import exp1

from atomx.constants import (
    ALPHA,
    BOHR_RADIUS_M,
    BOLTZMANN_EV,
    ELECTRON_MASS_KG,
    HARTREE_EV,
    RYDBERG_EV,
)


@dataclass
class CRMLevel:
    """Atomic energy level for CRM."""
    label: str
    energy_eV: float
    g: int = 1  # statistical weight (2J+1)
    config: Optional[str] = None
    autoionization_rate_s: float = 0.0


@dataclass
class CRMTransition:
    """Transition between CRM levels.

    Collisional excitation uses, in order of preference:

    1. ``omega_E_eV``/``omega`` — a tabulated collision strength
       Omega(E_incident) (e.g. from
       :func:`atomx.scattering.distorted_wave.solver.dw_excitation_cross_section`,
       FAC-verified to ~5% rms).  The rate is the Maxwellian average of
       the effective collision strength (:func:`effective_collision_strength`).
    2. the van Regemorter formula from ``gf`` — a ~20-40% E1-only
       approximation, kept as the explicitly labelled fallback.
    """
    upper: int     # index of upper level
    lower: int     # index of lower level
    A_rate: float  # Einstein A coefficient (s^-1)
    gf: float = 0.0  # weighted oscillator strength
    omega_E_eV: Optional[np.ndarray] = None   # incident-energy grid (eV)
    omega: Optional[np.ndarray] = None         # collision strength Omega(E)


@dataclass
class CRMDRSatelliteSource:
    """Dielectronic-capture source feeding a synthetic satellite level.

    The source is added to the CRM rate matrix as
    ``initial -> upper`` with rate ``n_e * alpha(Te)``.  The synthetic
    upper level then emits through ordinary :class:`CRMTransition` rows,
    so spectrum rendering continues to use ``n_upper * A_ul``.
    """

    upper: int
    initial: int
    rate_coefficient_cm3_s: Callable[[float], float]
    label: Optional[str] = None


@dataclass
class CRMBoundFreeChannel:
    """Adjacent charge-state channel for CRM bound-free rates.

    ``bound`` is the level before ionisation and ``continuum`` is the level
    representing the adjacent charge state after one electron has been
    removed.  EII rates are cm^3/s, TBR rates are cm^6/s, and PI rates are
    s^-1 when supplied directly.
    """

    bound: int
    continuum: int
    ionization_potential_eV: float
    g_bound: Optional[float] = None
    g_continuum: Optional[float] = None
    eii_rate_cm3_s: Optional[float | Callable[[float], float]] = None
    photoionization_rate_s: Optional[float | Callable[[float], float]] = None
    sigma0_cm2: Optional[float] = None
    eii_scf_result: Optional[object] = None
    eii_method: str = "BEB"
    eii_energy_grid_eV: Optional[np.ndarray] = None
    label: Optional[str] = None


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

    def __init__(self, levels, transitions, *, Z: Optional[int] = None,
                 charge: Optional[int] = None, configs: Optional[list[str]] = None,
                 bound_free_channels: Optional[list[CRMBoundFreeChannel]] = None,
                 dr_satellite_sources: Optional[list[CRMDRSatelliteSource]] = None):
        self.levels = levels
        self.transitions = transitions
        self.n_levels = len(levels)
        self.Z = Z
        self.charge = charge
        self.configs = list(configs) if configs is not None else None
        self.bound_free_channels = (
            list(bound_free_channels) if bound_free_channels is not None else []
        )
        self.dr_satellite_sources = (
            list(dr_satellite_sources) if dr_satellite_sources is not None else []
        )

    def add_dr_satellites(
        self,
        resonances: Sequence[object],
        initial_level_idx: int,
        *,
        lower_level_by_channel: Optional[Mapping[str, int]] = None,
        initial_state_g: Optional[float] = None,
        label_prefix: str = "DR",
    ) -> list[int]:
        """Add DR satellite levels and radiative-stabilising transitions.

        Parameters
        ----------
        resonances
            Sequence of ``AutoionizationState``-like objects.  Each resonance
            must provide ``E_resonance_eV``, ``gamma_auger_per_s``,
            ``a_radiative_per_s``, ``g``, and a ``channels`` mapping of
            stabilised-final-level labels to A-rates.
        initial_level_idx
            CRM level that dielectronic capture starts from.
        lower_level_by_channel
            Optional explicit mapping from each channel label to an existing
            CRM lower-level index.  When omitted, channel labels are matched
            against ``CRMLevel.label``.
        initial_state_g
            Statistical weight of the recombining level.  Defaults to the
            selected CRM level's ``g``.
        label_prefix
            Prefix for synthetic DR level labels.

        Returns
        -------
        list[int]
            Indices of synthetic DR-populated upper levels that were added.
        """
        if initial_level_idx < 0 or initial_level_idx >= self.n_levels:
            raise IndexError("initial_level_idx is outside CRM levels")

        label_to_idx = {lev.label: i for i, lev in enumerate(self.levels)}
        channel_map = dict(lower_level_by_channel or {})
        g_i = float(initial_state_g or max(self.levels[initial_level_idx].g, 1))
        added: list[int] = []

        for i_res, resonance in enumerate(resonances):
            channels = dict(getattr(resonance, "channels", {}) or {})
            if not channels:
                channels = {
                    self.levels[initial_level_idx].label: float(
                        resonance.a_radiative_per_s
                    )
                }

            lower_rates: list[tuple[int, str, float]] = []
            for channel_label, A_rate in channels.items():
                lower = channel_map.get(channel_label, label_to_idx.get(channel_label))
                if lower is None:
                    continue
                A = max(float(A_rate), 0.0)
                if A > 0.0:
                    lower_rates.append((int(lower), str(channel_label), A))
            if not lower_rates:
                continue

            upper = self.n_levels
            label = str(getattr(resonance, "label", f"resonance_{i_res}"))
            self.levels.append(
                CRMLevel(
                    f"{label_prefix}:{label}",
                    self.levels[initial_level_idx].energy_eV
                    + float(resonance.E_resonance_eV),
                    g=max(int(round(float(resonance.g))), 1),
                    config=f"{label_prefix} satellite",
                    autoionization_rate_s=float(resonance.gamma_auger_per_s),
                )
            )
            self.n_levels += 1
            added.append(upper)

            for lower, _, A in lower_rates:
                self.transitions.append(CRMTransition(upper=upper, lower=lower, A_rate=A))

            self.dr_satellite_sources.append(
                CRMDRSatelliteSource(
                    upper=upper,
                    initial=initial_level_idx,
                    rate_coefficient_cm3_s=_dr_capture_rate_coefficient_fn(
                        resonance,
                        initial_state_g=g_i,
                    ),
                    label=label,
                )
            )

        return added

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
        if atom._csf_result is not None:
            result = atom._csf_result
            for i, E in enumerate(atom.state_energies):
                # CI is block diagonal in J and parity. The dominant CSF
                # supplies the exact block label, not an estimated degeneracy.
                dominant = int(np.argmax(np.abs(result.mixing_coefficients[:, i])))
                two_j = int(result.csfs[dominant].total_J2)
                levels.append(CRMLevel(
                    label=f"State_{i}",
                    energy_eV=E * HARTREE_EV,
                    g=two_j + 1,
                ))
        else:
            orbital_g = {sh.label: 2 * abs(sh.kappa) for sh in atom._result.subshells}
            # Single-config: use orbital energies as levels
            for label, E in sorted(atom.orbital_energies.items(),
                                    key=lambda x: x[1]):
                levels.append(CRMLevel(
                    label=label,
                    energy_eV=E * HARTREE_EV,
                    g=orbital_g[label],
                ))

        # Build transitions
        transitions = []
        label_to_idx = {lev.label: i for i, lev in enumerate(levels)}
        for tr in tr_results:
            first = label_to_idx.get(tr.initial_label)
            second = label_to_idx.get(tr.final_label)
            if first is not None and second is not None:
                i_lower, i_upper = sorted(
                    (first, second), key=lambda idx: levels[idx].energy_eV)
                if levels[i_upper].energy_eV <= levels[i_lower].energy_eV:
                    raise ValueError("radiative CRM transition must have positive energy")
                transitions.append(CRMTransition(
                    upper=i_upper,
                    lower=i_lower,
                    A_rate=tr.A_rate_per_s,
                    gf=tr.gf_value,
                ))

        return cls(
            levels,
            transitions,
            Z=getattr(atom, "Z", None),
            charge=getattr(atom, "charge", None),
            configs=getattr(atom, "_config_strings", None),
        )

    def _build_rate_matrix(
        self,
        ne_cm3,
        Te_eV,
        *,
        include_eii: bool = True,
        include_three_body: bool = True,
        include_photoionization: bool = True,
        include_radiative_recombination: bool = True,
        radiation_temperature_eV: Optional[float] = None,
        radiation_field: Optional[Callable[[float], float]] = None,
    ):
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

            # Collisional excitation: lower -> upper.  Tabulated DW
            # collision strengths (FAC-verified machinery) when present;
            # van Regemorter (E1-only, ~20-40%) as labelled fallback.
            #
            # Both directions are built from the *reduced* rate (the
            # prefactor without the Boltzmann weight of the excitation
            # direction) so that the de-excitation survives the
            # exp(-dE/kT) underflow that sets in for dE/kT >~ 745: the
            # excitation rate is then correctly ~0 while C_deexc stays
            # finite.
            dE_eV = self.levels[u].energy_eV - self.levels[l].energy_eV
            C_exc = 0.0
            C_deexc = 0.0
            if dE_eV > 0 and kT > 0:
                g_l = max(self.levels[l].g, 1)
                g_u = max(self.levels[u].g, 1)
                x = dE_eV / kT
                if tr.omega is not None and tr.omega_E_eV is not None:
                    ups = effective_collision_strength(
                        np.asarray(tr.omega_E_eV, dtype=float),
                        np.asarray(tr.omega, dtype=float), dE_eV, kT)
                    # C = 8.629e-6 / (g_l sqrt(T_K)) * Upsilon * exp(-dE/kT)
                    T_K = kT / BOLTZMANN_EV
                    C_exc_reduced = 8.6291e-6 * ups / (g_l * np.sqrt(T_K))
                    C_exc = C_exc_reduced * np.exp(-x)
                    # Detailed balance: C_deexc = C_exc (g_l/g_u) exp(+x)
                    C_deexc = C_exc_reduced * (g_l / g_u)
                elif tr.gf > 0:
                    # van Regemorter approximation for excitation rate,
                    # written so that neither exp(+x) nor exp(-x) is ever
                    # formed on its own: the effective Gaunt factor is
                    # gaunt = max(0.2, 0.276 exp(x) E1(x)) with
                    # exp(x) E1(x) -> (x+1)/(x(x+2)) for x > 1.
                    gaunt = max(0.2, 0.276 * _e1_times_exp(x))
                    # van Regemorter carries 1/dE_eV (the review found
                    # the old base missing it — rates off by exactly
                    # dE_eV, 10x at 10 eV).  Constant: 8.63e-6 cm^3/s
                    # with E in eV and T in eV via sqrt(kT).
                    base = (8.63e-6 * tr.gf
                            / (dE_eV * g_l * np.sqrt(kT)))
                    C_exc = base * gaunt * np.exp(-x)
                    C_deexc = base * gaunt * (g_l / g_u)
            if C_exc > 0.0:
                R[u, l] += ne_cm3 * C_exc
            if C_deexc > 0.0:
                R[l, u] += ne_cm3 * C_deexc

        for src in self.dr_satellite_sources:
            u = src.upper
            i = src.initial
            if u < 0 or i < 0 or u >= n or i >= n:
                raise IndexError(
                    f"DR satellite source {src.label or ''} references "
                    f"invalid levels ({i}, {u}) for {n} CRM levels"
                )
            alpha = max(float(src.rate_coefficient_cm3_s(Te_eV)), 0.0)
            if alpha > 0.0:
                R[u, i] += ne_cm3 * alpha

        for ch in self.bound_free_channels:
            b = ch.bound
            c = ch.continuum
            if b < 0 or c < 0 or b >= n or c >= n:
                raise IndexError(
                    f"bound-free channel {ch.label or ''} references "
                    f"invalid levels ({b}, {c}) for {n} CRM levels"
                )
            if ch.ionization_potential_eV <= 0.0:
                raise ValueError("ionization_potential_eV must be positive")

            if include_eii:
                eii = _channel_eii_rate_cm3_s(ch, Te_eV)
                if eii > 0.0:
                    R[c, b] += ne_cm3 * eii

                    if include_three_body:
                        beta = three_body_recombination_rate_cm6_s(
                            eii,
                            ch.ionization_potential_eV,
                            Te_eV,
                            g_bound=ch.g_bound or self.levels[b].g,
                            g_continuum=ch.g_continuum or self.levels[c].g,
                        )
                        R[b, c] += ne_cm3 * ne_cm3 * beta

            if include_radiative_recombination:
                # Spontaneous radiative recombination (Milne inverse of the
                # module's own photoionisation cross-section).  Unlike
                # three-body recombination this channel survives in the
                # coronal limit (rate n_e alpha_RR, not n_e^2 beta), so it
                # is what makes a coronal ionisation balance reachable and
                # what cancels a Planck radiation field at T_rad = T_e.
                alpha_rr = radiative_recombination_rate_cm3_s(
                    ch.ionization_potential_eV,
                    Te_eV,
                    g_bound=ch.g_bound or self.levels[b].g,
                    g_continuum=ch.g_continuum or self.levels[c].g,
                    sigma0_cm2=ch.sigma0_cm2,
                    radiation_temperature_eV=radiation_temperature_eV,
                    radiation_field=radiation_field,
                )
                if alpha_rr > 0.0:
                    R[b, c] += ne_cm3 * alpha_rr

            if include_photoionization:
                pi_rate = _channel_photoionization_rate_s(
                    ch,
                    radiation_temperature_eV=radiation_temperature_eV,
                    radiation_field=radiation_field,
                )
                if pi_rate > 0.0:
                    R[c, b] += pi_rate

        # Diagonal: loss rate
        for i in range(n):
            R[i, i] = -np.sum(R[:, i]) + R[i, i]  # subtract off-diagonal sum

        # Fix diagonal
        for i in range(n):
            R[i, i] = 0.0
            R[i, i] = -np.sum(R[:, i])

        return R

    def steady_state(self, ne_cm3, Te_eV, **rate_options):
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
        R = self._build_rate_matrix(ne_cm3, Te_eV, **rate_options)
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

    def time_dependent(self, ne_cm3, Te_eV, t_span, n0=None, **rate_options):
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
        R = self._build_rate_matrix(ne_cm3, Te_eV, **rate_options)

        if n0 is None:
            n0 = np.zeros(self.n_levels)
            n0[0] = 1.0  # ground state

        def rhs(t, y):
            return R @ y

        sol = solve_ivp(rhs, t_span, n0, method='Radau', dense_output=True,
                        rtol=1e-8, atol=1e-12)

        return sol.t, sol.y

    def spectrum(self, ne_cm3, Te_eV, wavelength_min=1.0, wavelength_max=10000.0,
                 n_points=10000, **rate_options):
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

        populations = self.steady_state(ne_cm3, Te_eV, **rate_options)
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


_ELECTRON_MASS_G = ELECTRON_MASS_KG * 1000.0
_BOLTZMANN_ERG_PER_EV = 1.602_176_634e-12
_PLANCK_ERG_S = 6.626_070_15e-27
_LIGHT_CM_S = 2.997_924_58e10
_EV_TO_ERG = 1.602_176_634e-12


def saha_ionization_density_cm3(
    ionization_potential_eV: float,
    Te_eV: float,
    *,
    g_bound: float = 1.0,
    g_continuum: float = 1.0,
) -> float:
    """Return ``n_{q+1} n_e / n_q`` in cm^-3 from the Saha relation."""
    if Te_eV <= 0.0:
        return 0.0
    g_bound = max(float(g_bound), 1e-300)
    g_continuum = max(float(g_continuum), 1e-300)
    kT_erg = Te_eV * _BOLTZMANN_ERG_PER_EV
    lambda_cm = _PLANCK_ERG_S / np.sqrt(2.0 * np.pi * _ELECTRON_MASS_G * kT_erg)
    prefactor = 2.0 / lambda_cm ** 3
    return float(
        prefactor
        * (g_continuum / g_bound)
        * np.exp(-ionization_potential_eV / Te_eV)
    )


def three_body_recombination_rate_cm6_s(
    eii_rate_cm3_s: float,
    ionization_potential_eV: float,
    Te_eV: float,
    *,
    g_bound: float = 1.0,
    g_continuum: float = 1.0,
) -> float:
    """Detailed-balance inverse of EII using the Saha relation."""
    saha = saha_ionization_density_cm3(
        ionization_potential_eV,
        Te_eV,
        g_bound=g_bound,
        g_continuum=g_continuum,
    )
    if saha <= 0.0:
        return 0.0
    return max(float(eii_rate_cm3_s), 0.0) / saha


def radiative_recombination_rate_cm3_s(
    ionization_potential_eV: float,
    Te_eV: float,
    *,
    g_bound: float = 1.0,
    g_continuum: float = 1.0,
    sigma0_cm2: Optional[float] = None,
    radiation_temperature_eV: Optional[float] = None,
    radiation_field: Optional[Callable[[float], float]] = None,
) -> float:
    """Maxwellian Milne RR coefficient (cm³/s), spontaneous by default.

    For the Kramers PI cross section the spontaneous integral reduces to
    ``exp(x) E1(x)``, x=I/T. It has no activation Boltzmann factor and
    remains finite when the Saha density underflows. Stimulated capture
    multiplies the integrand by the photon occupation, only when a field
    is supplied. ``radiation_field`` is the angle-integrated photon number
    flux in photons cm⁻² s⁻¹ eV⁻¹, as in the PI channel; it takes precedence
    over ``radiation_temperature_eV``. With a Planck field at T_rad=T_e,
    spontaneous + stimulated RR obey Saha detailed balance with PI.
    """
    if Te_eV <= 0.0 or ionization_potential_eV <= 0.0:
        return 0.0
    if g_bound <= 0 or g_continuum <= 0:
        raise ValueError("statistical weights must be positive")
    I, T = float(ionization_potential_eV), float(Te_eV)
    sigma0 = _hydrogenic_sigma0_cm2(I) if sigma0_cm2 is None else float(sigma0_cm2)
    x = I / T
    if x <= 50.0:
        integral = float(np.exp(x) * exp1(x))
    else:
        # Alternating asymptotic expansion; x>=50 keeps all retained
        # terms decreasing, including far below the Saha underflow limit.
        term = 1.0 / x
        integral = term
        for n in range(1, 18):
            term *= -n / x
            integral += term
            if abs(term) < 2e-16 * abs(integral):
                break
    if radiation_field is not None or (
            radiation_temperature_eV is not None and radiation_temperature_eV > 0):
        def stimulated(y):
            omega = I * np.exp(y)
            if radiation_field is not None:
                flux = max(float(radiation_field(float(omega))), 0.0)
                occupation = flux / _photon_flux_prefactor(omega)
            else:
                z = omega / radiation_temperature_eV
                occupation = np.exp(-z) / (-np.expm1(-z))
            return np.exp(-x * np.expm1(y)) * occupation

        integral += quad(stimulated, 0.0, np.log1p(50.0 / x),
                         epsabs=1e-13, epsrel=2e-11)[0]
    me, ev, c = _ELECTRON_MASS_G, _EV_TO_ERG, _LIGHT_CM_S
    prefactor = (np.sqrt(8.0 / (np.pi * me)) * (T * ev) ** -1.5 * ev**2
                 * (g_bound / g_continuum) * sigma0 * I**3 * ev / (2 * me * c**2))
    return float(prefactor * integral)


def _photon_flux_prefactor(energy_eV):
    """Angle-integrated photon number flux per unit occupation and per eV."""
    return 8 * np.pi * energy_eV**2 * _EV_TO_ERG**3 / (_PLANCK_ERG_S**3 * _LIGHT_CM_S**2)


def _dr_capture_rate_coefficient_fn(resonance, *, initial_state_g: float):
    from atomx.scattering.dielectronic_recombination.solver import dr_rate_coefficient

    def alpha(Te_eV: float) -> float:
        if Te_eV <= 0.0:
            return 0.0
        return dr_rate_coefficient(
            [resonance],
            T_eV=Te_eV,
            initial_state_g=initial_state_g,
        ).alpha_cm3_s[0]

    return alpha


def _channel_eii_rate_cm3_s(ch: CRMBoundFreeChannel, Te_eV: float) -> float:
    if ch.eii_rate_cm3_s is not None:
        value = (
            ch.eii_rate_cm3_s(Te_eV)
            if callable(ch.eii_rate_cm3_s) else ch.eii_rate_cm3_s
        )
        return max(float(value), 0.0)
    if ch.eii_scf_result is None or Te_eV <= 0.0:
        return 0.0
    threshold = ch.ionization_potential_eV
    energies = ch.eii_energy_grid_eV
    if energies is None:
        energies = np.geomspace(
            threshold * (1.0 + 1e-6),
            max(50.0 * Te_eV, 20.0 * threshold),
            160,
        )
    from atomx.scattering.electron_ionization import (
        eii_cross_section,
        eii_rate_coefficient,
    )
    xs = eii_cross_section(ch.eii_scf_result, energies, method=ch.eii_method)
    return max(float(eii_rate_coefficient(xs, Te_eV)), 0.0)


def _channel_photoionization_rate_s(
    ch: CRMBoundFreeChannel,
    *,
    radiation_temperature_eV: Optional[float],
    radiation_field: Optional[Callable[[float], float]],
) -> float:
    if ch.photoionization_rate_s is not None:
        arg = radiation_temperature_eV if radiation_temperature_eV is not None else 0.0
        value = (
            ch.photoionization_rate_s(arg)
            if callable(ch.photoionization_rate_s) else ch.photoionization_rate_s
        )
        return max(float(value), 0.0)
    if radiation_field is not None:
        threshold = ch.ionization_potential_eV
        sigma0 = (_hydrogenic_sigma0_cm2(threshold) if ch.sigma0_cm2 is None else ch.sigma0_cm2)

        def integrand(t):
            # t=I/E maps the entire physical spectrum to (0,1], so
            # custom fields are not silently cut off at E=100 I.
            return t * max(float(radiation_field(float(threshold / t))), 0.0)

        return float(sigma0 * threshold * quad(integrand, 0., 1., epsabs=0., epsrel=2e-10)[0])
    if radiation_temperature_eV is None or radiation_temperature_eV <= 0.0:
        return 0.0
    return _planck_photoionization_rate_s(
        ch.ionization_potential_eV,
        radiation_temperature_eV,
        sigma0_cm2=ch.sigma0_cm2,
    )


def _hydrogenic_sigma0_cm2(ionization_potential_eV: float) -> float:
    sigma_h = (512 * np.pi**2 / 3) * ALPHA * (100 * BOHR_RADIUS_M)**2 * np.exp(-4)
    return sigma_h * (RYDBERG_EV / ionization_potential_eV)


def _kramers_photoionization_sigma_cm2(
    energies_eV,
    threshold_eV: float,
    sigma0_cm2: float,
):
    energies = np.asarray(energies_eV, dtype=float)
    sigma = np.zeros_like(energies)
    mask = energies >= threshold_eV
    sigma[mask] = sigma0_cm2 * (threshold_eV / energies[mask]) ** 3
    return sigma


def _planck_photoionization_rate_s(
    threshold_eV: float,
    radiation_temperature_eV: float,
    *,
    sigma0_cm2: Optional[float] = None,
) -> float:
    if radiation_temperature_eV <= 0 or threshold_eV <= 0:
        return 0.0
    sigma0 = (_hydrogenic_sigma0_cm2(threshold_eV)
              if sigma0_cm2 is None else sigma0_cm2)
    x = threshold_eV / radiation_temperature_eV
    # y=log(omega/I). The threshold is resolved at every I/T, and
    # exp(-I/T) underflows only when the physical PI rate does.
    def kernel(y):
        z = x * np.exp(y)
        return np.exp(-x * np.expm1(y)) / (-np.expm1(-z))
    integral = quad(kernel, 0.0, np.log1p(50.0 / x),
                    epsabs=1e-13, epsrel=2e-11)[0]
    return float(sigma0 * threshold_eV * _photon_flux_prefactor(threshold_eV)
                 * np.exp(-x) * integral)


def effective_collision_strength(E_eV, omega, dE_eV, Te_eV):
    """Maxwellian effective collision strength Upsilon(T_e).

    Upsilon = Int_0^inf Omega(E_f) exp(-E_f/kT) d(E_f/kT), with E_f the
    FINAL (scattered) electron energy (Burgess & Tully 1992 convention),
    so that

        C_exc = 8.629e-6 / (g_l sqrt(T_K)) * Upsilon * exp(-dE/kT).

    ``E_eV``/``omega`` tabulate Omega against the INCIDENT energy.  On
    each grid interval Omega is taken linear and the product with
    exp(-x) integrated in closed form (exact for piecewise-linear
    Omega — a constant Omega = c gives Upsilon = c to machine
    precision), plus an Omega = const tail beyond the last point (for a
    Bethe ln E growth the tail is slightly underestimated — extend the
    table if the high-T behaviour matters).
    """
    E = np.asarray(E_eV, dtype=float)
    om = np.asarray(omega, dtype=float)
    if len(E) == 0 or Te_eV <= 0:
        return 0.0
    Ef = E - float(dE_eV)
    keep = Ef >= 0.0
    Ef = Ef[keep]
    om = om[keep]
    if len(Ef) == 0:
        return 0.0
    x = Ef / float(Te_eV)
    # anchor the threshold point E_f = 0 if the table starts above it
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        om = np.concatenate([[om[0]], om])
    # Int (a + b t) e^-t on [x0, x1], Omega linear per interval:
    #   = [(a + b(1 + t)) e^-t]_{x1}^{x0}
    x0, x1 = x[:-1], x[1:]
    o0, o1 = om[:-1], om[1:]
    dx = x1 - x0
    good = dx > 0
    b = np.zeros_like(dx)
    b[good] = (o1[good] - o0[good]) / dx[good]
    a = o0 - b * x0
    def F(t):
        return (a + b * (1.0 + t)) * np.exp(-t)

    ups = float(np.sum((F(x0) - F(x1))[good]))
    # constant-Omega tail beyond the table
    ups += float(om[-1] * np.exp(-x[-1]))
    return ups


def _E1_integral(x):
    """Exponential integral E1(x) approximation for van Regemorter formula."""
    if x <= 0:
        return 0.0
    if x < 1:
        return -np.log(x) - 0.5772 + x - x ** 2 / 4
    else:
        return np.exp(-x) / x * (x + 1) / (x + 2)


def _e1_times_exp(x):
    """``exp(x) * E1(x)`` — the effective-Gaunt combination, overflow free.

    The van Regemorter effective Gaunt factor is
    ``0.276 * exp(x) * E1(x)``.  Evaluating the two factors separately
    overflows for ``x >~ 709`` (and gives ``inf * 0 = nan``), so for
    ``x >= 1`` the product is used directly:
    ``exp(x) E1(x) = (x + 1) / (x (x + 2))``, which decays like ``1/x``.
    """
    if not np.isfinite(x) or x <= 0:
        return 0.0
    if x < 1:
        return float(np.exp(x) * _E1_integral(x))
    return (x + 1.0) / (x * (x + 2.0))


@dataclass(frozen=True)
class ChargeStateRate:
    """Ionisation/recombination rate data for one adjacent charge pair."""

    charge: int
    ionization_cm3_s: float
    recombination_cm3_s: float


def coronal_charge_balance(
    charges: list[int],
    Te_eV: float,
    ionization_rate: Callable[[int, float], float],
    recombination_rate: Callable[[int, float], float],
) -> np.ndarray:
    """Solve a coronal charge-state balance for adjacent ion stages.

    The steady coronal relation is
    ``n_q S_q = n_{q+1} alpha_{q+1}``, so the population vector is built
    recursively from a reference abundance for the lowest charge state.
    ``charges`` must be contiguous and sorted.
    """
    if not charges:
        raise ValueError("charges must not be empty")
    if charges != list(range(charges[0], charges[-1] + 1)):
        raise ValueError("charges must be contiguous")
    if Te_eV <= 0:
        raise ValueError("Te_eV must be positive")

    abund = np.ones(len(charges), dtype=float)
    for i, q in enumerate(charges[:-1]):
        s_q = max(float(ionization_rate(q, Te_eV)), 0.0)
        alpha_next = max(float(recombination_rate(q + 1, Te_eV)), 1e-300)
        abund[i + 1] = abund[i] * s_q / alpha_next
    total = abund.sum()
    if total <= 0:
        raise ValueError("charge balance produced zero total abundance")
    return abund / total
