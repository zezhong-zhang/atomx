"""High-level transition calculator bridging structure to spectroscopy.

Computes all allowed transitions between atomic states (single-config
or MCDHF) with proper angular coupling and radial matrix elements.

Usage:
    from atomx import Atom
    he = Atom(Z=2, configs=["1s2", "1s1 2p1"]).solve()
    results = he.transitions(multipoles=['E1', 'M1'])
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from atomx.constants import (
    ALPHA, HARTREE_EV, HARTREE_CM1, BOHR_RADIUS_M, HARTREE_INV_ANGSTROM,
)
from atomx.prop.transitions.multipole import radial_eL, radial_eL_velocity, radial_mL
from atomx.prop.transitions.angular import (
    angular_eL, angular_mL, selection_rules, allowed_final_kappas,
    line_strength_factor, _kappa_to_lj,
)
from atomx.prop.transitions.rates import (
    einstein_A_eL, einstein_A_mL, oscillator_strength, lifetime,
)
from atomx.angular.csf import (
    csf_has_explicit_j,
    csf_parity,
    csf_twice_j,
    multipole_allows_state_transition,
    one_body_csf_coupling,
)


@dataclass
class TransitionResult:
    """Result for a single radiative transition."""
    initial_label: str
    final_label: str
    energy_diff_hartree: float
    wavelength_angstrom: float
    multipole: str           # 'E1', 'M1', 'E2', 'M2'
    A_rate_per_s: float      # Einstein A coefficient
    gf_value: float          # weighted oscillator strength
    f_value: float           # absorption oscillator strength
    line_strength: float     # in atomic units
    gauge: str = "length"    # 'length' or 'velocity'

    @property
    def wavelength_nm(self):
        return self.wavelength_angstrom / 10.0

    @property
    def energy_eV(self):
        return abs(self.energy_diff_hartree) * HARTREE_EV

    @property
    def energy_cm1(self):
        return abs(self.energy_diff_hartree) * HARTREE_CM1


def _energy_to_wavelength_angstrom(dE_hartree):
    """Convert energy difference (Hartree) to wavelength (Angstrom)."""
    if abs(dE_hartree) < 1e-15:
        return np.inf
    return 1.0 / (HARTREE_INV_ANGSTROM * abs(dE_hartree))


def _subshell_label(n, kappa):
    """Human-readable subshell label from (n, kappa)."""
    l = kappa if kappa > 0 else -kappa - 1
    tj = 2 * abs(kappa) - 1
    l_chars = "spdfghiklmno"
    l_char = l_chars[l] if l < len(l_chars) else f"l{l}"
    return f"{n}{l_char}{tj}/2"


def _csf_twice_J(csf):
    """Infer 2·J_total for a CSF from its subshell structure.

    Accurate for CSFs with zero or one partially-filled relativistic
    subshell (the common case enumerated by
    :func:`atomx.methods.mcdhf.config.enumerate_jj_csfs`):

    * closed CSF → J = 0 (returns 0)
    * one electron or one hole in ``(n, kappa)`` → J = j_kappa (returns
      ``2j = 2|kappa|-1``)

    Multiple open shells are reported as J = j of the single most-open
    shell; this is approximate but better than the previous hard-coded
    ``tj=1`` placeholder. If a CSF carries an explicit non-zero
    ``total_J2`` attribute that takes precedence.
    """
    # Honour explicit total_J2 if somebody populated it.
    explicit = int(getattr(csf, "total_J2", 0))
    if explicit > 0:
        return explicit

    open_shells = []
    for sh in csf.subshells:
        j2 = 2 * abs(sh.kappa) - 1  # twice j
        max_occ = j2 + 1
        if 1e-6 < sh.occupation < max_occ - 1e-6:
            open_shells.append((sh, j2))
    if not open_shells:
        return 0
    # Most-open shell: take the one furthest from half-filled.
    # For one electron or one hole, result = j of that shell.
    sh, j2 = min(
        open_shells,
        key=lambda s: abs(s[0].occupation - (s[1] + 1) / 2.0),
    )
    return int(j2)


class TransitionCalculator:
    """Compute radiative transitions for solved atomic states.

    Supports:
    - Single-configuration: transitions between different subshells
    - MCDHF: transitions between CI eigenstates
    """

    def __init__(self, atom):
        """Initialize from a solved Atom object.

        Parameters
        ----------
        atom : Atom
            Must be solved (atom.solve() called).
        """
        if not atom.converged:
            raise ValueError("Atom must be solved and converged before computing transitions.")
        self.atom = atom
        self.Z = atom.Z

    def compute_all(self, multipoles=None, gauge='length'):
        """Compute all allowed transitions.

        Parameters
        ----------
        multipoles : list of str, optional
            Multipole types to compute. Default: ['E1'].
        gauge : str
            'length' (Babushkin) or 'velocity' (Coulomb).

        Returns
        -------
        list of TransitionResult
        """
        if multipoles is None:
            multipoles = ['E1']

        if self.atom._mcdhf_result is not None:
            return self._compute_mcdhf_transitions(multipoles, gauge)
        else:
            return self._compute_single_config_transitions(multipoles, gauge)

    def _compute_single_config_transitions(self, multipoles, gauge):
        """Transitions for single-configuration (DHF/DFS) calculation.

        For single-config, we compute transitions between orbitals
        as if one electron is promoted from orbital i to orbital f.
        This gives orbital-to-orbital transition matrix elements.
        """
        result = self.atom._result
        bsp_data = self.atom._bsp_data
        results = []

        # Build list of orbitals: (label, n, kappa, P, Q, energy, occ)
        orbitals = []
        kappa_counter = {}
        for sh in result.subshells:
            idx = kappa_counter.get(sh.kappa, 0)
            kappa_counter[sh.kappa] = idx + 1
            orb_list = result.orbitals_raw.get(sh.kappa, [])
            if idx < len(orb_list):
                P, Q, occ = orb_list[idx]
                E = result.orbital_energies.get(sh.label, 0.0)
                orbitals.append((sh.label, sh.n, sh.kappa, P, Q, E, occ))

        # Compute transitions between all pairs
        for i, (lab_i, n_i, k_i, P_i, Q_i, E_i, occ_i) in enumerate(orbitals):
            for j, (lab_f, n_f, k_f, P_f, Q_f, E_f, occ_f) in enumerate(orbitals):
                if i == j:
                    continue  # no self-transition

                dE = E_f - E_i
                if dE <= 0:
                    continue  # only emission (upper -> lower)

                omega = abs(dE)
                wavelength = _energy_to_wavelength_angstrom(dE)
                _, tj_i = _kappa_to_lj(k_i)
                _, tj_f = _kappa_to_lj(k_f)

                for mp in multipoles:
                    mp_type = mp[0]  # 'E' or 'M'
                    L = int(mp[1:])

                    if not selection_rules(k_i, k_f, L, mp_type):
                        continue

                    # Radial matrix element
                    if mp_type == 'E':
                        if gauge == 'velocity':
                            rme = radial_eL_velocity(
                                P_i, Q_i, P_f, Q_f, bsp_data, L,
                                kappa_i=k_i, kappa_f=k_f)
                        else:
                            rme = radial_eL(P_i, Q_i, P_f, Q_f, bsp_data, L)
                        ang = angular_eL(k_i, k_f, L)
                    else:
                        rme = radial_mL(P_i, Q_i, P_f, Q_f, bsp_data, L)
                        ang = angular_mL(k_i, k_f, L)

                    S = ang ** 2 * rme ** 2  # line strength

                    if S < 1e-30:
                        continue

                    # A-coefficient
                    if mp_type == 'E':
                        A = einstein_A_eL(omega, S, L, tj_i)
                    else:
                        A = einstein_A_mL(omega, S, L, tj_i)

                    # Oscillator strength
                    f_val, gf_val = oscillator_strength(omega, S, L, tj_i, tj_f)

                    results.append(TransitionResult(
                        initial_label=lab_i,
                        final_label=lab_f,
                        energy_diff_hartree=dE,
                        wavelength_angstrom=wavelength,
                        multipole=mp,
                        A_rate_per_s=A,
                        gf_value=gf_val,
                        f_value=f_val,
                        line_strength=S,
                        gauge=gauge,
                    ))

        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _compute_mcdhf_transitions(self, multipoles, gauge):
        """Transitions between MCDHF CI eigenstates.

        For MCDHF, the transition matrix element between states |I> and |F> is:
            <F|T^L|I> = sum_{r,s} c_r^F * c_s^I * <CSF_r|T^L|CSF_s>

        where c_r^F are the CI mixing coefficients.
        """
        mcdhf = self.atom._mcdhf_result
        bsp_data = mcdhf.bsp_data
        csfs = mcdhf.csfs
        mixing = mcdhf.mixing_coefficients
        state_energies = mcdhf.state_energies
        orbitals_dict = mcdhf.orbitals_dict
        n_states = len(state_energies)
        results = []

        for I in range(n_states):
            for F in range(n_states):
                if I == F:
                    continue

                dE = state_energies[F] - state_energies[I]
                if dE <= 0:
                    continue

                omega = abs(dE)
                wavelength = _energy_to_wavelength_angstrom(dE)

                for mp in multipoles:
                    mp_type = mp[0]
                    L = int(mp[1:])

                    # Infer 2J/parity of each state from its dominant CSF and
                    # apply the many-electron tensor selection rule before the
                    # expensive CSF-pair accumulation.
                    dom_I = int(np.argmax(np.abs(mixing[:, I])))
                    dom_F = int(np.argmax(np.abs(mixing[:, F])))
                    explicit_I = csf_has_explicit_j(csfs[dom_I])
                    explicit_F = csf_has_explicit_j(csfs[dom_F])
                    if explicit_I and explicit_F:
                        tj_I = csf_twice_j(csfs[dom_I])
                        tj_F = csf_twice_j(csfs[dom_F])
                        pi_I = csf_parity(csfs[dom_I])
                        pi_F = csf_parity(csfs[dom_F])
                        if not multipole_allows_state_transition(
                            tj_I, pi_I, tj_F, pi_F, L, mp_type
                        ):
                            continue
                    else:
                        tj_I = _csf_twice_J(csfs[dom_I])
                        tj_F = _csf_twice_J(csfs[dom_F])

                    # Sum matrix element over CSF pairs
                    total_me = 0.0
                    for r, csf_r in enumerate(csfs):
                        for s, csf_s in enumerate(csfs):
                            c_F = mixing[r, F]
                            c_I = mixing[s, I]
                            if abs(c_F * c_I) < 1e-12:
                                continue

                            # Single-particle ME between CSF_r and CSF_s
                            me = self._csf_transition_me(
                                csf_r, csf_s, orbitals_dict, bsp_data,
                                L, mp_type, gauge)
                            total_me += c_F * c_I * me

                    S = total_me ** 2  # line strength

                    if S < 1e-30:
                        continue

                    # ``dE > 0`` means I is the lower state and F is the upper
                    # state.  The Einstein spontaneous rate is for F -> I, so
                    # the statistical denominator is the upper-state 2J+1.
                    tj_upper = tj_F
                    if mp_type == 'E':
                        A = einstein_A_eL(omega, S, L, tj_upper)
                    else:
                        A = einstein_A_mL(omega, S, L, tj_upper)

                    f_val, gf_val = oscillator_strength(omega, S, L, tj_I, tj_F)

                    results.append(TransitionResult(
                        initial_label=f"State_{I}",
                        final_label=f"State_{F}",
                        energy_diff_hartree=dE,
                        wavelength_angstrom=wavelength,
                        multipole=mp,
                        A_rate_per_s=A,
                        gf_value=gf_val,
                        f_value=f_val,
                        line_strength=S,
                        gauge=gauge,
                    ))

        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _csf_transition_me(self, csf_r, csf_s, orbitals_dict, bsp_data,
                            L, mp_type, gauge):
        """Transition matrix element between two CSFs.

        For CSFs differing by one orbital (a -> b), the ME is the
        single-particle transition ME times a phase/coupling factor.
        For CSFs differing by more than one orbital, the ME is zero
        for one-body operators.
        """
        coupling = one_body_csf_coupling(csf_r, csf_s, L, mp_type)
        if coupling is None:
            return 0.0

        n_i, k_i = coupling.source
        n_f, k_f = coupling.destination

        # Get orbital wavefunctions
        if (n_i, k_i) not in orbitals_dict or (n_f, k_f) not in orbitals_dict:
            return 0.0

        P_i, Q_i = orbitals_dict[(n_i, k_i)]
        P_f, Q_f = orbitals_dict[(n_f, k_f)]

        if not selection_rules(k_i, k_f, L, mp_type):
            return 0.0

        # Radial matrix element
        if mp_type == 'E':
            if gauge == 'velocity':
                rme = radial_eL_velocity(P_i, Q_i, P_f, Q_f, bsp_data, L,
                                          kappa_i=k_i, kappa_f=k_f)
            else:
                rme = radial_eL(P_i, Q_i, P_f, Q_f, bsp_data, L)
            ang = angular_eL(k_i, k_f, L)
        else:
            rme = radial_mL(P_i, Q_i, P_f, Q_f, bsp_data, L)
            ang = angular_mL(k_i, k_f, L)

        return coupling.population_factor * ang * rme


def compute_transitions(atom, multipoles=None, gauge='length'):
    """Convenience function to compute transitions for a solved Atom.

    Parameters
    ----------
    atom : Atom
        Solved Atom object.
    multipoles : list of str
        Default: ['E1'].
    gauge : str
        'length' or 'velocity'.

    Returns
    -------
    list of TransitionResult
    """
    calc = TransitionCalculator(atom)
    return calc.compute_all(multipoles, gauge)
