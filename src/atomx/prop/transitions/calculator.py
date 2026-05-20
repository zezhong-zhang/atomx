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
    A_rate_per_s: float      # Einstein A coefficient (in ``gauge``)
    gf_value: float          # weighted oscillator strength
    f_value: float           # absorption oscillator strength
    line_strength: float     # in atomic units
    gauge: str = "length"    # 'length' or 'velocity'
    # Populated when gauge='both' (otherwise None). ``A_length`` always
    # holds the length-gauge value and ``A_velocity`` the velocity-gauge
    # value, regardless of which one ``A_rate_per_s`` reports.
    A_length: Optional[float] = None
    A_velocity: Optional[float] = None

    @property
    def wavelength_nm(self):
        return self.wavelength_angstrom / 10.0

    @property
    def energy_eV(self):
        return abs(self.energy_diff_hartree) * HARTREE_EV

    @property
    def energy_cm1(self):
        return abs(self.energy_diff_hartree) * HARTREE_CM1

    @property
    def gauge_ratio(self):
        """Return A_velocity / A_length when both gauges were computed.

        FAC and Grant treat this ratio as a structure-quality diagnostic
        (Gu 2008 §3.2): values close to 1 indicate the basis is close to
        an eigenfunction of the same Hamiltonian for length and velocity
        operators. Returns ``None`` when either gauge is unavailable.
        """
        if self.A_length is None or self.A_velocity is None:
            return None
        if self.A_length == 0.0:
            return float("inf") if self.A_velocity != 0.0 else 1.0
        return self.A_velocity / self.A_length


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
    ``tj=1`` placeholder. If a CSF carries an explicit ``total_J2``
    attribute that takes precedence — including the J=0 case, which is
    recognised either from a non-default attribute *or* from the
    bracketed ``[J=0/2]`` label tag emitted by
    :func:`atomx.methods.mcdhf.config.enumerate_jj_csfs`.
    """
    # Honour explicit total_J2 if somebody populated it.  The closed-shell
    # J=0 CSF carries ``total_J2=0`` (default int) but its label tags it
    # with ``[J=0/2]``; treat that as explicit so the caller doesn't fall
    # back to the open-shell heuristic.
    explicit = int(getattr(csf, "total_J2", 0))
    label = str(getattr(csf, "label", ""))
    if explicit > 0 or " J=" in label or "[J=" in label:
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
            Multipole types to compute. Default: ``['E1']``.
        gauge : {'length', 'velocity', 'both'}
            ``'length'`` (Babushkin) — default; preserves the historical
            ``A_rate_per_s`` value for all existing callers.
            ``'velocity'`` (Coulomb) — reports velocity-gauge rates only.
            ``'both'`` — reports the length-gauge value in
            ``A_rate_per_s`` and populates the ``A_length`` and
            ``A_velocity`` fields so the gauge-ratio diagnostic
            (Gu 2008 §3.2) can be evaluated per transition.

            Velocity-gauge support for higher multipoles (EL with L ≥ 2,
            ML with L ≥ 1) currently uses the leading-order formula;
            production-quality higher-multipole gauges are tracked
            separately. Length gauge remains the only value the default
            ``A_rate_per_s`` ever reports.

        Returns
        -------
        list of TransitionResult
        """
        if multipoles is None:
            multipoles = ['E1']

        if gauge not in {'length', 'velocity', 'both'}:
            raise ValueError(
                f"gauge must be 'length', 'velocity', or 'both' (got {gauge!r})")

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

                    res = self._build_single_orbital_result(
                        lab_i, lab_f, dE, omega, wavelength,
                        P_i, Q_i, P_f, Q_f, k_i, k_f, tj_i, tj_f,
                        bsp_data, mp, mp_type, L, gauge)
                    if res is not None:
                        results.append(res)

        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _build_single_orbital_result(
        self, lab_i, lab_f, dE, omega, wavelength,
        P_i, Q_i, P_f, Q_f, k_i, k_f, tj_i, tj_f,
        bsp_data, mp, mp_type, L, gauge,
    ):
        """Construct a TransitionResult for one orbital pair / multipole.

        Centralised so the single-config and MCDHF paths share the
        gauge-handling logic.
        """
        ang = angular_eL(k_i, k_f, L) if mp_type == 'E' else angular_mL(k_i, k_f, L)

        rme_length = rme_velocity = None
        if mp_type == 'E':
            if gauge in ('length', 'both'):
                rme_length = radial_eL(P_i, Q_i, P_f, Q_f, bsp_data, L)
            if gauge in ('velocity', 'both'):
                rme_velocity = radial_eL_velocity(
                    P_i, Q_i, P_f, Q_f, bsp_data, L,
                    kappa_i=k_i, kappa_f=k_f, omega=omega)
        else:
            # Magnetic multipoles: ``radial_mL`` is the only formula we
            # surface today.  Treat it as identical for length/velocity
            # so callers can still use ``gauge='both'`` uniformly.
            rme_length = rme_velocity = radial_mL(
                P_i, Q_i, P_f, Q_f, bsp_data, L)

        def _A_from_rme(rme):
            if rme is None:
                return None
            S = ang ** 2 * rme ** 2
            if S < 1e-30:
                return (0.0, S)
            if mp_type == 'E':
                A = einstein_A_eL(omega, S, L, tj_i)
            else:
                A = einstein_A_mL(omega, S, L, tj_i)
            return (A, S)

        len_pair = _A_from_rme(rme_length)
        vel_pair = _A_from_rme(rme_velocity)

        # Pick the gauge that owns ``A_rate_per_s``/``line_strength``:
        # length-gauge in 'length' and 'both' modes, velocity in 'velocity'.
        primary = vel_pair if gauge == 'velocity' else len_pair
        if primary is None or primary[1] < 1e-30:
            return None
        A_primary, S_primary = primary

        f_val, gf_val = oscillator_strength(omega, S_primary, L, tj_i, tj_f)

        A_length = len_pair[0] if len_pair is not None else None
        A_velocity = vel_pair[0] if vel_pair is not None else None
        reported_gauge = gauge if gauge != 'both' else 'length'

        return TransitionResult(
            initial_label=lab_i,
            final_label=lab_f,
            energy_diff_hartree=dE,
            wavelength_angstrom=wavelength,
            multipole=mp,
            A_rate_per_s=A_primary,
            gf_value=gf_val,
            f_value=f_val,
            line_strength=S_primary,
            gauge=reported_gauge,
            A_length=A_length,
            A_velocity=A_velocity,
        )

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

                    # Sum matrix element over CSF pairs.  When the caller
                    # asked for both gauges, accumulate them in parallel so
                    # the same CSF iteration produces a length/velocity
                    # pair without the cost of two outer loops.
                    sum_gauges = ('length', 'velocity') if gauge == 'both' else (gauge,)
                    total_me = {g: 0.0 for g in sum_gauges}
                    for r, csf_r in enumerate(csfs):
                        for s, csf_s in enumerate(csfs):
                            c_F = mixing[r, F]
                            c_I = mixing[s, I]
                            if abs(c_F * c_I) < 1e-12:
                                continue
                            for g in sum_gauges:
                                me = self._csf_transition_me(
                                    csf_r, csf_s, orbitals_dict, bsp_data,
                                    L, mp_type, g, omega=omega)
                                total_me[g] += c_F * c_I * me

                    # Build per-gauge A-coefficients.  ``dE > 0`` means I is
                    # the lower state and F the upper state; the Einstein
                    # spontaneous rate is for F -> I, so the statistical
                    # denominator is the upper-state 2J+1.
                    tj_upper = tj_F

                    def _A_for(g):
                        me = total_me.get(g, 0.0)
                        S_g = me ** 2
                        if S_g < 1e-30:
                            return None
                        if mp_type == 'E':
                            return (einstein_A_eL(omega, S_g, L, tj_upper), S_g)
                        return (einstein_A_mL(omega, S_g, L, tj_upper), S_g)

                    len_pair = _A_for('length') if 'length' in sum_gauges else None
                    vel_pair = _A_for('velocity') if 'velocity' in sum_gauges else None

                    primary = vel_pair if gauge == 'velocity' else len_pair
                    if primary is None:
                        continue
                    A_primary, S_primary = primary

                    f_val, gf_val = oscillator_strength(omega, S_primary, L, tj_I, tj_F)

                    A_length = len_pair[0] if len_pair is not None else None
                    A_velocity = vel_pair[0] if vel_pair is not None else None
                    reported_gauge = gauge if gauge != 'both' else 'length'

                    results.append(TransitionResult(
                        initial_label=f"State_{I}",
                        final_label=f"State_{F}",
                        energy_diff_hartree=dE,
                        wavelength_angstrom=wavelength,
                        multipole=mp,
                        A_rate_per_s=A_primary,
                        gf_value=gf_val,
                        f_value=f_val,
                        line_strength=S_primary,
                        gauge=reported_gauge,
                        A_length=A_length,
                        A_velocity=A_velocity,
                    ))

        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _csf_transition_me(self, csf_r, csf_s, orbitals_dict, bsp_data,
                            L, mp_type, gauge, *, omega=None):
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
                                          kappa_i=k_i, kappa_f=k_f,
                                          omega=omega)
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
