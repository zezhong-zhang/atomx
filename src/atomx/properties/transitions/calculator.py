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
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity, radial_mL
from atomx.properties.transitions.angular import (
    angular_eL, angular_mL, selection_rules, allowed_final_kappas,
    line_strength_factor, _kappa_to_lj,
)
from atomx.properties.transitions.rates import (
    einstein_A_eL, einstein_A_mL, oscillator_strength, lifetime,
)
from atomx.angular.csf import (
    csf_has_explicit_j,
    csf_parity,
    csf_twice_j,
    multipole_allows_state_transition,
    one_body_csf_coupling,
)
from atomx.corrections.adapters import (
    _has_bspline_dirac_data,
    _has_radial_grid_data,
    _flatten_grid_orbitals,
    _grid_workspace_of,
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
    # Which line-resolution engine produced this row:
    #   'uta' — statistical/configuration-average machinery (single-config
    #           orbital pairs or the MCDHF UTA population-factor path);
    #   'ci'  — exact J-resolved determinant-CI line strengths
    #           (:mod:`atomx.properties.transitions.ci_lines`).
    resolution: str = "uta"
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

    def compute_all(self, multipoles=None, gauge='length',
                    line_resolution='uta', ci_det_cap=2000):
        """Compute all allowed transitions.

        Parameters
        ----------
        multipoles : list of str, optional
            Multipole types to compute. Default: ``['E1']``.
        line_resolution : {'uta', 'ci'}
            ``'uta'`` (default) — the statistical machinery: exact
            multiplet totals distributed with FAC's UTA population
            factor.  ``'ci'`` — exact J-resolved determinant-CI line
            strengths (:mod:`atomx.properties.transitions.ci_lines`)
            for single-configuration-pair atoms within the ``ci_det_cap``
            size cap; anything the CI engine cannot handle (more than
            two configurations, multi-hole configs, blocks above the
            cap, velocity gauge, magnetic multipoles) falls back to the
            UTA path.  Each :class:`TransitionResult` records the engine
            that produced it in ``resolution``.
        ci_det_cap : int
            Maximum determinants per (J, M) block for the CI engine.
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
        if line_resolution not in {'uta', 'ci'}:
            raise ValueError(
                f"line_resolution must be 'uta' or 'ci' (got {line_resolution!r})")

        if line_resolution == 'ci':
            ci_results = self._compute_ci_transitions(
                multipoles, gauge, ci_det_cap)
            if ci_results is not None:
                return ci_results
            # CI path not applicable — statistical fallback below.

        if self.atom._mcdhf_result is not None:
            return self._compute_mcdhf_transitions(multipoles, gauge)
        else:
            return self._compute_single_config_transitions(multipoles, gauge)

    def _compute_ci_transitions(self, multipoles, gauge, det_cap):
        """Exact J-resolved lines via the determinant-CI engine.

        Applies when the atom was built from exactly two configuration
        strings forming a single-hole pair (e.g. ``["1s2 2s2 2p6",
        "1s2 2s2 2p5 3d1"]``).  The closed-shell ground configuration is
        (re-)solved in the FAC-equivalent local-DFS basis
        (``numerics='shooting', exchange='gu'``, pure Coulomb) that the
        CI engine was validated with, and
        :func:`atomx.properties.transitions.ci_lines.ci_line_strengths`
        supplies the exact per-line strengths.

        Returns ``None`` whenever the CI path does not apply — the
        caller then falls back to the statistical (UTA) machinery.
        Magnetic multipoles in ``multipoles`` are not implemented in the
        CI engine; they are computed with the UTA machinery and marked
        ``resolution='uta'`` alongside the CI electric-multipole rows.
        """
        if gauge != 'length':
            return None
        cfgs = list(getattr(self.atom, '_config_strings', None) or [])
        if len(cfgs) != 2:
            return None

        from atomx.properties.transitions.ci_lines import (
            CISizeError, ci_line_strengths, ground_config_of_pair)

        try:
            ground_cfg = ground_config_of_pair(cfgs[0], cfgs[1])
        except ValueError:
            return None
        ground = self._ci_ground_result(ground_cfg)
        if ground is None:
            return None

        e_poles = [mp for mp in (multipoles or ['E1']) if mp[0] == 'E']
        m_poles = [mp for mp in (multipoles or ['E1']) if mp[0] != 'E']
        results = []
        for mp in e_poles:
            try:
                lineset = ci_line_strengths(
                    ground, cfgs[1], cfgs[0], multipole=mp, det_cap=det_cap)
            except (CISizeError, ValueError, NotImplementedError):
                return None
            for ln in lineset.lines:
                results.append(TransitionResult(
                    initial_label=ln.lower.label,
                    final_label=ln.upper.label,
                    energy_diff_hartree=ln.omega_au,
                    wavelength_angstrom=ln.wavelength_angstrom,
                    multipole=ln.multipole,
                    A_rate_per_s=ln.A_per_s,
                    gf_value=ln.gf,
                    f_value=ln.f_value,
                    line_strength=ln.S,
                    gauge='length',
                    A_length=ln.A_per_s,
                    A_velocity=None,
                    resolution='ci',
                ))
        if m_poles:
            if self.atom._mcdhf_result is not None:
                results.extend(self._compute_mcdhf_transitions(m_poles, gauge))
            else:
                results.extend(
                    self._compute_single_config_transitions(m_poles, gauge))
        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _ci_ground_result(self, ground_cfg):
        """Solved pure-Coulomb closed-shell ground in the DFS-gu basis.

        Cached per configuration string.  Returns ``None`` when the
        solve fails (caller falls back to the UTA path)."""
        cache = getattr(self, '_ci_ground_cache', None)
        if cache is None:
            cache = self._ci_ground_cache = {}
        if ground_cfg in cache:
            return cache[ground_cfg]
        from atomx.atom import Atom
        from atomx.scattering.autoionization.mscheme_ci import (
            _parse_config_tokens)
        res = None
        try:
            n_el = sum(_parse_config_tokens(ground_cfg).values())
            a = Atom(Z=self.Z, charge=int(self.Z) - int(n_el),
                     config=ground_cfg, numerics='shooting', exchange='gu',
                     breit=False, qed=False)
            a.solve()
            if a.converged:
                res = a._result
        except Exception:
            res = None
        cache[ground_cfg] = res
        return res

    def _single_config_payload(self):
        """``(orbitals, workspace)`` for whichever radial flavour the
        solved result carries.

        ``orbitals`` is a list of ``(label, n, kappa, P, Q, E, occ)``
        tuples.  For the B-spline flavour ``P``/``Q`` are expansion
        coefficient vectors and ``workspace`` is the B-spline
        ``bsp_data`` dict; for the radial-grid (ODE shooting) flavour
        they are grid-tabulated waves and ``workspace`` is a
        :func:`atomx.corrections.workspace.make_grid_workspace` dict.
        Either payload feeds the same radial integrals in
        :mod:`~atomx.properties.transitions.multipole` through the
        :mod:`atomx.corrections.workspace` seam.
        """
        result = self.atom._result

        if _has_bspline_dirac_data(result):
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
            return orbitals, result._bsp_data

        if _has_radial_grid_data(result):
            workspace = _grid_workspace_of(result)
            orbitals = []
            for (P, Q, occ, kappa, label), sh in zip(
                    _flatten_grid_orbitals(result), result.subshells):
                E = result.orbital_energies.get(label, 0.0)
                orbitals.append((label, sh.n, kappa, P, Q, E, occ))
            return orbitals, workspace

        raise NotImplementedError(
            "transitions() needs Dirac orbitals with a radial workspace: "
            "either B-spline (orbitals_raw + bsp_data) or grid-tabulated "
            "shooting (subshells + _grid/_difr/_pa/_qa) results are "
            f"supported; got {type(result).__name__} "
            f"(equation={getattr(result, '_equation', 'dirac')!r})")

    def _compute_single_config_transitions(self, multipoles, gauge):
        """Transitions for single-configuration (DHF/DFS) calculation.

        For single-config, we compute transitions between orbitals
        as if one electron is promoted from orbital i to orbital f.
        This gives orbital-to-orbital transition matrix elements.
        Works on both radial flavours (B-spline and ODE shooting) via
        :meth:`_single_config_payload`.
        """
        results = []

        # List of orbitals (label, n, kappa, P, Q, energy, occ) plus the
        # radial workspace the multipole integrals consume.
        orbitals, workspace = self._single_config_payload()

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
                        workspace, mp, mp_type, L, gauge)
                    if res is not None:
                        results.append(res)

        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _build_single_orbital_result(
        self, lab_i, lab_f, dE, omega, wavelength,
        P_i, Q_i, P_f, Q_f, k_i, k_f, tj_i, tj_f,
        workspace, mp, mp_type, L, gauge,
    ):
        """Construct a TransitionResult for one orbital pair / multipole.

        Centralised so the single-config and MCDHF paths share the
        gauge-handling logic.  ``workspace`` is either radial flavour
        (B-spline ``bsp_data`` or radial-grid workspace) — the multipole
        integrals dispatch through the workspace seam.
        """
        ang = angular_eL(k_i, k_f, L) if mp_type == 'E' else angular_mL(k_i, k_f, L)

        rme_length = rme_velocity = None
        if mp_type == 'E':
            if gauge in ('length', 'both'):
                rme_length = radial_eL(P_i, Q_i, P_f, Q_f, workspace, L)
            if gauge in ('velocity', 'both'):
                rme_velocity = radial_eL_velocity(
                    P_i, Q_i, P_f, Q_f, workspace, L,
                    kappa_i=k_i, kappa_f=k_f, omega=omega)
        else:
            # Magnetic multipoles: ``radial_mL`` is the only formula we
            # surface today.  Treat it as identical for length/velocity
            # so callers can still use ``gauge='both'`` uniformly.
            rme_length = rme_velocity = radial_mL(
                P_i, Q_i, P_f, Q_f, workspace, L)

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
        per_block_orbitals = getattr(mcdhf, "per_block_orbitals", {}) or {}
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

                    block_I = (int(tj_I), int(pi_I)) if explicit_I and explicit_F else None
                    block_F = (int(tj_F), int(pi_F)) if explicit_I and explicit_F else None
                    source_orbitals = per_block_orbitals.get(block_I, orbitals_dict)
                    destination_orbitals = per_block_orbitals.get(block_F, orbitals_dict)

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
                                    L, mp_type, g, omega=omega,
                                    source_orbitals=source_orbitals,
                                    destination_orbitals=destination_orbitals)
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
                            L, mp_type, gauge, *, omega=None,
                            source_orbitals=None, destination_orbitals=None):
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

        source_orbitals = source_orbitals or orbitals_dict
        destination_orbitals = destination_orbitals or orbitals_dict
        if (n_i, k_i) not in source_orbitals or (n_f, k_f) not in destination_orbitals:
            return 0.0

        P_i, Q_i = source_orbitals[(n_i, k_i)]
        P_f, Q_f = destination_orbitals[(n_f, k_f)]

        if not selection_rules(k_i, k_f, L, mp_type):
            return 0.0

        recoupling = 1.0
        if (csf_has_explicit_j(csf_r) and csf_has_explicit_j(csf_s)
                and csf_twice_j(csf_s) == 0):
            # FAC's Babushkin E1 radial operator includes a sqrt(2)
            # normalization relative to AtomX's long-wavelength radial_eL
            # convention.  For closed-shell J=0 -> explicit particle-hole
            # E1 CSFs, AngularZMix supplies the many-electron coefficient and
            # MultipoleRadialFRGrid supplies that one-electron normalization.
            # The previous sqrt((2j_particle+1)/(2J+1)) happened to match
            # sqrt(2) for 3d5/2 but underweighted the 3d3/2 branches by
            # sqrt(4/3)/sqrt(2), producing the residual Fe XVII 3C/3D split.
            if mp_type == 'E' and L == 1:
                recoupling = float(np.sqrt(2.0))
            else:
                recoupling = float(np.sqrt((2 * abs(k_f)) / (csf_twice_j(csf_r) + 1.0)))

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

        return recoupling * coupling.population_factor * ang * rme


def compute_transitions(atom, multipoles=None, gauge='length',
                        line_resolution='uta', ci_det_cap=2000):
    """Convenience function to compute transitions for a solved Atom.

    Parameters
    ----------
    atom : Atom
        Solved Atom object.
    multipoles : list of str
        Default: ['E1'].
    gauge : str
        'length' or 'velocity'.
    line_resolution : {'uta', 'ci'}
        ``'ci'`` uses the exact J-resolved determinant-CI engine for
        single-configuration-pair atoms (falls back to the statistical
        UTA machinery otherwise); each returned row records the engine
        in ``TransitionResult.resolution``.  Default ``'uta'``
        (unchanged behaviour).
    ci_det_cap : int
        Determinant-per-block cap for the CI engine.

    Returns
    -------
    list of TransitionResult
    """
    calc = TransitionCalculator(atom)
    return calc.compute_all(multipoles, gauge,
                            line_resolution=line_resolution,
                            ci_det_cap=ci_det_cap)
