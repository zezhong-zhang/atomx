"""High-level transition calculator bridging structure to spectroscopy.

Computes all allowed transitions between atomic states (single-config
or MCDHF) with proper angular coupling and radial matrix elements.

Usage:
    from atomx import Atom
    he = Atom(2, config=["1s2", "1s1 2p1"]).solve()
    results = he.transitions(multipoles=['E1', 'M1'])
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from atomx.angular.csf import (
    csf_has_explicit_j,
    csf_parity,
    csf_twice_j,
    multipole_allows_state_transition,
)
from atomx.angular.csf_hamiltonian import (
    csf_signature,
    nonorthogonal_tensor_terms,
    one_body_tensor_terms,
)
from atomx.angular.recoupling import lj_from_kappa
from atomx.config.subshell import l_letter
from atomx.constants import (
    HARTREE_CM1,
    HARTREE_EV,
    HARTREE_INV_ANGSTROM,
)
from atomx.corrections.adapters import (
    _flatten_grid_orbitals,
    _grid_workspace_of,
    _has_bspline_dirac_data,
    _has_radial_grid_data,
)
from atomx.properties.transitions.angular import (
    angular_eL,
    angular_mL,
    selection_rules,
)
from atomx.properties.transitions.multipole import (
    radial_eL,
    radial_eL_velocity,
    radial_mL,
)
from atomx.properties.transitions.rates import (
    einstein_A_eL,
    einstein_A_mL,
    oscillator_strength,
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
    photon_approximation: str = "long_wavelength"
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
    provenance: dict = field(default_factory=dict)

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
    l, tj = lj_from_kappa(kappa)
    return f"{n}{l_letter(l)}{tj}/2"


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
                    line_resolution='uta', ci_det_cap=2000, *,
                    fallback='error', ci_reference=None):
        """Compute all allowed transitions.

        Parameters
        ----------
        multipoles : list of str, optional
            Multipole types to compute. Default: ``['E1']``.
        line_resolution : {'uta', 'ci'}
            MCDHF states always use exact CSF one-body tensors, with the
            Hamiltonian's coupling and phase conventions, and report
            ``resolution='ci'``. For other result types, ``'uta'`` selects
            statistical multiplet strengths and ``'ci'`` requests the
            determinant-CI engine where supported; each row records the
            engine actually used in ``resolution``.
        fallback : {'error', 'uta'}
            Unsupported determinant-CI requests raise by default. Explicit
            'uta' permits statistical fallback and records the reason.
        ci_reference : solved result, optional
            Explicit closed-shell orbital reference for determinant CI. This
            path constructs a Dirac–Coulomb CI Hamiltonian; it does not inherit
            post-SCF corrections from the reference or run a replacement SCF.
        ci_det_cap : int
            Maximum determinants per (J, M) block for the CI engine.
        gauge : {'length', 'velocity', 'both'}
            ``'length'`` (Babushkin) — default; uses the finite-frequency
            electric operator at the transition energy.
            ``'velocity'`` (Coulomb) — reports velocity-gauge rates only.
            ``'both'`` — reports the length-gauge value in
            ``A_rate_per_s`` and populates the ``A_length`` and
            ``A_velocity`` fields so the gauge-ratio diagnostic
            (Gu 2008 §3.2) can be evaluated per transition.

            Both electric gauges retain all spherical-Bessel photon orders.
            Magnetic multipoles use the same finite-frequency operator in
            both entries; their gauge ratio is not an independent diagnostic.
            The determinant-CI engine currently supports length gauge only.

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

        if fallback not in {'error', 'uta'}:
            raise ValueError("fallback must be 'error' or 'uta'")
        if ci_reference is not None and line_resolution != 'ci':
            raise ValueError("ci_reference requires line_resolution='ci'")
        # Already computed correlated states define their own Hamiltonian and
        # tensor operators. Never replace them with a separately solved atom.
        if self.atom._csf_result is not None and ci_reference is None:
            return self._compute_mcdhf_transitions(multipoles, gauge)
        if line_resolution == 'ci':
            from atomx.properties.transitions.ci_lines import CISizeError
            try:
                return self._compute_ci_transitions(multipoles, gauge, ci_det_cap,
                                                    ci_reference)
            except (NotImplementedError, CISizeError) as exc:
                if fallback == 'error':
                    raise
                rows = self._compute_single_config_transitions(multipoles, gauge)
                for row in rows:
                    row.provenance.update(requested_resolution='ci', actual_resolution=row.resolution,
                        fallback_reason=str(exc), fallback_exception=type(exc).__name__)
                return rows
        return self._compute_single_config_transitions(multipoles, gauge)

    def _compute_ci_transitions(self, multipoles, gauge, det_cap, reference):
        """Evaluate determinant CI using an explicitly supplied orbital reference.

        This engine builds a Dirac–Coulomb CI Hamiltonian. The reference's
        orbital model is retained in provenance; no additional SCF is performed.
        """
        if gauge != 'length':
            raise NotImplementedError('determinant-CI transitions support length gauge only')
        cfgs = list(getattr(self.atom, '_config_strings', None) or [])
        if len(cfgs) != 2:
            raise NotImplementedError('determinant-CI transitions require two configurations')
        if any(mp[0] != 'E' for mp in multipoles):
            raise NotImplementedError('determinant-CI magnetic multipoles are unsupported')
        if reference is None:
            raise NotImplementedError('determinant-CI transitions require an explicit solved '
                                      'ci_reference; no replacement atom is calculated')
        from atomx.properties.transitions.ci_lines import (
            ci_line_strengths,
            ground_config_of_pair,
        )
        from atomx.scattering.autoionization.mscheme_ci import _parse_config_tokens
        ground_cfg = ground_config_of_pair(cfgs[0], cfgs[1])
        result = getattr(reference, '_result', reference)
        if result is None or not result.converged:
            raise ValueError('ci_reference must be solved and converged')
        if result.Z != self.Z:
            raise ValueError('ci_reference nuclear charge differs from the requested atom')
        occupations = {}
        for sh in result.subshells:
            occupations[sh.n, sh.l] = occupations.get((sh.n, sh.l), 0.) + sh.occupation
        if occupations != _parse_config_tokens(ground_cfg):
            raise ValueError('ci_reference is not the closed-shell reference for this pair')
        spec = getattr(result, 'calculation_spec', None)
        requested = getattr(self.atom._result, 'calculation_spec', None)
        provenance = dict(ci_hamiltonian='Dirac-Coulomb', reference_selection='explicit',
            orbital_reference=spec.to_dict() if spec else None,
            requested_calculation=requested.to_dict() if requested else None)
        results = []
        for mp in multipoles:
            lineset = ci_line_strengths(result, cfgs[1], cfgs[0],
                                       multipole=mp, det_cap=det_cap)
            for ln in lineset.lines:
                results.append(TransitionResult(
                    initial_label=ln.lower.label, final_label=ln.upper.label,
                    energy_diff_hartree=ln.omega_au,
                    wavelength_angstrom=ln.wavelength_angstrom,
                    multipole=ln.multipole, A_rate_per_s=ln.A_per_s,
                    gf_value=ln.gf, f_value=ln.f_value, line_strength=ln.S,
                    gauge='length', A_length=ln.A_per_s, resolution='ci',
                    photon_approximation=lineset.photon_approximation,
                    provenance=dict(provenance)))
        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _single_config_payload(self):
        """``(orbitals, workspace)`` for whichever radial flavour the
        solved result carries.

        ``orbitals`` is a list of ``(label, n, kappa, P, Q, E, occ)``
        tuples.  For the B-spline flavour ``P``/``Q`` are expansion
        coefficient vectors and ``workspace`` is the B-spline
        ``bsp_data`` dict; for the radial-grid (ODE shooting) flavour
        they are grid-tabulated waves and ``workspace`` is a
        :func:`atomx.numerics.radial_workspace.make_grid_workspace` dict.
        Either payload feeds the same radial integrals in
        :mod:`~atomx.properties.transitions.multipole` through the
        :mod:`atomx.numerics.radial_workspace` seam.
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
                    _flatten_grid_orbitals(result), result.subshells, strict=False):
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
        for i, (lab_i, _n_i, k_i, P_i, Q_i, E_i, _occ_i) in enumerate(orbitals):
            for j, (lab_f, _n_f, k_f, P_f, Q_f, E_f, _occ_f) in enumerate(orbitals):
                if i == j:
                    continue  # no self-transition

                dE = E_f - E_i
                if dE <= 0:
                    continue  # only emission (upper -> lower)

                omega = abs(dE)
                wavelength = _energy_to_wavelength_angstrom(dE)
                _, tj_i = lj_from_kappa(k_i)
                _, tj_f = lj_from_kappa(k_f)

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

        The caller only keeps pairs with ``dE = E_f − E_i > 0``, so
        ``i`` is the **lower** and ``f`` the **upper** state: the
        spontaneous-emission rate carries the upper-state statistical
        weight ``2 j_f + 1``, while the absorption oscillator strength
        keeps the lower-state weight ``2 j_i + 1``.
        """
        ang = angular_eL(k_i, k_f, L) if mp_type == 'E' else angular_mL(k_i, k_f, L)

        rme_length = rme_velocity = None
        if mp_type == 'E':
            if gauge in ('length', 'both'):
                rme_length = radial_eL(P_i, Q_i, P_f, Q_f, workspace, L,
                    kappa_i=k_i, kappa_f=k_f, omega=omega)
            if gauge in ('velocity', 'both'):
                rme_velocity = radial_eL_velocity(
                    P_i, Q_i, P_f, Q_f, workspace, L,
                    kappa_i=k_i, kappa_f=k_f, omega=omega)
        else:
            # Magnetic multipoles: ``radial_mL`` is the only formula we
            # surface today.  Treat it as identical for length/velocity
            # so callers can still use ``gauge='both'`` uniformly.
            rme_length = rme_velocity = radial_mL(
                P_i, Q_i, P_f, Q_f, workspace, L, omega=omega)

        def _A_from_rme(rme):
            if rme is None:
                return None
            S = ang ** 2 * rme ** 2
            if S < 1e-30:
                return (0.0, S)
            # ``i`` is the lower state here (dE > 0 upstream), so the
            # spontaneous rate needs the *upper*-state weight 2 j_f + 1 —
            # matching ``einstein_A_eL``'s "tj_i = 2 j_upper" contract and
            # the MCDHF (``tj_upper = tj_F``) and CI (``up.two_J``) engines.
            if mp_type == 'E':
                A = einstein_A_eL(omega, S, L, tj_f)
            else:
                A = einstein_A_mL(omega, S, L, tj_f)
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
            photon_approximation='finite_wavelength',
            A_length=A_length,
            A_velocity=A_velocity,
        )

    def _compute_mcdhf_transitions(self, multipoles, gauge):
        """Transitions between MCDHF CI eigenstates.

        For MCDHF, the transition matrix element between states |I> and |F> is:
            <F|T^L|I> = sum_{r,s} c_r^F * c_s^I * <CSF_r|T^L|CSF_s>

        where c_r^F are the CI mixing coefficients.
        """
        mcdhf = self.atom._csf_result
        bsp_data = mcdhf.bsp_data
        csfs = mcdhf.csfs
        mixing = mcdhf.mixing_coefficients
        state_energies = mcdhf.state_energies
        orbitals_dict = mcdhf.orbitals_dict
        per_block_orbitals = getattr(mcdhf, "per_block_orbitals", {}) or {}
        n_states = len(state_energies)
        results = []
        self._csf_radial_cache = {}

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
                        tj_I = csf_twice_j(csfs[dom_I])
                        tj_F = csf_twice_j(csfs[dom_F])

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

                    def _A_for(g, total_me=total_me, mp_type=mp_type,
                               omega=omega, L=L, tj_upper=tj_upper):
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
                        photon_approximation='finite_wavelength',
                        resolution="ci",
                        A_length=A_length,
                        A_velocity=A_velocity,
                    ))

        return sorted(results, key=lambda r: r.wavelength_angstrom)

    def _csf_transition_me(self, csf_r, csf_s, orbitals_dict, bsp_data,
                            L, mp_type, gauge, *, omega=None,
                            source_orbitals=None, destination_orbitals=None):
        """Exact reduced one-body tensor in the CI coupling convention."""
        source_orbitals = source_orbitals or orbitals_dict
        destination_orbitals = destination_orbitals or orbitals_dict
        total = 0.0
        bra, ket = csf_signature(csf_r), csf_signature(csf_s)
        if source_orbitals is destination_orbitals:
            terms = one_body_tensor_terms(bra, ket, L)
        else:
            from atomx.methods.dirac_fock import _orbital_overlap

            def overlap(a, b):
                return _orbital_overlap(*destination_orbitals[a], *source_orbitals[b], bsp_data)

            terms = nonorthogonal_tensor_terms(bra, ket, L, overlap)
        for destination, source, coefficient in terms:
            if source not in source_orbitals or destination not in destination_orbitals:
                continue
            k_i, k_f = source[1], destination[1]
            if not selection_rules(k_i, k_f, L, mp_type):
                continue
            # Radial integrals repeat across many CSF pairs in a state pair.
            cache_key = (source, destination, L, mp_type, gauge, omega,
                         id(source_orbitals), id(destination_orbitals))
            cache = getattr(self, "_csf_radial_cache", None)
            if cache is None:
                cache = self._csf_radial_cache = {}
            if cache_key not in cache:
                P_i, Q_i = source_orbitals[source]
                P_f, Q_f = destination_orbitals[destination]
                if mp_type == 'E':
                    if gauge == 'velocity':
                        radial = radial_eL_velocity(
                            P_i, Q_i, P_f, Q_f, bsp_data, L,
                            kappa_i=k_i, kappa_f=k_f, omega=omega)
                    else:
                        radial = radial_eL(P_i, Q_i, P_f, Q_f, bsp_data, L,
                            kappa_i=k_i, kappa_f=k_f, omega=omega)
                    # Same angular factor as the single-orbital path
                    # (:func:`angular_eL` ≡ reduced_C_jj(k_f, k_i, L) by the
                    # 3j column-swap/m-flip symmetry; pinned in
                    # tests/test_angular.py).
                    angular = angular_eL(k_i, k_f, L)
                else:
                    radial = radial_mL(P_i, Q_i, P_f, Q_f, bsp_data, L, omega=omega)
                    angular = angular_mL(k_i, k_f, L)
                cache[cache_key] = angular * radial
            total += coefficient * cache[cache_key]
        return total


def compute_transitions(atom, multipoles=None, gauge='length',
                        line_resolution='uta', ci_det_cap=2000, *,
                    fallback='error', ci_reference=None):
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
        MCDHF states use exact CSF tensors for either setting. For other
        result types, ``'uta'`` selects statistical strengths and ``'ci'``
        requests the determinant-CI engine where supported. Each row's
        ``resolution`` records the actual engine.
    ci_det_cap : int
        Determinant-per-block cap for the CI engine.

    Returns
    -------
    list of TransitionResult
    """
    calc = TransitionCalculator(atom)
    return calc.compute_all(multipoles, gauge,
                            line_resolution=line_resolution,
                            ci_det_cap=ci_det_cap, fallback=fallback,
                            ci_reference=ci_reference)
