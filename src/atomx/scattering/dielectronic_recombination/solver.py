"""Dielectronic recombination (DR) rate coefficients.

Dielectronic recombination is a two-step resonant process:

  1. A free electron is *captured* into a doubly-excited resonance
     (inverse Auger, rate ∝ detailed-balance ∝ Γ_Auger).
  2. The resonance decays *radiatively* to a lower singly-bound level,
     stabilising the recombined ion.

At plasma temperature T_e, the rate coefficient per resonance d is

    α_d^DR(T)
        =  (h² / (2π m_e k_B T_e))^{3/2}   ←   de-Broglie prefactor
           · (g_d / (2 g_i))                ←   statistical weight ratio
           · (Γ_Auger · Γ_radiative / Γ_total)   ←   branching ratio
           · exp(−E_d / k_B T_e)                   ←   Boltzmann tail   (1)

The total DR rate coefficient sums over all resonances.  In
convenient atomic units:

    α^DR(T_e) [cm³/s]
        = 6.603 × 10⁻²⁴  ·  T_e[eV]^{−3/2}
                         ·  Σ_d  ω_d · ⟨Γ⟩_d  ·  exp(−E_d/T_e)        (2)

with  ω_d = g_d / (2 g_i)  and  ⟨Γ⟩_d = Γ_A · Γ_r / (Γ_A + Γ_r).

References
----------
A. Burgess, *Astrophys. J.* **139**, 776 (1965).
D. W. Savin, *Astrophys. J.* **562**, 1259 (2001) — Fe XVI benchmark.
M. F. Gu, *Astrophys. J.* **590**, 1131 (2003) — FAC DR calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

from atomx.scattering.autoionization.auger import AutoionizationState
from atomx.scattering.dielectronic_recombination.enumerate import (
    DRResonanceSpec,
    build_autoionization_states,
    enumerate_dr_resonances,
    resonance_strength_rows,
    _config_string,
    _initial_statistical_weight,
)
from atomx.scattering.dielectronic_recombination.widths import (
    has_solved_result,
    make_abinitio_auger_rate_fn,
    make_abinitio_radiative_rate_fn,
    solved_result,
)

# --- Universal prefactor for eq. (2) -------------------------------------
#
# Expression: (h² / (2π m_e k_B))^{3/2}  in units of cm³ · eV^{3/2} · s.
# CODATA 2018:
#   h  = 6.62607015e-34  J·s
#   m_e = 9.1093837015e-31 kg
#   k_B = 1.380649e-23  J/K
#   1 eV = 1.602176634e-19 J
# Gives (numerically, with T in eV rather than K):
#   Prefactor = 6.60305e-24  cm³/s · eV^{3/2}
#
# See Burgess 1965 eq. (6), or Osterbrock "Astrophysics of Gaseous
# Nebulae" §5.7.
_DR_PREFACTOR_CM3_S_EV_32 = 6.60305e-24


@dataclass
class DRRateCoefficient:
    """Dielectronic-recombination rate coefficient, tabulated on T_e.

    Attributes
    ----------
    T_eV : np.ndarray
        Temperature grid in eV.
    alpha_cm3_s : np.ndarray
        α^DR(T_e) in cm³/s at each temperature.
    per_resonance : dict, optional
        ``{resonance_label: α_cm3_s}`` — contribution of each resonance
        summed over T.  Useful for identifying dominant resonances.
    initial_state_label : str, optional
        Label of the initial ion ground state.
    initial_state_g : float
        Statistical weight of the initial ion ground state (``2J_i + 1``).
    """

    T_eV: np.ndarray
    alpha_cm3_s: np.ndarray
    per_resonance: Dict[str, np.ndarray] = field(default_factory=dict)
    initial_state_label: str = ""
    initial_state_g: float = 1.0
    resonances: List[AutoionizationState] = field(default_factory=list)

    def at_T(self, T_eV: float) -> float:
        """Linear-interpolate α^DR at an arbitrary temperature."""
        return float(np.interp(T_eV, self.T_eV, self.alpha_cm3_s))


def dr_rate_coefficient(
    resonances: Sequence[AutoionizationState],
    T_eV,
    *,
    initial_state_g: float = 1.0,
    initial_state_label: str = "",
    ne_cm3: float | None = None,
    Z_ion: int | None = None,
    series_limit_eV: float | None = None,
    ipd_model: str = "stewart_pyatt",
) -> DRRateCoefficient:
    """Burgess-formula DR rate coefficient from a list of resonances.

    Implements eq. (2) above.  Each resonance contributes a
    ``T_e^{−3/2} · exp(−E_d / T_e)`` term weighted by its statistical
    branching ratio.

    Density suppression (continuum lowering): when ``ne_cm3``, ``Z_ion``
    and ``series_limit_eV`` (the excitation threshold the resonance
    series converges to) are given, a resonance whose captured-electron
    binding energy ``series_limit_eV - E_resonance_eV`` is smaller than
    the ionization-potential depression
    (:func:`atomx.plasma.screening.ionization_potential_depression`,
    Stewart-Pyatt by default) is dissolved into the lowered continuum
    and dropped, per temperature.  This removes the high-n Rydberg
    resonances that dominate the low-density rate but cannot survive at
    ICF/stellar-interior densities (20-50% of the total there); the
    finer collisional l-mixing/ionization suppression of surviving
    levels (Burgess & Summers 1969) is a documented upgrade.  With
    ``ne_cm3 = None`` (default) the zero-density rate is unchanged.

    Parameters
    ----------
    resonances : sequence of :class:`AutoionizationState`
        Doubly-excited resonances with populated ``E_resonance_eV``,
        ``J``, ``gamma_auger_per_s``, ``a_radiative_per_s``.
    T_eV : float or array_like
        Electron temperatures in eV.  Output is vectorised over this.
    initial_state_g : float
        ``2 J_i + 1`` statistical weight of the initial ion ground
        state (the one the free electron captures onto).
    initial_state_label : str, optional
        Informational only.

    Returns
    -------
    :class:`DRRateCoefficient`
    """
    T_eV = np.atleast_1d(np.asarray(T_eV, dtype=float))
    if np.any(T_eV <= 0):
        raise ValueError("Temperatures must be positive (eV).")

    survival = np.ones_like(T_eV, dtype=float)
    ipd_eV = None
    if ne_cm3 is not None and ne_cm3 > 0.0:
        if Z_ion is None or series_limit_eV is None:
            raise ValueError(
                "density suppression needs Z_ion and series_limit_eV "
                "alongside ne_cm3")
        from atomx.plasma.screening import ionization_potential_depression
        ipd_eV = np.array([
            float(ionization_potential_depression(Z_ion, ne_cm3, t,
                                                  model=ipd_model))
            for t in T_eV])

    alpha_total = np.zeros_like(T_eV, dtype=float)
    per = {}

    for d in resonances:
        w = d.total_width_per_s
        if w <= 0.0:
            continue
        # Branching ratio: Γ_A Γ_r / (Γ_A + Γ_r)  in s⁻¹
        branch = d.gamma_auger_per_s * d.a_radiative_per_s / w
        if branch <= 0.0:
            continue
        omega = d.g / (2.0 * initial_state_g)

        if ipd_eV is not None:
            e_bind = float(series_limit_eV) - d.E_resonance_eV
            survival = (e_bind > ipd_eV).astype(float)

        # Boltzmann factor  exp(-E_d / T_e)
        boltz = np.exp(-d.E_resonance_eV / T_eV)
        contribution = (
            _DR_PREFACTOR_CM3_S_EV_32
            * T_eV ** (-1.5)
            * omega
            * branch
            * boltz
            * survival
        )
        alpha_total += contribution
        per[d.label] = contribution

    return DRRateCoefficient(
        T_eV=T_eV,
        alpha_cm3_s=alpha_total,
        per_resonance=per,
        initial_state_label=initial_state_label,
        initial_state_g=initial_state_g,
        resonances=list(resonances),
    )


def resonant_excitation_rate(
    resonances,
    T_eV,
    *,
    initial_state_g: float = 1.0,
    partial_auger_to_final,
):
    """Resonant-excitation rate coefficient g -> f via capture + re-Auger.

    The DR twin channel: capture into a doubly-excited resonance
    (detailed balance from the ground-channel Auger width) followed by
    autoionization INTO AN EXCITED final state f,

        alpha_RE(T) = P * T^{-3/2} sum_d omega_d
                      * Gamma_A^g(d) * Gamma_A^f(d) / Gamma_tot(d)
                      * exp(-E_d / T),

    the analog of the Burgess formula with the radiative width replaced
    by the partial Auger width to f.  This is the resonance
    contribution to the effective excitation rate that the direct DW
    collision strength misses (review M6).

    ``partial_auger_to_final`` maps each resonance label to
    Gamma_A(d -> f) in s^-1; resonances absent from the map contribute
    nothing.  Ground-channel widths and total widths come from the
    :class:`AutoionizationState` entries (a_radiative_per_s counts in
    Gamma_tot as usual).
    """
    T_eV = np.atleast_1d(np.asarray(T_eV, dtype=float))
    if np.any(T_eV <= 0):
        raise ValueError("Temperatures must be positive (eV).")
    alpha = np.zeros_like(T_eV)
    for d in resonances:
        g_f = float(partial_auger_to_final.get(d.label, 0.0))
        if g_f <= 0.0:
            continue
        w_tot = d.gamma_auger_per_s + g_f + d.a_radiative_per_s
        if w_tot <= 0.0:
            continue
        omega = d.g / (2.0 * initial_state_g)
        branch = d.gamma_auger_per_s * g_f / w_tot
        alpha += (_DR_PREFACTOR_CM3_S_EV_32 * T_eV ** (-1.5) * omega
                  * branch * np.exp(-d.E_resonance_eV / T_eV))
    return alpha


def dr_strengths(
    resonances: Sequence[AutoionizationState] | DRRateCoefficient,
    T_eV=None,
    *,
    initial_state_g: float | None = None,
) -> List[dict]:
    """Return per-resonance DR contribution rows for diagnostics.

    Parameters
    ----------
    resonances
        Either a resonance sequence or the :class:`DRRateCoefficient`
        returned by :func:`dr_calculate`.
    T_eV
        Temperature grid in eV.  Optional when ``resonances`` is a
        :class:`DRRateCoefficient`; its stored grid is used by default.
    initial_state_g
        Statistical weight of the recombining ion ground state.  Defaults
        to the value stored on ``DRRateCoefficient`` or ``1``.
    """

    if isinstance(resonances, DRRateCoefficient):
        rc = resonances
        seq = rc.resonances
        T = rc.T_eV if T_eV is None else T_eV
        g_i = rc.initial_state_g if initial_state_g is None else initial_state_g
    else:
        if T_eV is None:
            raise ValueError("T_eV is required when passing raw resonances")
        seq = resonances
        T = T_eV
        g_i = 1.0 if initial_state_g is None else initial_state_g
    return resonance_strength_rows(seq, T, initial_state_g=g_i)


def _build_mcdhf_result_for_dr(
    atom,
    n_max: int,
    l_max: int,
) -> object:
    """Build a compact MCDHF result for DR width calculations.

    Lightweight DR drivers often provide only ``orbital_energies`` and a
    core configuration.  This helper runs a single-reference MCDHF
    calculation with single excitations into the virtual shells needed
    for DR (``n <= n_max``, ``l <= l_max``).  The resulting orbitals and
    B-spline workspace are used by the ab-initio Auger and E1 radiative
    width factories.
    """
    from atomx.atom import Atom
    from atomx.methods.mcdhf.excitations import generate_sd_excitations

    Z = int(getattr(atom, "Z", 0))
    charge = int(getattr(atom, "charge", 0))
    core_config = _config_string(atom)
    if Z <= 0 or not core_config:
        raise ValueError(
            "Automatic DR structure result requires atom.Z and a core "
            "configuration (atom._config_strings or atom.config)."
        )

    l_symbols = "spdfghiklmno"
    virtuals: list[str] = []
    for n in range(1, n_max + 1):
        for l in range(0, min(l_max, n - 1) + 1):
            label = f"{n}{l_symbols[l]}"
            if label not in core_config.replace(" ", ""):
                virtuals.append(label)

    configs = generate_sd_excitations(
        core_config,
        virtuals,
        core_orbitals=["1s"],
        max_excitations=1,
    )

    struct = Atom(
        Z=Z,
        charge=charge,
        configs=configs,
        exchange="hf",
        ns=40,
        ks=7,
    )
    struct.solve()
    return struct.result


def _dr33_manifold_states(initial_atom) -> List[AutoionizationState]:
    """Ab-initio dr33 resonance list from determinant-CI manifolds.

    Builds every 2p- and 2s-hole ``3l3l'`` (dr33) resonance manifold of a
    Ne-like recombining ion with
    :class:`~atomx.scattering.autoionization.mscheme_ci.ResonanceManifold`
    and returns one :class:`AutoionizationState` per CI level, carrying
    the ab-initio resonance energy (``E_level − E_ion`` in the manifold's
    frozen-orbital accounting), the exact determinant Auger width, and
    the total E1 stabilizing width (``ResonanceManifold.dr_levels``).
    The n=3 orbitals are frozen-core V^{N-1} eigenstates (FAC's
    fixed-potential convention) from
    :func:`atomx.methods.mbpt.build_virtual_basis`.

    Verified against the complete clean-room FAC dr33 reference
    (``tests/data/fe17_dr33_fac_cleanroom.json``; SetSE/VP/MS/Breit off,
    AICut 0): α_DR ratios 0.49 / 0.99 / 1.08 / 1.17 at 1 / 5 / 10 / 50 MK
    for Fe XVII — see ``tests/test_dr_manifold_e2e.py`` and
    ``benchmarks/bench_fac_dr.py``.

    Raises
    ------
    NotImplementedError
        If the recombining ion is not a solved Ne-like ``1s2 2s2 2p6``
        system — the only manifold topology implemented (and FAC-verified)
        so far.
    """
    from atomx.constants import HARTREE_EV
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold
    from atomx.numerics.quadrature import make_fine_grid, p_rel

    unsupported = (
        "dr_calculate without width hooks drives the ab-initio "
        "determinant-CI manifold pipeline (ResonanceManifold.dr_levels), "
        "which currently implements only Ne-like recombining ions "
        "(core '1s2 2s2 2p6', dr33 3l3l' complex — e.g. Fe XVII) over a "
        "solved MCDHF result. "
    )
    try:
        core = _config_string(initial_atom)
    except TypeError:
        core = ""
    if sorted(core.split()) != ["1s2", "2p6", "2s2"]:
        raise NotImplementedError(
            unsupported
            + f"Got core config {core!r}.  Pass explicit auger_rate_fn / "
            "radiative_rate_fn hooks, or feed externally computed "
            "AutoionizationState rows to dr_rate_coefficient."
        )

    result = solved_result(initial_atom)
    bsp = None
    if result is not None:
        bsp = getattr(result, "_bsp_data", None) or getattr(result, "bsp_data", None)
    if result is None or bsp is None or not getattr(result, "orbitals_raw", None):
        raise NotImplementedError(
            unsupported
            + "The atom does not carry a solved MCDHF result with raw "
            "B-spline orbitals (orbitals_raw / bsp_data); solve the "
            "recombining ion first, e.g. "
            "Atom(Z=26, charge=16, configs=['1s2 2s2 2p6']).solve()."
        )

    Z = int(getattr(initial_atom, "Z", 0) or getattr(result, "Z", 0))

    # Fine radial grid sized for the fastest Auger electron (~500 eV for
    # Fe XVII dr33; harmless overkill for lighter Ne-like systems).
    r_fine = make_fine_grid(float(bsp["t"][-1]), p_rel(500.0 / HARTREE_EV))
    dr_fine = np.gradient(r_fine)
    gr = bsp["gr"].ravel()
    order = np.argsort(gr)
    grs = gr[order]

    def _tofine(P, Q):
        Pg = eval_P(P, bsp).ravel()[order]
        Qg = eval_Q(Q, bsp).ravel()[order]
        return np.interp(r_fine, grs, Pg), np.interp(r_fine, grs, Qg)

    # n=2 hole orbitals from the solved ground configuration; n=3 excited
    # orbitals from the frozen-core V^{N-1} eigenbasis.
    orbital_map = {}
    orbital_eps = {}
    for kappa, lab, idx in ((-1, "2s1/2", 1), (1, "2p1/2", 0), (-2, "2p3/2", 0)):
        P, Q, _ = result.orbitals_raw[kappa][idx]
        orbital_map[(2, kappa)] = _tofine(P, Q)
        orbital_eps[(2, kappa)] = float(result.orbital_energies[lab])
    for kappa in (-1, 1, -2, 2, -3):
        _occ, virt = build_virtual_basis(result, kappa)
        P, Q, e = virt[0]
        orbital_map[(3, kappa)] = _tofine(P, Q)
        orbital_eps[(3, kappa)] = float(e)

    # Active-space groups: hole shell + the three n=3 nl groups.
    G3 = [([(3, -1)], 0), ([(3, 1), (3, -2)], 0), ([(3, 2), (3, -3)], 0)]

    def _make(hole_subs, hole_ne, confs):
        return ResonanceManifold(
            result, [(hole_subs, hole_ne)] + G3,
            r_fine=r_fine, dr_fine=dr_fine, orbital_map=orbital_map,
            orbital_energies=orbital_eps, configs=confs)

    # Parity-blocked multi-configuration manifolds (matching the FAC dr33
    # configuration set '2s2 2p5 3*2' + '2s1 2p6 3*2'):
    #   2p^5 {3s3p + 3p3d}
    #   2p^5 {3s3d + 3p² + 3d² + 3s²}
    #   2s^1 {3s² + 3s3d + 3p² + 3d²}
    #   2s^1 {3s3p + 3p3d}
    manifolds = [
        _make([(2, 1), (2, -2)], 5, [(5, 1, 1, 0), (5, 0, 1, 1)]),
        _make([(2, 1), (2, -2)], 5,
              [(5, 1, 0, 1), (5, 0, 2, 0), (5, 0, 0, 2), (5, 2, 0, 0)]),
        _make([(2, -1)], 1,
              [(1, 2, 0, 0), (1, 1, 0, 1), (1, 0, 2, 0), (1, 0, 0, 2)]),
        _make([(2, -1)], 1, [(1, 1, 1, 0), (1, 0, 1, 1)]),
    ]

    states: List[AutoionizationState] = []
    for M in manifolds:
        e_ion = M.ion_energy()
        for tM in range(1, 13, 2):
            for L in M.dr_levels(tM, Z=Z, bsp_data=bsp, e_ion=e_ion):
                states.append(AutoionizationState(
                    label=L.dominant[:48],
                    E_resonance_eV=L.E_res_au * HARTREE_EV,
                    J=tM / 2.0,
                    gamma_auger_per_s=L.gamma_auger_per_s,
                    a_radiative_per_s=L.a_radiative_per_s,
                ))
    return states


def dr_calculate(
    initial_atom,
    target_iso,
    *,
    n_max: int = 10,
    l_max: int | None = None,
    T_eV=1000.0,
    initial_state_g: float | None = None,
    initial_state_label: str = "",
    min_resonance_eV: float = 1e-9,
    auger_rate_fn: Callable[[DRResonanceSpec, float], float] | None = None,
    radiative_rate_fn: Callable[
        [DRResonanceSpec, float],
        float | tuple[float, dict[str, float]],
    ] | None = None,
    energy_fn: Callable[[DRResonanceSpec], float] | None = None,
) -> DRRateCoefficient:
    """End-to-end dielectronic-recombination rate driver.

    ``initial_atom`` is the recombining ion and ``target_iso`` is the
    recombined ion ground configuration, either as an ``Atom``-like object or
    a configuration string.

    Default path (no hooks) — ab-initio determinant-CI manifolds
    -----------------------------------------------------------
    When no external width/energy hooks are supplied, the driver builds
    the resonance manifolds with :func:`_dr33_manifold_states`
    (:class:`~atomx.scattering.autoionization.mscheme_ci.ResonanceManifold`
    ``dr_levels``: ab-initio resonance energies, exact determinant Auger
    widths, and E1 stabilizing widths) and feeds them to
    :func:`dr_rate_coefficient` unchanged.  This is the FAC-verified
    physics (Fe XVII dr33: α ratios 0.49/0.99/1.08/1.17 at 1/5/10/50 MK
    against the complete clean-room FAC reference; the 1-MK deficit is
    Boltzmann suppression from the documented ~+40 eV frozen-orbital
    E_res offset, not a width error).  Currently implemented for Ne-like
    recombining ions (core ``1s2 2s2 2p6``); the Δn=0 dr33 (3l3l')
    complex is computed regardless of ``n_max``/``l_max`` (higher-n
    capture manifolds, e.g. dr34, are a documented follow-up).  Other
    systems raise :class:`NotImplementedError`.

    Hook path (external widths)
    ---------------------------
    Passing any of ``auger_rate_fn`` / ``radiative_rate_fn`` /
    ``energy_fn`` selects the legacy enumeration path: doubly-excited
    resonances up to ``n_max``/``l_max`` are enumerated and the supplied
    hooks provide the widths (an omitted hook defaults to the coarse
    ab-initio width factory when a solved result is present — note this
    coarse path historically overshoots FAC by orders of magnitude; see
    ``benchmarks/REPORT.md`` §11).  This path is intended for feeding
    externally computed (e.g. FAC) widths/energies through the Burgess
    kernel, as in ``dielectronic_recombination.fac_reference``.

    ``energy_fn`` overrides the resonance energy per spec, useful when
    benchmarking against an external level list whose energies are
    trusted more than the one-electron estimate.
    """

    if hasattr(initial_atom, "solve") and not getattr(initial_atom, "converged", False):
        initial_atom.solve()

    g_i = (
        _initial_statistical_weight(initial_atom)
        if initial_state_g is None else float(initial_state_g)
    )
    label = initial_state_label or getattr(initial_atom, "_config_strings", [""])[0]

    if auger_rate_fn is None and radiative_rate_fn is None and energy_fn is None:
        # Verified ab-initio manifold pipeline (raises NotImplementedError
        # for unsupported systems rather than silently falling back to the
        # old hydrogenic-width placeholders).
        states = [
            s for s in _dr33_manifold_states(initial_atom)
            if s.E_resonance_eV >= min_resonance_eV
        ]
        return dr_rate_coefficient(
            states,
            T_eV,
            initial_state_g=g_i,
            initial_state_label=label,
        )

    # ---- hook path -------------------------------------------------------
    # Build a minimal structure result for lightweight Atom-like mocks so the
    # ab-initio width factories can run when only one hook is supplied.
    if (
        not has_solved_result(initial_atom)
        and getattr(initial_atom, "orbital_energies", None)
        and (auger_rate_fn is None or radiative_rate_fn is None)
    ):
        # Best-effort: lightweight mocks may lack Z / a parseable core
        # config; in that case fall back to the screened-hydrogenic default
        # widths rather than failing the whole driver.
        try:
            built_result = _build_mcdhf_result_for_dr(
                initial_atom, n_max, l_max if l_max is not None else n_max
            )
        except Exception:
            built_result = None
        if built_result is not None:
            initial_atom._result = built_result
            if getattr(initial_atom, "result", None) is None:
                initial_atom.result = built_result

    specs = enumerate_dr_resonances(
        initial_atom,
        target_iso,
        n_max=n_max,
        l_max=l_max,
    )
    if has_solved_result(initial_atom):
        if auger_rate_fn is None:
            auger_rate_fn = make_abinitio_auger_rate_fn(initial_atom)
        if radiative_rate_fn is None:
            radiative_rate_fn = make_abinitio_radiative_rate_fn(initial_atom)
    resonances = build_autoionization_states(
        specs,
        initial_atom,
        min_resonance_eV=min_resonance_eV,
        auger_rate_fn=auger_rate_fn,
        radiative_rate_fn=radiative_rate_fn,
        energy_fn=energy_fn,
    )
    return dr_rate_coefficient(
        resonances,
        T_eV,
        initial_state_g=g_i,
        initial_state_label=label,
    )
