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
        = 3.3128 × 10⁻²²  ·  T_e[eV]^{−3/2}
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

from atomx.atom_inputs import Basis
from atomx.config.subshell import L_LETTERS
from atomx.constants import AU_TIME_S, ELECTRON_MASS_KG, HARTREE_EV, HARTREE_J
from atomx.scattering.autoionization.auger import AutoionizationState
from atomx.scattering.dielectronic_recombination.enumerate import (
    DRResonanceSpec,
    _config_string,
    _initial_statistical_weight,
    build_autoionization_states,
    enumerate_dr_resonances,
    resonance_strength_rows,
)
from atomx.scattering.dielectronic_recombination.widths import (
    has_solved_result,
    make_abinitio_auger_rate_fn,
    make_abinitio_radiative_rate_fn,
    solved_result,
)

# --- Universal prefactor for eq. (2) -------------------------------------
#
# Expression: (h² / (2π m_e k_B T))^{3/2} with the temperature written in
# eV, i.e. (h² / (2π m_e · 1 eV))^{3/2}, in cm³ (the 1/s needed to turn
# the widths Γ into a rate coefficient comes from the branching ratio):
#
#   α [cm³/s] = P · T_e[eV]^{-3/2} · ω · ⟨Γ⟩[s⁻¹] · exp(−E/T_e)
#
# Derived here from :mod:`atomx.constants` so the temperature unit is
# unambiguous.  Historically this module carried the closed-form value
# computed with kT in *Rydberg* (6.60305e-24), which is 13.605693^{3/2} =
# 50.186x too small once T is in eV.  The module's own
# ``plasma.crm.saha_ionization_density_cm3`` uses the eV convention, so
# the two must match for DR/RR detailed balance to hold.
_PLANCK_J_S = AU_TIME_S * 2.0 * np.pi * HARTREE_J          # h = 2π ħ
_EV_IN_J = HARTREE_J / HARTREE_EV                          # 1 eV in joules
_DR_PREFACTOR_CM3_S_EV_32 = (
    _PLANCK_J_S ** 2 / (2.0 * np.pi * ELECTRON_MASS_KG * _EV_IN_J)
) ** 1.5 * 1.0e6                                           # m³ -> cm³


def dielectronic_capture_rate_cm3_s(temperature_eV, resonance_energy_eV,
                                   partial_auger_rate_per_s, *, g_resonance, g_continuum):
    """Unbranched inverse of a specified Auger channel, in cm³/s.

    Isolated narrow-resonance Maxwellian impact limit. The resonance energy
    is relative to this channel's continuum level.
    No radiative stabilisation probability is included: an explicit kinetic
    resonance must decay through its separately supplied Auger and A channels.
    Multiplying this capture coefficient by A/sum(widths) gives the isolated
    low-density DR coefficient used by :func:`dr_rate_coefficient`.
    """
    T = np.asarray(temperature_eV, dtype=float)
    if (np.any(~np.isfinite(T)) or np.any(T <= 0)
            or not np.isfinite(resonance_energy_eV) or resonance_energy_eV < 0
            or not np.isfinite(partial_auger_rate_per_s) or partial_auger_rate_per_s < 0
            or not all(np.isfinite(v) and v > 0 for v in (g_resonance, g_continuum))):
        raise ValueError("positive temperature/weights and nonnegative finite resonance energy/width required")
    return (_DR_PREFACTOR_CM3_S_EV_32 * T**(-1.5)
            * g_resonance / (2 * g_continuum) * partial_auger_rate_per_s
            * np.exp(-resonance_energy_eV / T))


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
    if np.any(~np.isfinite(T_eV)) or np.any(T_eV <= 0):
        raise ValueError("Temperatures must be finite and positive (eV).")

    survival = np.ones_like(T_eV, dtype=float)
    ipd_eV = None
    if ne_cm3 is not None:
        ne_cm3 = float(ne_cm3)
        if not np.isfinite(ne_cm3) or ne_cm3 < 0.0:
            raise ValueError("ne_cm3 must be finite and non-negative")
    if ne_cm3 is not None and ne_cm3 > 0.0:
        if Z_ion is None or series_limit_eV is None:
            raise ValueError(
                "density suppression needs Z_ion and series_limit_eV "
                "alongside ne_cm3")
        series_limit_eV = float(series_limit_eV)
        if not np.isfinite(series_limit_eV):
            raise ValueError("series_limit_eV must be finite")
        from atomx.plasma.screening import ionization_potential_depression
        ipd_eV = np.array([
            float(ionization_potential_depression(Z_ion, ne_cm3, t,
                                                  model=ipd_model))
            for t in T_eV])
        if np.any(~np.isfinite(ipd_eV)):
            raise ValueError("density suppression produced non-finite IPD")

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
            if not np.isfinite(e_bind):
                raise ValueError(
                    f"resonance {d.label!r} produced non-finite binding energy")
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

    l_symbols = L_LETTERS
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
                  config=configs,
                  exchange="hf",
                  basis=Basis(ns=40, ks=7),
             )
    struct.solve()
    return struct.result


def _dr33_manifold_states(initial_atom) -> List[AutoionizationState]:
    """Ne-like dr33 states from the shared native manifold constructor.

    The result's local shooting potential or B-spline Fock basis determines
    the virtual orbitals. No new SCF calculation or engine conversion occurs.
    Hole/parity spaces remain separate, with coherent mixing within each.
    Continuum sampling covers a bound on the spectrum of the actual CI
    Hamiltonian, rather than assuming a fixed Fe electron energy at every Z.
    """
    from atomx.scattering.autoionization.mscheme_ci import manifold_from_configs

    unsupported = (
        "The default DR manifold calculation supports solved Ne-like "
        "'1s2 2s2 2p6' ions and the dr33 capture complex. ")
    try:
        core = _config_string(initial_atom)
    except TypeError:
        core = ""
    if sorted(core.split()) != ["1s2", "2p6", "2s2"]:
        raise NotImplementedError(unsupported + f"Got core config {core!r}.")
    result = solved_result(initial_atom)
    if result is None:
        raise NotImplementedError(unsupported + "Solve the recombining ion first.")
    Z = int(result.Z)
    groups = {
        'p_even': ['2p5 3s1 3p1', '2p5 3p1 3d1'],
        'p_odd': ['2p5 3s2', '2p5 3p2', '2p5 3d2', '2p5 3s1 3d1'],
        's_even': ['2s1 3s2', '2s1 3p2', '2s1 3d2', '2s1 3s1 3d1'],
        's_odd': ['2s1 3s1 3p1', '2s1 3p1 3d1'],
    }
    states = []
    for name, configs in groups.items():
        grid_energy = 1000.
        manifold, workspace = manifold_from_configs(result, configs, e_max_eV=grid_energy)
        # Every half-integer J has an M=1/2 component. Gershgorin's upper
        # bound on this complete determinant block covers every resonance,
        # without a second eigenvalue solve. Bound integrals stay native and
        # are independent of the continuum interpolation grid being sized.
        h = manifold.hamiltonian(manifold.determinants(1))
        upper = np.max(np.diag(h)+np.sum(np.abs(h), axis=1)-np.abs(np.diag(h)))
        upper_eV = float((upper-manifold.ion_energy())*HARTREE_EV)
        if upper_eV > grid_energy:
            manifold, workspace = manifold_from_configs(
                result, configs, e_max_eV=upper_eV)
        tm = 1
        while manifold.determinants(tm):
            for index, level in enumerate(manifold.dr_levels(tm, Z=Z, bsp_data=workspace)):
                states.append(AutoionizationState(
                    label=f"{name}:2J={tm}:level={index}",
                    E_resonance_eV=level.E_res_au*HARTREE_EV,
                    J=tm/2., gamma_auger_per_s=level.gamma_auger_per_s,
                    a_radiative_per_s=level.a_radiative_per_s,
                    channels={"ground": level.gamma_auger_per_s},
                ))
            tm += 2
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
    allow_default_widths: bool = False,
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
    :func:`dr_rate_coefficient` unchanged. The matched Fe benchmark uses
    an independently optimized local potential, with aggregate rates within
    2% of FAC for the same dr33 spaces. This does not establish uniform
    accuracy of individual widths. Currently implemented for Ne-like
    recombining ions (core ``1s2 2s2 2p6``); the Δn=1 dr33 (3l3l')
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

    When ``initial_atom`` carries no solved result and a width hook is
    omitted, a compact MCDHF result is built first so the ab-initio
    factories can fill the gap; a failed build raises rather than
    degrading the widths.  Pass ``allow_default_widths=True`` to select
    the semi-empirical screened-hydrogenic default widths
    (:mod:`~atomx.scattering.dielectronic_recombination.enumerate`)
    explicitly instead.

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
    # Build a minimal structure result for lightweight Atom-like inputs so
    # the ab-initio width factories can run when only one hook is supplied.
    # A failed build propagates: silently dropping to the semi-empirical
    # default widths would change the physics without the caller asking
    # (review E7).  ``allow_default_widths=True`` selects those defaults
    # explicitly instead.
    if (
        not has_solved_result(initial_atom)
        and getattr(initial_atom, "orbital_energies", None)
        and (auger_rate_fn is None or radiative_rate_fn is None)
        and not allow_default_widths
    ):
        built_result = _build_mcdhf_result_for_dr(
            initial_atom, n_max, l_max if l_max is not None else n_max
        )
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
    elif auger_rate_fn is None or radiative_rate_fn is None:
        if not allow_default_widths:
            missing = [
                name
                for name, fn in (("auger_rate_fn", auger_rate_fn),
                                 ("radiative_rate_fn", radiative_rate_fn))
                if fn is None
            ]
            raise ValueError(
                f"dr_calculate: initial_atom carries no solved result, so "
                f"the missing width hook(s) {missing} cannot be filled "
                f"ab-initio.  Solve the atom first, supply the hook(s) "
                f"explicitly, or pass allow_default_widths=True to select "
                f"the semi-empirical screened-hydrogenic default widths."
            )
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
