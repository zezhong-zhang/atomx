"""Normal Mass Shift (NMS) and Specific Mass Shift (SMS).

Normal mass shift
-----------------
The Breit–Salpeter one-body mass-shift operator in the non-relativistic
limit is

    H_NMS = + (m_e / M) · Σ_i (p_i² / 2 m_e) = + (m_e / M) · T_e,

where M is the *nuclear* mass (atomic mass minus Z electron masses) and
T_e is the total electron kinetic energy.  A finite nuclear mass makes a
bound level *less* bound, so the level rises and the shift is positive:

    ΔE_NMS = + (m_e / M) · ⟨T_e⟩ > 0.                              (1)

For a Coulomb-bound system the virial theorem states ⟨T_e⟩ = −E_total
(non-relativistic) with a small relativistic correction of order
``(Zα)²``.  atomx uses the virial theorem as a *default* estimator and
accepts an explicit ``kinetic_energy`` override for systems where the
Dirac correction matters.

Specific mass shift
-------------------
The SMS comes from the two-body piece of the mass-shift operator,

    H_SMS = +(1/M) · Σ_{i<j} p_i · p_j,

and is sensitive to electron-electron correlations.  A full ab-initio
SMS evaluation on an SCF determinant requires two-electron momentum
integrals and is non-trivial in the Dirac-Fock basis; atomx does not
yet implement it automatically.

Instead, this module exposes an **SMS parameter** ``K_SMS`` (in atomic
units of ``h / (M_e · a_0²)``) that users can supply externally — for
example from the Anisimova *et al.* 2023 model-QED recoil tables
(``papers/2301.04132v1.pdf``) or from an MBPT calculation in a
complementary code.  The SMS contribution to an isotope shift is then

    δν_SMS(A → A')  =  K_SMS · (1/M_A − 1/M_A').                   (2)

Relativistic generalisation
---------------------------
For a fully-relativistic treatment the one-body mass-shift operator
becomes (Shabaev 1985, 1988)

    H_NMS^{rel} = (1/(2M)) · Σ_i [ p_i² − (Zα/r_i) α_i · p_i
                                     − (Zα/r_i)(α_i · r̂_i)(r̂_i · p_i) ],

and the model-operator expansion of Anisimova–Shabaev *et al.* (2023)
provides κ-dependent local + non-local potentials that give the full
QED result. The virial-theorem helper here reproduces the leading
non-relativistic piece. The solver correction stack instead uses the
one-body Breit recoil radial operator in ``atomx.corrections.recoil``
by default; two-body and radiative recoil remain absent there.

Reference
---------
V. M. Shabaev, *Sov. J. Nucl. Phys.* **47**, 69 (1988).
I. S. Anisimova *et al.*, arXiv:2301.04132 (2023) — model-operator
approach to nuclear recoil in many-electron atoms.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.constants import PROTON_ELECTRON_MASS_RATIO


@dataclass
class MassShift:
    """Normal mass shift of an atomic level / total energy.

    Attributes
    ----------
    E_NMS_Ha : float
        NMS shift in Hartree:  ``E_NMS = +(m_e/M) · ⟨T_e⟩``.  Positive
        for bound states (T > 0, M > 0 → +1/M × T > 0): a finite nuclear
        mass raises the level.
    E_NMS_MHz : float
        Same, converted to MHz via :data:`atomx.properties.hyperfine.HARTREE_MHZ`.
    kinetic_energy_au : float
        ``⟨T_e⟩`` used in the calculation (Hartree).  Typically obtained
        from the virial theorem ``T = −E_total`` (see
        :func:`normal_mass_shift` below).
    mass_me : float
        Nuclear mass used, in electron-mass units.
    isotope : str
        Label of the isotope, e.g. ``"H-1"`` or ``"Cs-133"``.
    method : str
        ``"virial"`` or ``"explicit"`` — records how ⟨T_e⟩ was obtained.
    """

    E_NMS_Ha: float
    E_NMS_MHz: float
    kinetic_energy_au: float
    mass_me: float
    isotope: str
    method: str = "virial"


def normal_mass_shift(
    total_energy: float,
    mass_me: float,
    *,
    isotope_label: str = "",
    kinetic_energy: float | None = None,
) -> MassShift:
    """Evaluate the NMS from a converged SCF result.

    Parameters
    ----------
    total_energy : float
        Converged total energy in Hartree.
    mass_me : float
        Nuclear mass in electron-mass units (``A × m_p/m_e`` suffices
        for isotope-shift work to ~0.1 %).  Must be positive.
    isotope_label : str
        Human-readable isotope tag, e.g. ``"H-1"``.
    kinetic_energy : float, optional
        Explicit ⟨T_e⟩ in Hartree.  If ``None``, the virial approximation
        ``T = −E_total`` is used and ``method="virial"`` is recorded.

    Returns
    -------
    :class:`MassShift`
    """
    from atomx.properties.hyperfine.dipole import HARTREE_MHZ

    if mass_me <= 0:
        raise ValueError(f"Nuclear mass must be positive; got {mass_me}.")

    if kinetic_energy is None:
        kinetic = -total_energy
        method = "virial"
    else:
        kinetic = kinetic_energy
        method = "explicit"

    E_NMS_Ha = kinetic / mass_me
    return MassShift(
        E_NMS_Ha=E_NMS_Ha,
        E_NMS_MHz=E_NMS_Ha * HARTREE_MHZ,
        kinetic_energy_au=kinetic,
        mass_me=mass_me,
        isotope=isotope_label,
        method=method,
    )


def default_nuclear_mass_me(Z: int, A: int) -> float:
    """Simple proxy for the nuclear mass in electron-mass units.

    Uses ``A × m_p/m_e`` where ``m_p/m_e ≈ 1836.15``.  Accurate to ~0.1 %
    for isotope-shift work.  For precision spectroscopy, callers should
    instead pass the atomic mass via
    :attr:`atomx.properties.hyperfine.NuclearMoment.mass_me`.
    """
    return A * PROTON_ELECTRON_MASS_RATIO


# ---------------------------------------------------------------------------
# Specific mass shift (user-supplied)
# ---------------------------------------------------------------------------

@dataclass
class SpecificMassShift:
    """Specific mass shift for a given isotope, from a user-supplied K_SMS.

    Attributes
    ----------
    K_SMS_au : float
        SMS coefficient in atomic units.  Positive or negative — the
        sign convention is that δν_SMS(A → A') = K_SMS · (1/M_A − 1/M_A'),
        so a positive K_SMS means the lighter isotope is *blue-shifted*
        relative to the heavier one from SMS alone.
    E_SMS_Ha : float
        Energy contribution ``K_SMS / M_nuc``  in Hartree.
    E_SMS_MHz : float
        Same, in MHz.
    mass_me : float
        Nuclear mass used (electron-mass units).
    isotope : str
    """

    K_SMS_au: float
    E_SMS_Ha: float
    E_SMS_MHz: float
    mass_me: float
    isotope: str = ""


def specific_mass_shift(
    K_SMS_au: float,
    mass_me: float,
    *,
    isotope_label: str = "",
) -> SpecificMassShift:
    """Build a :class:`SpecificMassShift` from a user-supplied K_SMS.

    Parameters
    ----------
    K_SMS_au : float
        SMS coefficient in atomic units (Hartree · m_e).
    mass_me : float
        Nuclear mass in electron-mass units.  Must be positive.
    isotope_label : str
        Informational label.

    Returns
    -------
    :class:`SpecificMassShift`
    """
    from atomx.properties.hyperfine.dipole import HARTREE_MHZ

    if mass_me <= 0:
        raise ValueError(f"Nuclear mass must be positive; got {mass_me}.")

    E_SMS_Ha = K_SMS_au / mass_me
    return SpecificMassShift(
        K_SMS_au=K_SMS_au,
        E_SMS_Ha=E_SMS_Ha,
        E_SMS_MHz=E_SMS_Ha * HARTREE_MHZ,
        mass_me=mass_me,
        isotope=isotope_label,
    )
