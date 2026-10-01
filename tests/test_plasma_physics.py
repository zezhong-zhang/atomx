"""Physics gates for the plasma / collision layer (review C5 + M4).

Each gate pins a limit the pre-2026-09 code could not reach; the measured
"before" numbers are recorded in the docstrings so a regression is
recognisable without re-deriving the bug.

1. ``crm`` had no spontaneous radiative recombination, so the only
   recombination was three-body (``n_e^2 beta``): with a Planck field at
   ``T_rad = T_e`` the balance sat above Saha (10.7x at n_e = 1e18 for the
   channel used here; 98.3x in the review with their EII rate) and no
   density-independent coronal balance existed.
2. The DR prefactor was the closed form with kT in Rydberg, 50.186x too
   small for T in eV.
3. ``exp(-dE/kT)`` underflow (dE/kT >~ 745) zeroed *both* collisional
   directions, so a 2-level system returned [0.5, 0.5] instead of
   Boltzmann.
4. ``stewart_pyatt_potential`` jumped by ~250 eV at the u = 10 branch
   switch because its interpolation length collapsed to 0.
5. ``average_atom_thomas_fermi`` never iterated mu, returning 769.4 bound
   electrons for Fe (Z = 26) at 0.01 eV.
6. ``plasma/rates.effective_collision_rate`` passed ``sqrt(T_eV)`` to the
   ``8.629e-6 / sqrt(T_K)`` constant, a constant 107.72x overestimate (and
   called the removed ``np.trapz`` alias, so it raised on numpy 2.x).

Markers: none of these need a solved atom, so the whole module runs in the
fast (``-m "not slow"``) selection.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants as sc
from scipy.special import exp1

from atomx.constants import (
    ALPHA,
    BOHR_RADIUS_M,
    BOLTZMANN_EV,
    HARTREE_EV,
    RYDBERG_EV,
)
from atomx.plasma.average_atom import average_atom_thomas_fermi
from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    CRMBoundFreeChannel,
    CRMLevel,
    CRMTransition,
    _hydrogenic_sigma0_cm2,
    radiative_recombination_rate_cm3_s,
    saha_ionization_density_cm3,
    three_body_recombination_rate_cm6_s,
)
from atomx.plasma.rates import effective_collision_rate, maxwellian_rate
from atomx.plasma.screening import (
    debye_length,
    debye_potential,
    ion_sphere_potential,
    ion_sphere_radius,
    stewart_pyatt_potential,
    stewart_pyatt_screening_length,
)
from atomx.scattering.autoionization import AutoionizationState
from atomx.scattering.dielectronic_recombination.solver import (
    _DR_PREFACTOR_CM3_S_EV_32,
    dr_rate_coefficient,
)

CHI_EV = 100.0
TE_EV = 100.0
G_BOUND = 1
G_CONTINUUM = 2


def _two_level_crm(chi_eV=CHI_EV, *, eii_cm3_s=1.0e-8, **channel_kwargs):
    """Bound + continuum pair with one bound-free channel."""
    return CollisionalRadiativeModel(
        [CRMLevel("bound", 0.0, g=G_BOUND),
         CRMLevel("continuum", chi_eV, g=G_CONTINUUM)],
        [],
        bound_free_channels=[
            CRMBoundFreeChannel(
                bound=0,
                continuum=1,
                ionization_potential_eV=chi_eV,
                g_bound=G_BOUND,
                g_continuum=G_CONTINUUM,
                eii_rate_cm3_s=eii_cm3_s,
                **channel_kwargs,
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1.  CRM radiative recombination / LTE limit
# ---------------------------------------------------------------------------

def test_crm_lte_limit_recovers_saha():
    """Planck field at T_rad = T_e must give exactly the Saha balance.

    Pre-fix (no RR channel): n_c/n_b / (Saha/n_e) = 10.728 / 1.000973 /
    1.000010 at n_e = 1e18 / 1e22 / 1e24 for this channel.  With the
    Milne-consistent RR channel the cancellation is exact at every
    density, not merely at high density.
    """
    crm = _two_level_crm()
    saha = saha_ionization_density_cm3(
        CHI_EV, TE_EV, g_bound=G_BOUND, g_continuum=G_CONTINUUM)

    ratios = {}
    for ne in (1.0e18, 1.0e22, 1.0e24):
        populations = crm.steady_state(
            ne, TE_EV, radiation_temperature_eV=TE_EV)
        ratios[ne] = (populations[1] / populations[0]) / (saha / ne)

    # measured 1.000000000000000 / 0.9999999999999999 / 0.9999999999999998
    for ne in (1.0e18, 1.0e22, 1.0e24):
        assert pytest.approx(1.0, rel=1e-9) == ratios[ne]


def test_hydrogenic_sigma0_matches_the_stobbe_threshold_closed_form():
    """sigma_0 = (2^9 pi^2 / 3) alpha a_0^2 e^{-4} = 6.304318e-18 cm^2.

    ``e`` here is EULER'S NUMBER (``exp(-4)``), NOT the elementary charge:
    reading it as the charge gives 3.442e-16 cm^2, 54.6x too large.  This
    is the Kramers/Unsöld hydrogenic threshold cross-section for n = 1
    (the Stobbe Gaunt factor at threshold is absorbed in the prefactor),
    written in terms of the fine-structure constant ``alpha`` and the Bohr
    radius ``a_0``, so it is derivable without atomx.  The sibling module
    pins the same number against NIST at 1 %
    (``test_hydrogenic_cross_sections.py:18``).

    Measured: module literal 6.3043e-18 vs closed form 6.304318e-18
    — 2.87e-6 relative (the literal is rounded to 5 s.f.), which is why the
    gate is at 2e-5 and not 1 %.
    """
    a0_cm = BOHR_RADIUS_M * 100.0
    sigma0_closed = (2 ** 9 * np.pi ** 2 / 3.0) * ALPHA * a0_cm ** 2 * np.exp(-4.0)
    assert pytest.approx(6.304318e-18, rel=1e-6, abs=0.0) == sigma0_closed  # the closed form
    assert pytest.approx(sigma0_closed, rel=2e-5, abs=0.0) == _hydrogenic_sigma0_cm2(RYDBERG_EV)


_MILNE_POINTS = (
    # chi_eV, T_eV, g_bound, g_continuum
    (100.0, 100.0, 1, 2), (13.6, 13.6, 1, 1), (20.0, 10.0, 1, 1),
    (1.0, 0.5, 1, 1), (13.6, 5.0, 2, 1), (100.0, 10.0, 1, 1),
    (13.6, 0.86, 1, 1),
)


@pytest.mark.parametrize("chi, te, g_b, g_c", _MILNE_POINTS)
def test_total_recombination_in_planck_bath_matches_integral(chi, te, g_b, g_c):
    """Spontaneous + stimulated RR = PI_Planck / Saha in an explicit bath.

    The production rate integrates the spontaneous/stimulated Milne
    cross section. This independent check derives PI in a Planck field
    and divides by the Saha density:

        PI = Int_chi^inf sigma(E) c n_ph(E) dE,
        sigma(E) = sigma_0 (chi/E)^3,      (Kramers)
        n_ph(E) dE = 8 pi E^2 / (h^3 c^3) * dE / (e^{E/kT} - 1),

    i.e. the E^2 of the photon density cancels the (chi/E)^3 above the
    threshold and leaves

        PI = 8 pi sigma_0 chi^3 / (h^3 c^2) * Int_{chi/kT}^inf dx/(x(e^x-1))
           = 8 pi sigma_0 chi^3 / (h^3 c^2) * sum_{n>=1} E1(n chi / kT),

    with ``E1`` the exponential integral (the expansion
    ``1/(e^x-1) = sum_n e^{-nx}``).  chi is in erg, so the prefactor is a
    rate in s^-1 once divided by a Saha density in cm^-3.

    The two derivations must agree in normalization, degeneracies and
    stimulated emission. Adaptive integration makes a tight numerical
    comparison possible even when chi/T is large.
    """
    ev_to_erg = sc.e * 1.0e7
    h_erg_s = sc.h * 1.0e7
    c_cm_s = sc.c * 100.0
    # The test's own sigma_0: the same closed form as the test above, with
    # the hydrogenic Z-scaling of the threshold cross-section
    # (chi ~ Z^2 Ry => sigma_K(threshold) ~ 1/Z^2 ~ Ry/chi).  Deliberately
    # NOT ``_hydrogenic_sigma0_cm2``: sharing that helper would hide exactly
    # the mutation (a wrong prefactor) this gate exists to catch.
    a0_cm = BOHR_RADIUS_M * 100.0
    sigma0 = ((2 ** 9 * np.pi ** 2 / 3.0) * ALPHA * a0_cm ** 2 * np.exp(-4.0)
              * (RYDBERG_EV / chi))
    series = float(sum(exp1(n * chi / te) for n in range(1, 500)))
    pi_rate = (8.0 * np.pi * sigma0 * (chi * ev_to_erg) ** 3
               / (h_erg_s ** 3 * c_cm_s ** 2)) * series
    saha = saha_ionization_density_cm3(chi, te, g_bound=g_b, g_continuum=g_c)
    alpha = radiative_recombination_rate_cm3_s(
        chi, te, g_bound=g_b, g_continuum=g_c, radiation_temperature_eV=te)

    assert alpha > 0.0
    assert pytest.approx(pi_rate / saha, rel=2e-10, abs=0.0) == alpha


def test_crm_rate_matrix_has_a_separate_rr_channel():
    """n_e alpha_RR adds to (does not replace) the three-body term."""
    ne, chi, te = 1.0e12, CHI_EV, TE_EV
    crm = _two_level_crm(chi, eii_cm3_s=2.0e-8)
    beta = three_body_recombination_rate_cm6_s(
        2.0e-8, chi, te, g_bound=G_BOUND, g_continuum=G_CONTINUUM)
    alpha_rr = radiative_recombination_rate_cm3_s(
        chi, te, g_bound=G_BOUND, g_continuum=G_CONTINUUM)

    matrix = crm._build_rate_matrix(ne, te)
    assert pytest.approx(ne * ne * beta + ne * alpha_rr, rel=1e-12, abs=0.0) == matrix[0, 1]
    assert matrix[0, 1] > ne * ne * beta

    without_rr = crm._build_rate_matrix(
        ne, te, include_radiative_recombination=False)
    assert pytest.approx(ne * ne * beta, rel=1e-12, abs=0.0) == without_rr[0, 1]


def test_crm_reaches_a_coronal_balance():
    """At low density n_c/n_b must stop depending on n_e.

    The coronal balance is n_c/n_b = S / (S/Saha + alpha_RR); with only
    three-body recombination it was n_c/n_b = Saha/n_e, i.e. 1/n_e, so no
    coronal limit existed (pre-fix 4.44e18 / 4.44e14 / 4.44e10 at
    n_e = 1e6 / 1e8 / 1e10 with no radiation field).
    """
    crm = _two_level_crm()
    fractions = []
    for ne in (1.0e6, 1.0e8, 1.0e10):
        populations = crm.steady_state(ne, TE_EV)
        fractions.append(populations[1] / populations[0])

    assert fractions[0] > 0.0
    assert pytest.approx(fractions[0], rel=1e-6) == fractions[1]
    assert pytest.approx(fractions[0], rel=1e-6) == fractions[2]


# ---------------------------------------------------------------------------
# 2.  Dielectronic-recombination prefactor
# ---------------------------------------------------------------------------

def test_dr_prefactor_matches_the_ev_convention_closed_form():
    """P = (h^2 / 2 pi m_e kT)^{3/2} with kT in eV, in cm^3.

    Pre-fix the module carried 6.60305e-24, the same closed form with kT
    in Rydberg — 13.605693^{3/2} = 50.186x too small.
    """
    closed_ev = (sc.h ** 2 / (2.0 * np.pi * sc.m_e * sc.e)) ** 1.5 * 1.0e6
    closed_ry = (sc.h ** 2 / (
        2.0 * np.pi * sc.m_e * sc.e * RYDBERG_EV)) ** 1.5 * 1.0e6

    assert pytest.approx(closed_ev, rel=1e-9, abs=0.0) == _DR_PREFACTOR_CM3_S_EV_32
    assert pytest.approx(closed_ry * RYDBERG_EV ** 1.5, rel=1e-9, abs=0.0) == (
        _DR_PREFACTOR_CM3_S_EV_32)
    assert pytest.approx(3.312830e-22, rel=1e-6, abs=0.0) == _DR_PREFACTOR_CM3_S_EV_32
    # the historical constant was the Rydberg-convention value
    assert pytest.approx(6.60305e-24, rel=1e-3, abs=0.0) == closed_ry


def test_dr_prefactor_is_consistent_with_saha_detailed_balance():
    """The prefactor must be 2 exp(-chi/T) / Saha at every (chi, T).

    Saha = (2 / lambda_th^3) exp(-chi/T) with lambda_th^3 = P T^{-3/2},
    so P = T^{3/2} * 2 exp(-chi/T) / Saha — the same Saha function the CRM
    uses, which is what makes DR/RR/collisional balance mutually
    consistent.  Measured agreement 2e-9 (limited by the duplicated
    CODATA literals inside ``saha_ionization_density_cm3``).
    """
    for chi, te in ((100.0, 100.0), (13.6, 5.0), (20.0, 10.0), (1.0, 0.5)):
        saha = saha_ionization_density_cm3(chi, te)  # g_bound = g_continuum = 1
        prefactor_from_saha = te ** 1.5 * 2.0 * np.exp(-chi / te) / saha
        assert pytest.approx(prefactor_from_saha, rel=1e-6, abs=0.0) == (
            _DR_PREFACTOR_CM3_S_EV_32)


def test_dr_rate_coefficient_uses_the_corrected_prefactor():
    """One resonance, checked against the Burgess closed form.

    E = 10 eV, Gamma_A = Gamma_r = 1e13 s^-1, J = 1/2 -> g_d = 2,
    g_i = 1: omega = 1, branch = 5e12 s^-1, T_e = 10 eV.
    Pre-fix alpha = 3.84e-13 cm^3/s; with the eV prefactor it is
    1.927e-11 cm^3/s (50.2x larger).
    """
    resonance = AutoionizationState(
        label="X", E_resonance_eV=10.0, J=0.5,
        gamma_auger_per_s=1.0e13, a_radiative_per_s=1.0e13)
    alpha = dr_rate_coefficient(
        [resonance], 10.0, initial_state_g=1.0).alpha_cm3_s[0]
    closed = (_DR_PREFACTOR_CM3_S_EV_32 * 10.0 ** -1.5
              * (2.0 / 2.0) * 0.5e13 * np.exp(-1.0))
    assert pytest.approx(closed, rel=1e-12, abs=0.0) == alpha
    assert pytest.approx(1.92697e-11, rel=1e-4, abs=0.0) == alpha


# ---------------------------------------------------------------------------
# 3.  Collisional de-excitation under Boltzmann underflow
# ---------------------------------------------------------------------------

def test_crm_deexcitation_survives_boltzmann_underflow():
    """dE/kT = 750: C_exc underflows, C_deexc must not.

    Pre-fix the ``if C_exc > 0`` gate dropped both directions, leaving a
    singular matrix whose least-squares solution was [0.5, 0.5].
    """
    te = 0.8
    ne = 1.0e12
    crm = CollisionalRadiativeModel(
        [CRMLevel("lower", 0.0, g=1), CRMLevel("upper", 600.0, g=3)],
        [CRMTransition(upper=1, lower=0, A_rate=0.0, gf=1.0)],
    )
    matrix = crm._build_rate_matrix(ne, te)

    # van Regemorter effective Gaunt factor for x >> 1 (corrected
    # constant: the OLD closed form lacked the 1/dE_eV factor — the
    # review's "test pins the wrong formula" finding; q now carries
    # 8.63e-6 * gf / (dE g sqrt(T)) * gbar)
    x = 600.0 / te
    gaunt = max(0.2, 0.276 * (x + 1.0) / (x * (x + 2.0)))
    c_deexc_closed = (8.63e-6 * 1.0 / (600.0 * 1.0 * np.sqrt(te))
                      * gaunt * (1.0 / 3.0))

    assert pytest.approx(c_deexc_closed, rel=1e-3, abs=0.0) == matrix[0, 1] / ne
    assert pytest.approx(c_deexc_closed, rel=1e-12, abs=0.0) == matrix[0, 1] / ne
    assert matrix[1, 0] == 0.0  # exp(-750) is genuinely zero

    populations = crm.steady_state(ne, te)
    assert pytest.approx(crm.boltzmann_populations(te), abs=1e-12) == populations
    assert pytest.approx(1.0, abs=1e-12) == populations[0]


def test_crm_collisional_detailed_balance_is_exact_when_finite():
    """Both directions must satisfy C_deexc = C_exc (g_l/g_u) exp(+dE/kT)."""
    te = 10.0
    ne = 1.0e12
    crm = CollisionalRadiativeModel(
        [CRMLevel("lower", 0.0, g=1), CRMLevel("upper", 5.0, g=3)],
        [CRMTransition(upper=1, lower=0, A_rate=0.0, gf=1.0)],
    )
    matrix = crm._build_rate_matrix(ne, te)
    c_exc = matrix[1, 0] / ne
    c_deexc = matrix[0, 1] / ne
    assert c_exc > 0.0
    assert pytest.approx(c_exc * (1.0 / 3.0) * np.exp(5.0 / te), rel=1e-12, abs=0.0) == c_deexc


# ---------------------------------------------------------------------------
# 4.  Stewart-Pyatt screening
# ---------------------------------------------------------------------------

def _temperature_for_u(z_ion, ne_cm3, u):
    """T_e that puts the coupling parameter R_IS/lambda_D at ``u``."""
    r_is = ion_sphere_radius(z_ion, ne_cm3)
    ne_au = ne_cm3 * (BOHR_RADIUS_M * 100.0) ** 3
    lam_d = r_is / u
    return lam_d ** 2 * 4.0 * np.pi * ne_au * HARTREE_EV


def test_stewart_pyatt_screening_length_limits():
    lam_d = 0.5
    # u -> 0: Debye length; u -> inf: (2/3) R_IS (the ion-sphere value for
    # which the Yukawa shift Z/lam_eff matches 3 Z_ion/(2 R_IS))
    assert pytest.approx(lam_d, rel=1e-9) == (
        stewart_pyatt_screening_length(lam_d, 1e-9 * lam_d))
    r_is = 1.0e4 * lam_d
    assert pytest.approx(2.0 / 3.0 * r_is, rel=1e-6) == (
        stewart_pyatt_screening_length(lam_d, r_is))
    # monotone in between
    lengths = [stewart_pyatt_screening_length(lam_d, u * lam_d)
               for u in np.geomspace(1e-3, 1e3, 40)]
    assert np.all(np.diff(lengths) > 0.0)


def test_stewart_pyatt_potential_is_continuous_at_branch_switch():
    """No jump at u = 0.1 or u = 10, where the old code switched branches.

    Pre-fix at Z = 6, Z_ion = 1, n_e = 1e24, u = 10: V(0.5 R_IS) was
    -3.99e-8 Ha on the interpolation branch and -9.063 Ha on the
    ion-sphere branch (a 247 eV discontinuity, relative jump 1.0).
    """
    z, z_ion, ne = 6, 1, 1.0e24
    r = 0.5 * ion_sphere_radius(z_ion, ne)

    for u_switch in (0.1, 10.0):
        for delta in (1e-6, 1e-9):
            v_lo = stewart_pyatt_potential(
                z, z_ion, ne,
                _temperature_for_u(z_ion, ne, u_switch * (1.0 - delta)))(r)
            v_hi = stewart_pyatt_potential(
                z, z_ion, ne,
                _temperature_for_u(z_ion, ne, u_switch * (1.0 + delta)))(r)
            assert abs(v_hi - v_lo) / abs(v_lo) < 1e-6

    # the two limits are reproduced exactly, as the old branches did
    te_debye = _temperature_for_u(z_ion, ne, 0.05)
    assert pytest.approx(debye_potential(z, ne, te_debye)(r), rel=1e-12) == (
        stewart_pyatt_potential(z, z_ion, ne, te_debye)(r))
    te_ion_sphere = _temperature_for_u(z_ion, ne, 20.0)
    assert pytest.approx(
        ion_sphere_potential(z, z_ion, ne)(r), rel=1e-12) == (
        stewart_pyatt_potential(z, z_ion, ne, te_ion_sphere)(r))


def test_stewart_pyatt_potential_log_sweep_has_no_silent_jump():
    """Adjacent coupling points must not step (measured max 1.7e-2)."""
    z, z_ion, ne = 6, 1, 1.0e24
    r = 0.5 * ion_sphere_radius(z_ion, ne)
    values = np.array([
        stewart_pyatt_potential(z, z_ion, ne, _temperature_for_u(z_ion, ne, u))(r)
        for u in np.geomspace(0.02, 50.0, 60)
    ])
    jumps = np.abs(np.diff(values)) / np.abs(values[:-1])
    assert jumps.max() < 0.05


def test_stewart_pyatt_screening_length_reduces_to_debye_in_weak_coupling():
    """lam_eff -> lam_D as u -> 0 (the u <= 0.1 end is the Debye potential)."""
    z_ion, ne, te = 1, 1.0e20, 1000.0
    lam_d = debye_length(ne, te)
    r_is = ion_sphere_radius(z_ion, ne)
    u = r_is / lam_d
    assert u < 0.1
    # small-u series: (2/3)u^3 / expm1((2/3)log1p(u^3)) = 1 + u^3/6 + O(u^6)
    assert pytest.approx(lam_d * (1.0 + u ** 3 / 6.0), rel=1e-6) == (
        stewart_pyatt_screening_length(lam_d, r_is))
    assert pytest.approx(1.0, rel=1e-3) == (
        stewart_pyatt_screening_length(lam_d, r_is) / lam_d)


# ---------------------------------------------------------------------------
# 5.  Average-atom electron budget
# ---------------------------------------------------------------------------

def _total_occupation(result):
    return sum(result.occupations.values())


def test_average_atom_never_binds_more_than_z_electrons():
    """Fe at T = 0.01 eV, n_e = 1e20: pre-fix Z_bar = 0 with 769.41 bound.

    The free-gas chemical potential at that density sits above the bound
    shells, so mu must be iterated down until the shells hold exactly Z.
    """
    result = average_atom_thomas_fermi(Z=26, T_eV=0.01, ne_cm3=1.0e20)
    total = _total_occupation(result)

    assert total <= 26.0 + 1e-9
    assert pytest.approx(26.0, rel=1e-12) == total
    assert pytest.approx(26.0 - result.Z_bar, rel=1e-12) == total
    assert pytest.approx(0.0, abs=1e-9) == result.Z_bar
    assert result.budget_saturated is True
    assert result.mu_eV < result.mu_free_eV


@pytest.mark.parametrize(
    "t_ev,ne_cm3",
    [
        (0.01, 1.0e20),
        (0.1, 1.0e22),
        (1.0, 1.0e15),
        (100.0, 1.0e22),
        (5000.0, 1.0e20),
    ],
)
def test_average_atom_electron_budget_closes(t_ev, ne_cm3):
    """Sigma occupations = Z - Z_bar and never exceeds Z, at any (T, n_e)."""
    result = average_atom_thomas_fermi(Z=26, T_eV=t_ev, ne_cm3=ne_cm3)
    total = _total_occupation(result)

    assert 0.0 <= total <= 26.0 + 1e-9
    assert 0.0 <= result.Z_bar <= 26.0
    assert pytest.approx(26.0 - result.Z_bar, rel=1e-9) == total
    # each (n, l) stays inside its own 2(2l+1) capacity
    for (_n, l), occupation in result.occupations.items():
        assert 0.0 <= occupation <= 2 * (2 * l + 1) + 1e-12


# ---------------------------------------------------------------------------
# 6.  Maxwellian collisional rate coefficients (plasma/rates.py)
# ---------------------------------------------------------------------------

def test_effective_collision_rate_uses_kelvin_in_the_8_629e_6_constant():
    """q = 8.629e-6 Omega / (g_i sqrt(T_K)) for constant Omega.

    Pre-fix ``sqrt(temperature_eV)`` made the rate a constant
    ``sqrt(1/BOLTZMANN_EV) = 107.7235`` times too large (and the call
    raised AttributeError on numpy 2.x, where ``np.trapz`` is gone).
    """
    omega0 = 3.0
    for te in (1.0, 10.0, 100.0):
        energies = np.linspace(0.0, 40.0 * te, 200_001)
        strengths = np.full_like(energies, omega0)
        rate = effective_collision_rate(
            strengths, energies, te, g_i=1, threshold_eV=0.0)
        closed = 8.629e-6 * omega0 / np.sqrt(te / BOLTZMANN_EV)
        assert pytest.approx(closed, rel=1e-6, abs=0.0) == rate
        # the eV-temperature variant is 107.72x larger
        assert pytest.approx(1.0 / np.sqrt(BOLTZMANN_EV), rel=1e-6) == (
            (8.629e-6 * omega0 / np.sqrt(te)) / rate)

    # where the 8.629e-6 constant comes from: sqrt(8/(pi m_e)) pi a_0^2 Ry
    # / sqrt(k_B) in cgs (exact value 8.62913e-6)
    m_e_cgs = 9.109_383_7015e-28
    k_b_erg = 1.380_649e-16
    ry_erg = RYDBERG_EV * 1.602_176_634e-12
    constant = (np.sqrt(8.0 / (np.pi * m_e_cgs))
                * np.pi * (BOHR_RADIUS_M * 100.0) ** 2 * ry_erg
                / np.sqrt(k_b_erg))
    assert pytest.approx(8.629e-6, rel=1e-4, abs=0.0) == constant


def test_effective_collision_rate_matches_maxwellian_average():
    """Independent path: sigma(E) = pi a0^2 Omega(E_f)/(g_i E/Ry) + <sigma v>.

    ``effective_collision_rate`` uses the rounded 8.629e-6 constant while
    ``maxwellian_rate`` integrates the cgs Maxwellian explicitly, so the
    two agree to the rounding of that constant: measured 1.53e-5 relative.
    """
    bohr_cm = BOHR_RADIUS_M * 100.0
    for te, threshold, omega0, g_i in (
        (10.0, 5.0, 4.0, 1),
        (100.0, 50.0, 2.0, 3),
        (1.0, 0.5, 1.0, 1),
    ):
        final_energies = np.linspace(0.0, 60.0 * te + threshold, 200_001)
        strengths = np.full_like(final_energies, omega0)
        from_strengths = effective_collision_rate(
            strengths, final_energies, te, g_i=g_i, threshold_eV=threshold)

        incident = final_energies + threshold
        cross_section = (np.pi * bohr_cm ** 2) * omega0 / (
            g_i * incident / RYDBERG_EV)
        from_cross_section = maxwellian_rate(cross_section, incident, te)

        assert pytest.approx(from_cross_section, rel=1e-4, abs=0.0) == from_strengths


def test_maxwellian_rate_matches_the_ideal_gas_closed_form():
    """Constant cross-section: <sigma v> = sigma_0 sqrt(8 kT/(pi m_e))."""
    sigma0 = 1.0e-18
    for te in (0.5, 10.0, 1000.0):
        energies = np.linspace(0.0, 40.0 * te, 200_001)
        sigma = np.full_like(energies, sigma0)
        rate = maxwellian_rate(sigma, energies, te)
        kt_erg = te * 1.602_176_634e-12
        closed = sigma0 * np.sqrt(8.0 * kt_erg / (np.pi * 9.109_383_7015e-28))
        assert pytest.approx(closed, rel=1e-6, abs=0.0) == rate
