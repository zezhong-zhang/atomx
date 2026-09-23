"""Single-channel Wigner R-matrix scaffold regression tests."""

import numpy as np
import pytest

from atomx.constants import HARTREE_EV, PROTON_ELECTRON_MASS_RATIO
from atomx.physics.dirac import hydrogenic_dirac_energy
from atomx.scattering.r_matrix import (
    RMatrixResult,
    compute_single_channel,
)
from atomx.scattering.r_matrix.channel_table import enumerate_channels
from atomx.scattering.r_matrix.target_states import (
    TargetState,
    helike_target_states,
    heplus_target_states,
)

# NIST ASD ionisation energies (H I 13.598434599702 eV, He II 54.417760 eV).
# These were attributed to a live ``sfac`` pre-run until the S5 audit: the
# literals actually sit 8e-7 (H) and 3.3e-7 (He II) from NIST, i.e. they are
# the NIST values rounded to 8 decimals, not FAC output — and atomx's own
# reduced-mass hydrogenic energy reproduces NIST to 2.5e-6 (H) and 8.3e-6
# (He II).  The 5e-4 gate was ~60x looser than the agreement it claimed and
# would have accepted an infinite-nuclear-mass He II (1.45e-4 off, see
# ``test_heplus_threshold_gate_needs_the_reduced_mass``).
NIST_H_IONIZATION_EV = 13.598434599702
NIST_HEPLUS_IONIZATION_EV = 54.417760
# Mass ratios used for the finite-nuclear-mass correction. The proton
# ratio is already exposed by atomx.constants; the 4He nucleus mass
# ratio (alpha particle / electron) is from CODATA 2018.
M_ALPHA_OVER_E = 7294.29954136
# Measured agreement with NIST: 2.5e-6 (H, 1s) and 8.3e-6 (He II, 1s); the
# gate is a factor 1.2 above the *worst* measured value.
FAC_THRESHOLD_RELTOL = 1.0e-5


def _V_zero(r):
    return np.zeros_like(r)


def _V_attractive_well(r):
    return np.where(r < 5.0, -0.1, 0.0)


def test_returns_RMatrixResult():
    res = compute_single_channel(_V_zero, np.array([1.0]), a_au=10.0,
                                   n_basis=50, l=0)
    assert isinstance(res, RMatrixResult)
    assert res.a_au == 10.0
    assert res.energies_eV.shape == (1,)


def test_inputs_finite_and_real():
    """All outputs are finite real numbers."""
    E = np.geomspace(0.1, 10.0, 10)
    res = compute_single_channel(_V_zero, E, a_au=10.0, n_basis=80, l=0)
    assert np.all(np.isfinite(res.phase_shift_rad))
    assert np.all(np.isfinite(res.cross_section_cm2))
    assert np.all(res.cross_section_cm2 >= 0)


def test_phase_shift_in_range():
    """Phase shifts are wrapped to (-π/2, π/2]."""
    E = np.geomspace(0.1, 10.0, 20)
    res = compute_single_channel(_V_attractive_well, E, a_au=15.0,
                                   n_basis=120, l=0)
    assert np.all(res.phase_shift_rad > -np.pi / 2 - 1e-9)
    assert np.all(res.phase_shift_rad <= np.pi / 2 + 1e-9)


def test_partial_wave_cross_section_units():
    """σ_l = (4π/k²)(2l+1) sin²(δ_l) — bounded by the unitarity limit
    σ_max = 4π(2l+1)/k² (in a₀² before cm² conversion)."""
    E = np.array([1.0])
    res = compute_single_channel(_V_attractive_well, E, a_au=15.0,
                                   n_basis=120, l=0)
    k_au = np.sqrt(2.0 * 1.0 / 27.211)
    sigma_max_a0 = 4.0 * np.pi / (k_au ** 2)
    sigma_a0 = res.cross_section_cm2[0] / 2.8e-17
    assert sigma_a0 <= sigma_max_a0 * 1.001


def test_higher_l_runs_finite():
    """Higher-l partial waves return finite cross sections."""
    E = np.array([0.1, 1.0, 10.0])
    res_p = compute_single_channel(_V_attractive_well, E, a_au=15.0,
                                     n_basis=120, l=1)
    assert np.all(np.isfinite(res_p.cross_section_cm2))
    assert np.all(res_p.cross_section_cm2 >= 0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([1.0]), a_au=0.0)
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([1.0]), n_basis=2)
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([1.0]), l=-1)
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([-1.0]))


def test_eigenvalues_sorted():
    """The interior eigenvalues E_n come out sorted ascending."""
    res = compute_single_channel(_V_attractive_well,
                                   np.array([1.0]),
                                   a_au=15.0, n_basis=80, l=0)
    assert np.all(np.diff(res.eigenvalues_E_n_au) >= 0)


def test_multi_channel_compute_dispatches_to_close_coupling():
    """``compute`` aliases the phase-2 ``compute_multichannel`` driver.

    Phase-1 left this entry point as a ``NotImplementedError`` stub; phase
    2 wires it through to the multi-channel R-matrix solver (HYP-188).
    Calling with no arguments now raises ``TypeError`` for the missing
    required parameters, not ``NotImplementedError``.
    """
    from atomx.scattering.r_matrix import (
        compute_multichannel,
        heplus_target_states,
    )
    targets = list(heplus_target_states(Z=2.0).values())
    res = compute_multichannel(
        targets, L_total=0, S_total=0, parity=+1,
        energies_au=np.array([-0.8]), Z=2.0, a_au=20.0,
        n_basis=30, n_gauss=6, l_c_max=1,
    )
    assert res.eigenphase_sum.shape == (1,)
    assert np.isfinite(res.eigenphase_sum[0])


# -----------------------------------------------------------------------------
# Target-state threshold pins against live FAC reference values.
# -----------------------------------------------------------------------------
#
# An R-matrix close-coupling solver needs the target-state energies to
# place the channel thresholds. Pojo and the codex-review reviewer
# pre-ran ``sfac`` to lock in the H and He+ ground-state energies
# (HYP-172, 2026-05-18). These tests pin atomx's atomic-structure path
# (Sommerfeld/Dirac hydrogenic + finite-nuclear-mass scaling) against
# those FAC numbers to ≤ 5e-4 relative — the gate Pojo demanded before
# the R-matrix surface result is taken seriously.


def _reduced_mass_factor(M_over_me: float) -> float:
    """μ / m_e for an electron bound to a nucleus of mass ``M``."""
    return M_over_me / (M_over_me + 1.0)


def _atomx_hydrogenic_target_eV(Z: int, M_over_me: float) -> float:
    """atomx's hydrogenic target threshold with finite-nuclear-mass scaling."""
    E_inf = hydrogenic_dirac_energy(n=1, kappa=-1, Z=Z) * HARTREE_EV
    return E_inf * _reduced_mass_factor(M_over_me)


def test_hydrogen_target_threshold_matches_nist():
    """atomx H 1s threshold (reduced mass) must match NIST to 1e-5.

    Measured 2026-09-22: atomx 13.59846830 eV vs NIST 13.598434599702 eV,
    relerr 2.5e-6.
    """
    E_atomx = -_atomx_hydrogenic_target_eV(
        Z=1, M_over_me=PROTON_ELECTRON_MASS_RATIO)
    relerr = abs(E_atomx - NIST_H_IONIZATION_EV) / NIST_H_IONIZATION_EV
    assert relerr <= FAC_THRESHOLD_RELTOL, (
        f"H 1s: atomx={E_atomx:.8f} eV  NIST={NIST_H_IONIZATION_EV:.8f} eV  "
        f"relerr={relerr:.3e} (gate {FAC_THRESHOLD_RELTOL:.0e})"
    )


def test_heplus_target_threshold_matches_nist():
    """atomx He+ 1s threshold (reduced mass) must match NIST to 1e-5.

    Measured 2026-09-22: atomx 54.41821051 eV vs NIST 54.417760 eV,
    relerr 8.3e-6 (the residual is the hydrogenic-Dirac vs NIST QED
    difference at Z=2).
    """
    E_atomx = -_atomx_hydrogenic_target_eV(Z=2, M_over_me=M_ALPHA_OVER_E)
    relerr = abs(E_atomx - NIST_HEPLUS_IONIZATION_EV) / NIST_HEPLUS_IONIZATION_EV
    assert relerr <= FAC_THRESHOLD_RELTOL, (
        f"He+ 1s: atomx={E_atomx:.8f} eV  NIST={NIST_HEPLUS_IONIZATION_EV:.8f}"
        f" eV  relerr={relerr:.3e} (gate {FAC_THRESHOLD_RELTOL:.0e})"
    )


def test_heplus_threshold_gate_needs_the_reduced_mass():
    """The tightened gate is sensitive to the finite-nuclear-mass factor.

    atomx's infinite-mass hydrogenic He+ 1s energy is 54.42567089 eV, i.e.
    1.45e-4 from NIST — 14x the gate.  Without the reduced-mass factor this
    test would fail, so the threshold agreement above is not an artefact of a
    loose band.  (Both H, 5.1e-4 off, and He II fail at infinite mass; only H
    failed the old 5e-4 gate, which is exactly the hole the S5 audit found.)
    """
    E_inf = -hydrogenic_dirac_energy(n=1, kappa=-1, Z=2) * HARTREE_EV
    relerr = abs(E_inf - NIST_HEPLUS_IONIZATION_EV) / NIST_HEPLUS_IONIZATION_EV
    assert relerr > FAC_THRESHOLD_RELTOL, (
        f"the infinite-mass He+ threshold {E_inf:.8f} eV is {relerr:.2e} from "
        f"NIST — inside the {FAC_THRESHOLD_RELTOL:.0e} gate, so the gate no "
        "longer tests the reduced-mass correction"
    )
    assert relerr == pytest.approx(1.45e-4, rel=0.05)


def test_r_matrix_carries_target_thresholds_into_kinematics():
    """Sanity check: ``compute_single_channel`` operates above the H 1s
    threshold (i.e. the projectile energies passed to it are positive
    *kinetic* energies, not total energies that would dip below the
    target ground state). This guards the gate that an open-channel
    R-matrix call cannot accidentally describe a closed channel."""
    E_kin = np.array([0.1, 1.0, 10.0])
    res = compute_single_channel(_V_zero, E_kin, a_au=12.0, n_basis=60, l=0)
    # Total energy of the (e + H) system in the elastic 1s channel:
    total_E = res.energies_eV - NIST_H_IONIZATION_EV
    # All total energies must stay above the H 1s threshold
    # (= -NIST_H_IONIZATION_EV); equivalently the kinetic energy is positive.
    assert np.all(total_E > -NIST_H_IONIZATION_EV)
    assert np.all(res.cross_section_cm2 >= 0)


# -----------------------------------------------------------------------------
# Channel spin-filtering for general target spins (Milestone 1).
# -----------------------------------------------------------------------------


def _make_target_state(label: str, l: int, S_t: float) -> TargetState:
    """Lightweight target state for channel-enumeration spin tests."""
    from atomx.scattering.r_matrix.target_states import heplus_target_states
    template = heplus_target_states(Z=2.0)["2p"]
    return TargetState(
        label=label,
        n=2,
        l=l,
        Z=2.0,
        energy_au=-0.5,
        P=template.P,
        S_t=S_t,
    )


def test_enumerate_channels_one_electron_target_spin_filter():
    """For a one-electron target (S_t=1/2) only S_total=0 and 1 are allowed."""
    targets = [_make_target_state("2p", l=1, S_t=0.5)]
    # S_total=0 and 1 keep the (2p, εp) L=0 even-parity channel.
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=0)) == 1
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=1)) == 1
    # S_total=1/2 cannot be formed from S_t=1/2 + s_e=1/2.
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=0.5)) == 0


def test_enumerate_channels_two_electron_singlet_target():
    """A singlet two-electron target (S_t=0) only couples to S_total=1/2."""
    targets = [_make_target_state("1s2p_1P", l=1, S_t=0.0)]
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=0.5)) == 1
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=0)) == 0
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=1)) == 0


def test_enumerate_channels_two_electron_triplet_target():
    """A triplet two-electron target (S_t=1) couples to S_total=1/2 and 3/2."""
    targets = [_make_target_state("1s2p_3P", l=1, S_t=1.0)]
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=0.5)) == 1
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=1.5)) == 1
    assert len(enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1,
                                  S_total=1.0)) == 0


def test_enumerate_channels_no_spin_filter_retains_all_targets():
    """The default S_total=None retains all targets regardless of spin."""
    targets = [
        _make_target_state("2p", l=1, S_t=0.5),
        _make_target_state("1s2p_1P", l=1, S_t=0.0),
    ]
    channels = enumerate_channels(targets, L_total=0, parity=+1, l_c_max=1)
    assert len(channels) == 2
    assert {ch.target.label for ch in channels} == {"2p", "1s2p_1P"}


def test_target_residual_charge():
    """Residual Coulomb charge is Z - N_electrons."""
    heplus = heplus_target_states(Z=2.0)["1s"]
    assert heplus.n_electrons == 1
    assert abs(heplus.residual_charge - 1.0) < 1e-12

    helike = helike_target_states(Z=2.0)["1s^2 ^1S"]
    assert helike.n_electrons == 2
    assert abs(helike.residual_charge - 0.0) < 1e-12

    # A synthetic neutral one-electron target also has zero residual charge.
    h_like = _make_target_state("1s", l=0, S_t=0.5)
    assert h_like.Z == 2.0  # synthetic, but n_electrons defaults to 1
    assert abs(h_like.residual_charge - 1.0) < 1e-12
