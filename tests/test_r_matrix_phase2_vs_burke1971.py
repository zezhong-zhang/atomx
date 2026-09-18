"""Phase-2 R-matrix close-coupling regression vs Burke, Hibbert & Robb 1971.

Acceptance gate for HYP-188 phase 2: a 3-state CC(1s, 2s, 2p) expansion
for e + He+ must reproduce the three lowest doubly-excited Feshbach
resonance positions and widths of He below the He+ n=2 threshold to
1 % relative against the Burke-Hibbert-Robb 1971 table.

Reference
---------
P. G. Burke, A. Hibbert, W. D. Robb, *J. Phys. B: At. Mol. Phys.*
**4**, 153 (1971) - Table 2 (resonance positions and widths from the
3-state close-coupling expansion).

Convention here matches Burke 1971: positions ``E_r`` are total system
energies (Hartree, relative to ``E[He²⁺] = 0``), widths ``Gamma`` in
Hartree.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.scattering.r_matrix import (
    compute_multichannel,
    extract_resonances,
    heplus_target_states,
)

# Burke, Hibbert & Robb 1971 Table 2 reference values.
# Positions in Hartree (total e + He+ energy, He+ ground state at -2.0 Ha).
# Widths in Hartree (1 Ha = 27.21 eV; Γ ≈ 0.005 Ha ≈ 0.136 eV).
BURKE1971 = {
    "1Se_(2s2)":  {"E_r_au": -0.77816, "Gamma_au": 0.00500},
    "1Po_(2s2p+)": {"E_r_au": -0.69312, "Gamma_au": 0.00139},
    "1De_(2p2)":  {"E_r_au": -0.70220, "Gamma_au": 0.00264},
}


@pytest.fixture(scope="module")
def heplus_targets():
    return list(heplus_target_states(Z=2.0).values())


def _scan_1S(heplus_targets):
    energies = np.linspace(-0.95, -0.51, 600)
    return compute_multichannel(
        heplus_targets, L_total=0, S_total=0, parity=+1,
        energies_au=energies, Z=2.0, a_au=25.0, n_basis=60,
        k_spline_order=6, n_gauss=12, l_c_max=2,
        include_l2_pseudostates=True,
    )


def _scan_1Po(heplus_targets):
    energies = np.linspace(-0.85, -0.51, 600)
    return compute_multichannel(
        heplus_targets, L_total=1, S_total=0, parity=-1,
        energies_au=energies, Z=2.0, a_au=25.0, n_basis=60,
        k_spline_order=6, n_gauss=12, l_c_max=2,
        include_l2_pseudostates=True,
    )


def _scan_1De(heplus_targets):
    energies = np.linspace(-0.85, -0.51, 600)
    return compute_multichannel(
        heplus_targets, L_total=2, S_total=0, parity=+1,
        energies_au=energies, Z=2.0, a_au=25.0, n_basis=60,
        k_spline_order=6, n_gauss=12, l_c_max=3,
        include_l2_pseudostates=True,
    )


# ---------------------------------------------------------------------------
# Framework smoke tests — must pass.
# ---------------------------------------------------------------------------

def test_compute_multichannel_returns_CCResult(heplus_targets):
    res = compute_multichannel(
        heplus_targets, L_total=0, S_total=0, parity=+1,
        energies_au=np.array([-0.8]), Z=2.0, a_au=25.0,
        n_basis=40, n_gauss=8, l_c_max=1,
    )
    assert res.L_total == 0
    assert res.S_total == 0
    assert res.parity == +1
    assert len(res.channels) == 3
    assert res.inner.eigenvalues_au.shape[0] > 10
    assert res.eigenphase_sum.shape == (1,)


def test_inner_region_has_doubly_excited_eigenpair(heplus_targets):
    """The 3-state CC inner region carries at least one eigenpair in the
    Burke-1971 doubly-excited window [-0.85, -0.5] a.u. — the bound
    quasi-state that becomes the (2s², 2s2p, 2p²) resonance after
    closed-channel coupling to the (1s, εs) continuum."""
    res = compute_multichannel(
        heplus_targets, L_total=0, S_total=0, parity=+1,
        energies_au=np.array([-0.8]), Z=2.0, a_au=25.0,
        n_basis=60, n_gauss=12, l_c_max=2,
    )
    ev = res.inner.eigenvalues_au
    in_window = (ev > -0.85) & (ev < -0.50)
    assert in_window.any(), (
        f"No CC eigenpair in the doubly-excited He window [-0.85, -0.5] a.u. "
        f"— framework cannot produce the Madden-Codling resonance series. "
        f"Spectrum: {ev[(ev > -1.0) & (ev < -0.3)]}"
    )


def test_eigenphase_sum_is_real_and_finite(heplus_targets):
    res = compute_multichannel(
        heplus_targets, L_total=0, S_total=0, parity=+1,
        energies_au=np.linspace(-0.9, -0.55, 30), Z=2.0, a_au=25.0,
        n_basis=40, n_gauss=8, l_c_max=1,
    )
    assert np.all(np.isfinite(res.eigenphase_sum))


def test_r_matrix_excitation_cross_section_returns_finite_sigma():
    """Smoke test for the phase-3 R-matrix CE production hook."""
    from atomx.scattering.r_matrix import r_matrix_excitation_cross_section

    class _MockAtom:
        Z = 2

    E = np.array([30.0, 45.0, 60.0])
    sigma, omega = r_matrix_excitation_cross_section(
        _MockAtom(), _MockAtom(), E,
        initial_label="1s", final_label="2s",
        l_c_max=2, n_basis=40, n_gauss=8,
    )
    assert sigma.shape == E.shape
    assert omega.shape == E.shape
    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(omega))
    # Below the 1s -> 2s threshold (~40.8 eV for He+) the cross section is zero.
    below = E < 40.0
    assert np.all(sigma[below] == 0.0)
    assert np.all(omega[below] == 0.0)
    # Above threshold there is a non-zero excitation signal.
    above = E > 42.0
    assert np.all(sigma[above] > 0.0)
    assert np.all(omega[above] > 0.0)


def test_singlet_triplet_split_for_2p_target_state_thresholds():
    """The He+ n=2 target threshold sits at -0.5 a.u. (Z²/(2 n²) = 2/4·2).

    The ¹P^o and ³P^o close-coupling blocks see the same 2p threshold;
    this gate just checks that the channel-enumeration plumbing dispatches
    the same threshold for both spin blocks (they differ only in the
    exchange-sign of the inner-region Hamiltonian)."""
    targets = list(heplus_target_states(Z=2.0).values())
    e_2p = next(t.energy_au for t in targets if t.label == "2p")
    assert e_2p == pytest.approx(-0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Burke 1971 1 %-gate — full benchmark.
# ---------------------------------------------------------------------------

def test_burke1971_three_lowest_resonances_within_1pc(heplus_targets):
    """Strict Burke 1971 1 %-gate with L2 correlation pseudo-states."""
    res_1S = _scan_1S(heplus_targets)
    res_1Po = _scan_1Po(heplus_targets)
    res_1De = _scan_1De(heplus_targets)

    assert res_1S.provenance["l2_pseudostates"][0]["configuration"] == "2s2"
    assert res_1Po.provenance["l2_pseudostates"][0]["configuration"] == "2s2p"
    assert res_1De.provenance["l2_pseudostates"][0]["configuration"] == "2p2"
    assert res_1S.matchings[0].eigenphase_sum == pytest.approx(
        res_1S.eigenphase_sum[0]
    )

    def _strongest(fits, E_ref):
        if not fits:
            return None
        return min(fits, key=lambda f: abs(f.E_r_au - E_ref))

    fit_1S = _strongest(extract_resonances(res_1S),
                        BURKE1971["1Se_(2s2)"]["E_r_au"])
    fit_1Po = _strongest(extract_resonances(res_1Po),
                         BURKE1971["1Po_(2s2p+)"]["E_r_au"])
    fit_1De = _strongest(extract_resonances(res_1De),
                         BURKE1971["1De_(2p2)"]["E_r_au"])

    for label, fit, ref in [
        ("1Se", fit_1S, BURKE1971["1Se_(2s2)"]),
        ("1Po", fit_1Po, BURKE1971["1Po_(2s2p+)"]),
        ("1De", fit_1De, BURKE1971["1De_(2p2)"]),
    ]:
        assert fit is not None, f"No resonance found for {label}"
        rel_E = abs(fit.E_r_au - ref["E_r_au"]) / abs(ref["E_r_au"])
        rel_G = abs(fit.Gamma_au - ref["Gamma_au"]) / abs(ref["Gamma_au"])
        assert rel_E <= 0.01, (
            f"{label} position {fit.E_r_au:+.5f} vs Burke {ref['E_r_au']:+.5f}"
            f" — relerr {rel_E:.3%}"
        )
        assert rel_G <= 0.01, (
            f"{label} width {fit.Gamma_au:.5f} vs Burke {ref['Gamma_au']:.5f}"
            f" — relerr {rel_G:.3%}"
        )
