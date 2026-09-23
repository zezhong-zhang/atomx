"""Tests for dipole polarisability and the black-body Stark shift.

Reference values used here are external:

* α(He 1s², exact) = 1.383192 a₀³ — Mitroy, Safronova & Clark,
  *J. Phys. B* **43**, 202001 (2010), Table 1 (the same review the BBR
  formula in ``atomx.properties.polarizability`` is taken from);
* α(H 1s) = 9/2 a₀³ = 4.5 a₀³ — exact hydrogenic closed form
  (``α = (9/2) a₀³`` for the 1s state, e.g. Bethe & Salpeter §60);
* ⟨E²⟩ = (8.319 V/cm)² (T/300 K)⁴ — Mitroy-Safronova-Clark Eq. 4.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from atomx import Atom
from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
from atomx.properties.polarizability import (
    PolarizabilityResult,
    dipole_polarizability,
)

#: 1 a.u. of electric field, V/m (E_h / (e a_0)) — derived, not retyped.
AU_FIELD_VPM = HARTREE_EV / BOHR_RADIUS_M
#: RMS black-body field at 300 K, V/cm (Mitroy-Safronova-Clark Eq. 4).
BBR_FIELD_300K_VCM = 8.319
#: Exact helium static dipole polarisability, a₀³ (Mitroy et al. 2010).
HE_ALPHA_EXACT = 1.383192
#: Measured atomx plateau for He at a *converged* B-spline basis:
#: 1.117810 a₀³ at kappa_max=6, n_virt_max=80 (and 1.117810 again at
#: kappa_max=7, n_virt_max=100), i.e. 19.2 % below the exact value.
HE_ALPHA_MEASURED = 1.117810


@functools.lru_cache(maxsize=1)
def _he_result():
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    return atom._result


@functools.lru_cache(maxsize=1)
def _h_result():
    atom = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
    atom.solve()
    return atom._result


def test_helium_polarizability_is_converged_in_the_basis():
    """α(He) is a *converged* number at κmax ≥ 6, n_virt ≥ 80.

    A-class: the sequence must be monotone (length-gauge SOS with V^{N-1}
    virtuals is variational in the partial-wave basis) and must stop moving
    once the basis is saturated — the plateau, not the exact value, is what
    a finite basis can promise.  Measured: 0.892802 (2,15) →
    1.093864 (3,30) → 1.112394 (4,40) → 1.117768 (5,60) → 1.117810 (6,80)
    → 1.117810 (7,100).
    """
    res = _he_result()
    a_2_15 = dipole_polarizability(res, kappa_max=2, n_virt_max=15).alpha_static
    a_3_30 = dipole_polarizability(res, kappa_max=3, n_virt_max=30).alpha_static
    a_6_80 = dipole_polarizability(res, kappa_max=6, n_virt_max=80).alpha_static
    a_7_100 = dipole_polarizability(res, kappa_max=7, n_virt_max=100).alpha_static

    assert a_2_15 < a_3_30 < a_6_80, (
        f"α(He) must grow monotonically with the partial-wave basis: "
        f"{a_2_15:.6f} / {a_3_30:.6f} / {a_6_80:.6f}")
    # Saturated: doubling the virtual space past (6,80) changes nothing.
    assert pytest.approx(a_6_80, rel=1e-5) == a_7_100, (
        f"α(He) not converged: (6,80)={a_6_80:.6f} vs (7,100)={a_7_100:.6f}")


@pytest.mark.xfail(strict=True, reason=(
    "He α(0) at a converged B-spline basis is 1.117810 a₀³ "
    "(kappa_max=6, n_virt_max=80; unchanged at kappa_max=7, n_virt_max=100) "
    "against the exact 1.383192 a₀³ (Mitroy-Safronova-Clark 2010) — 19.2 % "
    "low, and the deficit is NOT basis truncation (the sequence has "
    "plateaued: 0.892802 at (2,15) → 1.093864 at (3,30) → 1.112394 at "
    "(4,40) → 1.117768 at (5,60) → 1.117810 at (6,80) → 1.117810 at "
    "(7,100)).  A frozen-core DHF SOS with V^{N-1} virtuals cannot reach "
    "the exact value (the Hartree-Fock limit is ≈1.32 a₀³ and the remaining "
    "0.06 a₀³ is correlation), so the missing physics is core relaxation / "
    "correlation in the virtual space, not the basis.  The previous gate was "
    "0.5 < α < 1.6, which passed at the manifestly unconverged 0.892802."))
def test_helium_polarizability_matches_exact_value():
    res = _he_result()
    alpha = dipole_polarizability(res, kappa_max=6, n_virt_max=80).alpha_static
    assert pytest.approx(HE_ALPHA_EXACT, rel=0.02) == alpha, (
        f"α(He) = {alpha:.6f} a₀³ (measured plateau {HE_ALPHA_MEASURED})")


def test_polarizability_grows_with_basis():
    res = _he_result()

    a1 = dipole_polarizability(res, kappa_max=1, n_virt_max=10).alpha_static
    a2 = dipole_polarizability(res, kappa_max=3, n_virt_max=30).alpha_static
    # Larger partial-wave basis → α(0) closer to the limit (always
    # increases for length gauge with V^{N-1} virtuals).
    assert a2 >= a1


def test_dynamic_polarizability_below_resonance():
    """α(ω) is positive and increasing for ω below the lowest
    transition (1s→2p for He, ~0.78 hartree at HF level).
    """
    res = _he_result()

    omega = np.array([0.0, 0.05, 0.1])
    r = dipole_polarizability(res, omega=omega, kappa_max=2, n_virt_max=15)
    assert r.alpha_dynamic is not None
    a = r.alpha_dynamic
    assert np.all(a > 0)
    assert a[0] < a[1] < a[2]


def test_bbr_shift_public_path_hydrogen_1s():
    """The public ``include_bbr=True`` path against the exact H 1s value.

    Δ_BBR = −½⟨E²⟩·α(0) with ⟨E²⟩ = (8.319 V/cm)² at 300 K (Mitroy-
    Safronova-Clark Eq. 4) and α(H 1s) = 4.5 a₀³ exactly.  Measured at
    κmax=4, n_virt=40: α_static = 4.499413 a₀³ (1.3e-4 below exact) and
    black_body_shift_K = −5.888017e-18 Ha vs the exact −5.888785e-18 Ha.
    This is the first test in the repository to touch ``include_bbr`` /
    ``black_body_shift_K`` at all (the old BBR test called the private
    ``_bbr_shift_from_alpha`` with a constant α and re-derived the
    implementation's own branch).
    """
    r = dipole_polarizability(
        _h_result(), kappa_max=4, n_virt_max=40, include_bbr=True)

    assert isinstance(r, PolarizabilityResult)
    assert r.black_body_shift_K is not None
    # α(H 1s) = 9/2 a₀³ exactly.
    assert pytest.approx(4.5, rel=1e-3) == r.alpha_static

    E_field_au = (BBR_FIELD_300K_VCM / 100.0) / AU_FIELD_VPM
    expected = -0.5 * E_field_au * E_field_au * 4.5
    assert pytest.approx(expected, rel=1e-3) == r.black_body_shift_K, (
        f"BBR shift {r.black_body_shift_K!r} Ha vs -<E^2>alpha/2 "
        f"{expected!r} Ha")


def test_bbr_dynamic_path_matches_static_limit_below_resonance():
    """At 300 K the Planck-weighted BBR shift equals the static limit.

    kT = 0.0259 eV is 400x below the H 1s→2p resonance (10.2 eV), so
    integrating α(ω) over the Planck spectrum must reproduce −½⟨E²⟩α(0).
    Measured ratio dynamic/static = 1.000107 at κmax=4, n_virt=40.
    """
    res = _h_result()
    static = dipole_polarizability(
        res, kappa_max=4, n_virt_max=40, include_bbr=True)
    omega = np.linspace(1e-4, 1.0, 400)
    dynamic = dipole_polarizability(
        res, omega=omega, kappa_max=4, n_virt_max=40, include_bbr=True)

    assert dynamic.black_body_shift_K is not None
    ratio = dynamic.black_body_shift_K / static.black_body_shift_K
    assert pytest.approx(1.0, rel=1e-3) == ratio, (
        f"dynamic/static BBR ratio {ratio:.6f} at 300 K")
    # Both are negative (a Stark shift always lowers the ground state).
    assert static.black_body_shift_K < 0.0
    assert dynamic.black_body_shift_K < 0.0
