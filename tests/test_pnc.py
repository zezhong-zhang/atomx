"""Tests for parity-non-conservation helpers.

The PNC *amplitude* is a scaffold (``pnc_amplitude_dirac_only`` raises
``NotImplementedError``; the two tests for that are API-contract tests,
not physics gates).  The two quantities that do return numbers are gated
here against closed forms:

* ``weak_charge_qw(Z, N)`` is the Standard-Model closed form
  ``Z(1 − 4sin²θ_W) − N`` evaluated with the shipped ``Q_W_PROTON`` /
  ``Q_W_NEUTRON``; for ¹³³Cs (Z=55, N=78) the Standard-Model prediction is
  ``Q_W = −73.22`` (the value the atomic-PNC experiments quote at tree
  level with the radiative corrections folded into sin²θ_W).
* ``weak_hamiltonian_radial_me`` is, by its own definition,
  ``(G_F/(2√2))·Q_W·∫ρ_nuc(r)·[P_aQ_b − Q_aP_b] dr``, so a *normalised*
  density (``∫ρ dr = 1``, as the docstring requires) acting on constant
  spinor components has the exact value
  ``(G_F/(2√2))·Q_W·c²`` — plus the exact antisymmetry
  ``ME(a,b) = −ME(b,a)`` and linearity in ``Q_W``.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.properties.pnc import (
    G_F_AU,
    Q_W_NEUTRON,
    Q_W_PROTON,
    pnc_amplitude_dirac_only,
    weak_charge_qw,
    weak_hamiltonian_radial_me,
)

#: PDG weak mixing angle (MS-bar), used to check the shipped SM inputs.
SIN2_THETA_W_PDG = 0.23122
#: Standard-Model weak charge of ¹³³Cs (Z=55, N=78), as quoted by the
#: atomic-PNC experiments.
QW_CS_STANDARD_MODEL = -73.22


def test_weak_charge_cs_matches_standard_model():
    """Q_W(¹³³Cs) = Z·Q_W(p) + N·Q_W(n) = −73.2205 (SM −73.22).

    The old gate was ``< -70.0``, which accepted −1000 (and any other
    "neutron-dominated" number).  The shipped constants are first checked
    against the Standard Model: Q_W(n) = −1 and Q_W(p) = 1 − 4sin²θ_W at
    tree level (PDG sin²θ_W = 0.23122), the shipped values carrying the
    known radiative corrections.
    """
    assert pytest.approx(-1.0, abs=0.012) == Q_W_NEUTRON
    assert pytest.approx(1.0 - 4.0 * SIN2_THETA_W_PDG, abs=4e-3) == Q_W_PROTON

    QW = weak_charge_qw(55, 78)
    assert pytest.approx(55 * Q_W_PROTON + 78 * Q_W_NEUTRON, rel=1e-15) == QW, (
        f"Q_W(55, 78) = {QW!r} is not the closed form "
        f"Z·Q_W(p) + N·Q_W(n)")
    assert pytest.approx(-73.2205, abs=1e-3) == QW, (
        f"Q_W(¹³³Cs) = {QW!r} vs the Standard-Model {QW_CS_STANDARD_MODEL}")
    # The closed form's Z- and N-dependence, checked independently.
    assert weak_charge_qw(1, 0) == pytest.approx(Q_W_PROTON, rel=1e-14)
    assert weak_charge_qw(0, 1) == pytest.approx(Q_W_NEUTRON, rel=1e-14)


def test_pnc_amplitude_scaffold_raises_on_hydrogen():
    """Low-level PNC amplitude stays loud until the SOS implementation lands."""
    atom = Atom(Z=1, config="1s1", corrections="none")
    atom.solve()
    with pytest.raises(NotImplementedError, match="opposite-parity"):
        pnc_amplitude_dirac_only(atom._result, "1s1/2", "1s1/2", N=0)


def test_compute_pnc_high_level_api_scaffold_raises_not_typeerror():
    """High-level compute_pnc preserves the scaffold raise contract."""
    from atomx.methods.pnc import compute_pnc

    atom = Atom(Z=3, config="1s2 2s1", corrections="none")
    atom.solve()
    with pytest.raises(NotImplementedError, match="opposite-parity"):
        compute_pnc(atom, "2s1/2", "1s1/2", N=4)


# ---------------------------------------------------------------------------
# weak_hamiltonian_radial_me — closed-form kernel checks
# ---------------------------------------------------------------------------

def _identity_workspace(n: int):
    """A workspace whose ``bsp_p = bsp_q = I`` makes eval_P/Q the identity.

    Same construction as the original smoke test: with ``nv = n`` and the
    identity basis table, the "coefficient vector" *is* the tabulated
    radial function on the grid.  Keep ``n`` small — the table is dense.
    """
    return {
        "gr": np.arange(n, dtype=float),
        "grw": np.ones(n),
        "nv": n,
        "ks": 1,
        "ns": n,
        "bsp_p": np.eye(n)[:, :, None],
        "bsp_q": np.eye(n)[:, :, None],
        "ksp": 1,
        "ksq": 1,
        "nsp": n,
        "nsq": n,
    }


def test_weak_hamiltonian_radial_me_normalised_density_closed_form():
    """Normalised ρ_nuc acting on constant spinors: ME = G_F/(2√2)·Q_W·c².

    A-class: the kernel is defined as
    ``(G_F/(2√2))·Q_W·∫ρ·[P_aQ_b − Q_aP_b] dr``; with ``P_a = Q_b = c``,
    ``Q_a = P_b = 0`` the bracket is the constant ``c²``, and the
    docstring requires ``∫ρ dr = 1`` — so the integral is ``c²`` exactly.

    The density here is the discrete delta ``ρ(r_i) = δ_{i0}`` on the
    workspace grid (``∫ρ dr = Σ grw·ρ = 1`` by construction).  This is the
    magnitude gate the old ``np.isfinite(me)`` check could not provide — a
    wrong prefactor, a sign error, or a ``P_aQ_b + Q_aP_b`` integrand all
    fail it — and quadratic scaling in the spinor amplitude is checked
    separately.
    """
    n = 5
    bsp = _identity_workspace(n)
    P_a = np.full(n, 2.0)
    Q_a = np.zeros(n)
    P_b = np.zeros(n)
    Q_b = np.full(n, 2.0)
    rho_delta = lambda r: 1.0 if r == 0.0 else 0.0  # noqa: E731

    me = weak_hamiltonian_radial_me(
        P_a, Q_a, P_b, Q_b, bsp, rho_nuc_func=rho_delta, Q_W=-1.0)
    c = 2.0
    expected = G_F_AU / (2.0 * np.sqrt(2.0)) * (-1.0) * c * c
    assert pytest.approx(expected, rel=1e-12) == me, (
        f"ME = {me!r} vs closed form {expected!r}")
    # Quadratic in the spinor amplitude (a missing square would give 2x).
    me_half = weak_hamiltonian_radial_me(
        P_a / 2.0, Q_a, P_b, Q_b / 2.0, bsp,
        rho_nuc_func=rho_delta, Q_W=-1.0)
    assert pytest.approx(me / 4.0, rel=1e-12) == me_half


def test_weak_hamiltonian_radial_me_is_antisymmetric_and_linear():
    """Exact identities of the kernel (A-class).

    The integrand ``ρ·(P_aQ_b − Q_aP_b)`` is antisymmetric under
    ``(a,b) → (b,a)``, and the matrix element is linear in ``Q_W`` with
    ``ME(Q_W=0) = 0``.  (The old smoke test's fixture — the same P/Q
    arrays the audit inherited — gives ``ME = -0.0`` identically, so its
    ``isfinite`` check could not even see a zero.)
    """
    n = 3
    bsp = _identity_workspace(n)
    P_a = np.array([1.0, 0.0, 0.0])
    Q_a = np.zeros(n)
    P_b = np.zeros(n)
    Q_b = np.array([1.0, 0.0, 0.0])
    rho = lambda _r: 1.0  # noqa: E731

    me_ab = weak_hamiltonian_radial_me(
        P_a, Q_a, P_b, Q_b, bsp, rho_nuc_func=rho, Q_W=-1.0)
    me_ba = weak_hamiltonian_radial_me(
        P_b, Q_b, P_a, Q_a, bsp, rho_nuc_func=rho, Q_W=-1.0)
    assert me_ab != 0.0, "the test fixture must give a non-zero ME"
    assert pytest.approx(-me_ab, rel=1e-12) == me_ba
    assert weak_hamiltonian_radial_me(
        P_a, Q_a, P_b, Q_b, bsp, rho_nuc_func=rho, Q_W=0.0) == 0.0
    assert pytest.approx(2.0 * me_ab, rel=1e-12) == weak_hamiltonian_radial_me(
        P_a, Q_a, P_b, Q_b, bsp, rho_nuc_func=rho, Q_W=-2.0)
