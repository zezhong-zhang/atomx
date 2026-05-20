"""Hand-checkable analytic anchor for off-diagonal J-coupled CI (HYP-184/185).

The off-diagonal J-coupled CI correction in
:func:`atomx.methods.mcdhf.matrix._off_diagonal_J_correction` is the missing
physics piece that, in the Fe XVII 2p^5 3l J=1- block, redistributes the
oscillator strength among the three 3d J=1- CSFs to recover the proper
[1]P_1 / [3]D_1 / [3]P_1 intensity pattern that Bernitt 2012 measures.

The cleanest hand-checkable anchor is He-like 1s2p, where the two J=1 jj-coupled
CSFs

    |alpha> = |1s 2p_{1/2}; J=1>   (subshells (1, -1, 1) + (2,  1, 1), total_J2=2)
    |beta>  = |1s 2p_{3/2}; J=1>   (subshells (1, -1, 1) + (2, -2, 1), total_J2=2)

mix via an off-diagonal CI element which, in the LS-coupling limit, reproduces
the canonical [1]P_1 / [3]P_1 singlet/triplet splitting

    E([1]P_1) - E([3]P_1) = 2 G^1(1s, 2p)

through the 2x2 J=1 block.  The off-diagonal element of this 2x2 block in the
jj basis follows from the Grant Sec. 8.3 / Cowan Sec. 12.3 formula

    <(j_a j_c) J | g | (j_b j_c) J>_exchange
        = sum_K (-1)^{phase}
          * { 6j(j_a, j_c, J; j_b, j_c, K) }
          * <j_a||C^K||j_c> <j_c||C^K||j_b>
          * R^K(ac, cb)

with j_a = 1/2 (2p_{1/2}), j_b = 3/2 (2p_{3/2}), j_c = 1/2 (1s), J = 1.
The K-range is the intersection of the (j_a, j_c) and (j_c, j_b) exchange
triangles plus parity, which evaluates to K = 1 only (in agreement with the
configuration-averaged formula in :func:`_one_diff_element`).  The 6j weight
differs from the configuration-averaged q_c/(2j_c+1) prefactor, and the
difference is precisely what the He-like singlet/triplet splitting reveals.

Until :func:`_off_diagonal_J_correction` lands a non-zero body (HYP-189
follow-up), the analytic xfail assertion below is the documented expectation
that the implementation must reproduce.  The scaffold-locks-in-zero assertion
runs unconditionally so that any accidental drift in the scaffold's return
value is caught early.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.methods.dirac_fock import Subshell
from atomx.methods.mcdhf.config import CSF
from atomx.methods.mcdhf.matrix import _off_diagonal_J_correction
from atomx.angular.coefficients import exchange_k_range, exchange_lambda
from atomx.angular.wigner import wigner_6j


def _make_helike_csfs():
    """Build the two He-like J=1 1s2p CSFs in jj coupling.

    Both CSFs carry an explicit ``total_J2=2`` and parity ``-1`` (odd, l=1)
    plus a ``[J=2/2]`` label tag so that the J-resolved code paths recognise
    them.  Occupation is 1 in each open subshell -> unique parentage, which is
    the regime where the Grant Sec. 8.3 6j formula applies and the
    configuration-averaged form differs.
    """
    alpha = CSF(
        subshells=[
            Subshell(1, -1, 1.0),   # 1s   (j=1/2, l=0)
            Subshell(2,  1, 1.0),   # 2p_{1/2}  (j=1/2, l=1)
        ],
        total_J2=2,
        parity=-1,
        label="1s 2p1/2 [J=2/2]",
    )
    beta = CSF(
        subshells=[
            Subshell(1, -1, 1.0),
            Subshell(2, -2, 1.0),   # 2p_{3/2}  (j=3/2, l=1)
        ],
        total_J2=2,
        parity=-1,
        label="1s 2p3/2 [J=2/2]",
    )
    return alpha, beta


def test_off_diag_j_ci_scaffold_returns_zero_for_helike_1s2p_j1_pair():
    """Scaffold lock-in: the no-op scaffold must return exactly 0.0.

    This guards against accidental drift in
    :func:`_off_diagonal_J_correction` until the Grant Sec. 8.3 body is
    landed in HYP-189.  When the body lands, this assertion should be
    relaxed (the value will be non-zero) and the xfail below should flip
    to a strict numeric expectation.
    """
    alpha, beta = _make_helike_csfs()
    # n_diff between alpha and beta is 1 (2p_{1/2} <-> 2p_{3/2}).
    assert alpha.n_electron_diff(beta) == 1

    # The body of _off_diagonal_J_correction is a no-op scaffold today; it
    # must return exactly 0.0 regardless of orbital/bsp contents, because
    # the implementation has not yet been filled in.  orbitals_dict and
    # bsp_data are passed as empty dicts to confirm the early-return path.
    delta = _off_diagonal_J_correction(
        alpha, beta, orbitals_dict={}, bsp_data={}, ns=1, n_diff=1,
    )
    assert delta == 0.0


def test_off_diag_j_ci_helike_1s2p_j1_pair_block_filter_is_consistent():
    """Sanity: the (J, parity) block guard accepts the He-like pair.

    Both CSFs share total_J2=2 and parity=-1, so the block-filter early
    returns inside :func:`_off_diagonal_J_correction` must not trip.
    Conversely, a perturbed CSF with parity flipped is filtered out and
    returns 0.0.  This protects against a future refactor that
    accidentally widens the filter beyond the same-(J, parity) regime.
    """
    alpha, beta = _make_helike_csfs()

    # Same (J, parity) -> reaches the body (which is itself a no-op today).
    same_block = _off_diagonal_J_correction(
        alpha, beta, orbitals_dict={}, bsp_data={}, ns=1, n_diff=1,
    )
    assert same_block == 0.0

    # Different parity -> early return, also 0.0 but for a different reason.
    beta_wrong_parity = CSF(
        subshells=beta.subshells,
        total_J2=beta.total_J2,
        parity=+1,                  # flipped
        label=beta.label,
    )
    cross_block = _off_diagonal_J_correction(
        alpha, beta_wrong_parity, orbitals_dict={}, bsp_data={}, ns=1, n_diff=1,
    )
    assert cross_block == 0.0


def test_off_diag_j_ci_helike_1s2p_grant_section_8_3_kernel_inputs_resolve():
    """Reachability check on the Grant Sec. 8.3 angular kernel inputs.

    Before the body of :func:`_off_diagonal_J_correction` is landed, we
    verify here that the angular kernel inputs the formula needs (the
    exchange K-range, the exchange-lambda values, and the relevant 6j
    symbol) all resolve to finite, non-trivial numbers for the He-like
    1s 2p J=1 pair.  This guards against a future refactor that removes
    or renames an angular helper and silently leaves the off-diagonal
    body un-implementable.

    The check encodes the formula

        K-range = (j_a, j_c) intersect (j_c, j_b) exchange triangles
        v_K     = (-1)^{(j_a + j_b)/2 + j_c + J + K}
                  * { 6j(j_a, j_c, J; j_b, j_c, K) }
                  * sqrt(lambda^K(a,c) * lambda^K(c,b))

    with j_a = 1/2 (2p_{1/2}, kappa = +1), j_b = 3/2 (2p_{3/2}, kappa = -2),
    j_c = 1/2 (1s, kappa = -1), J = 1 (total_J2 = 2).
    """
    # kappa values follow the codebase convention (j = |kappa| - 1/2, l from sign).
    kappa_a = 1    # 2p_{1/2}
    kappa_b = -2   # 2p_{3/2}
    kappa_c = -1   # 1s
    total_J2 = 2

    k_range = sorted(set(exchange_k_range(kappa_a, kappa_c))
                     & set(exchange_k_range(kappa_c, kappa_b)))
    # The configuration-averaged formula in _one_diff_element uses this same
    # K-range; for He-like 1s 2p it evaluates to K = 1 only (l_a + l_c = 1
    # odd, l_c + l_b = 1 odd -> K odd; (1/2,1/2) and (1/2,3/2) triangles).
    assert k_range == [1], f"expected K-range = [1] for 1s 2p_(j) exchange, got {k_range}"

    K = 1
    lam_ac = exchange_lambda(kappa_a, kappa_c, K)
    lam_cb = exchange_lambda(kappa_c, kappa_b, K)
    assert lam_ac > 1e-9, "lambda^1(2p_{1/2}, 1s) must be non-zero"
    assert lam_cb > 1e-9, "lambda^1(1s, 2p_{3/2}) must be non-zero"

    # 6j(j_a, j_c, J; j_b, j_c, K) with twice-j convention
    sixj = wigner_6j(
        2 * abs(kappa_a) - 1,    # tj_a = 1
        2 * abs(kappa_c) - 1,    # tj_c = 1
        total_J2,                # 2 J  = 2
        2 * abs(kappa_b) - 1,    # tj_b = 3
        2 * abs(kappa_c) - 1,    # tj_c = 1
        2 * K,                   # 2 K  = 2
    )
    assert abs(sixj) > 1e-9, (
        "The 6j symbol that drives the off-diagonal J-CI correction must be "
        "non-zero for the He-like 1s 2p J=1 pair -- otherwise the formula "
        "would predict no singlet/triplet mixing in the LS limit."
    )


@pytest.mark.xfail(
    reason=(
        "HYP-189: off-diagonal J-coupled CI correction (Grant Sec. 8.3) is "
        "scaffolded but not yet implemented in "
        "atomx.methods.mcdhf.matrix._off_diagonal_J_correction.  When the "
        "Grant Sec. 8.3 / 6j body lands, the off-diagonal element of the "
        "He-like 1s 2p J=1 2x2 jj-block must be non-zero with the sign "
        "that reproduces the canonical E([1]P_1) - E([3]P_1) = 2 G^1(1s,2p) "
        "singlet/triplet splitting.  The minimal observable for this "
        "anchor is just |delta| > 0 with synthetic non-zero orbitals."
    ),
    strict=True,
)
def test_off_diag_j_ci_helike_1s2p_j1_pair_is_non_zero():
    """Future expectation: the implemented off-diag J-CI is non-zero here.

    Uses minimal synthetic orbitals (constant non-zero P, zero Q -- the
    Slater integral R^K(ac, cb) only needs the radial product P_a*P_c to
    be non-vanishing on the grid; the angular factor carries the rest).
    No B-spline machinery is required to confirm the angular kernel
    is wired up -- a flat radial product is enough for the magnitude
    check.

    This test will be flipped from xfail to a strict numeric check
    against an analytic R^1(1s, 2p) value as soon as the body of
    :func:`_off_diagonal_J_correction` lands the Grant Sec. 8.3 formula
    in HYP-189.
    """
    import numpy as np

    alpha, beta = _make_helike_csfs()
    ns = 4
    # Constant P channel, zero Q -- yields a finite R^K through the
    # numerical Slater-integral kernel without requiring a real B-spline
    # grid.  The bsp_data dict is the same minimal dict the kernel needs.
    P = np.ones(ns)
    Q = np.zeros(ns)
    orbitals_dict = {
        (1, -1): (P, Q),    # 1s
        (2,  1): (P, Q),    # 2p_{1/2}
        (2, -2): (P, Q),    # 2p_{3/2}
    }
    # Bare bsp_data dict.  The off-diagonal scaffold ignores its contents
    # today, but a real implementation will read the grid here.
    bsp_data: dict = {}

    delta = _off_diagonal_J_correction(
        alpha, beta, orbitals_dict, bsp_data, ns=ns, n_diff=1,
    )
    assert abs(delta) > 0.0, (
        "off-diagonal J-CI must mix the two J=1 He-like CSFs once "
        "Grant Sec. 8.3 is implemented (HYP-189)"
    )
