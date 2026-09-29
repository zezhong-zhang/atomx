"""First-principles gates on the Gaunt (Breit) operator's angular algebra.

`atomx.methods.mcdhf.breit_ci` claims in its module docstring that its
angular algebra is checked here.  This file is that check.

The Gaunt operator is the piece of the Breit interaction that carries all of
its non-scalar content, and it is the piece the historical MCDHF path threw
away by evaluating ``compute_breit_energy`` once on an averaged orbital set
and adding the same number to every CI state.  So it is exactly the kind of
code where a wrong angular coefficient is invisible: it shifts every level
identically, which looks like a constant offset rather than a bug.

Everything asserted below is *derivable*, and the derivations are stated:

1. **Dirac quantum-number bookkeeping.**  ``2j = 2|κ| − 1`` and
   ``κ = (l − j)(2j + 1)``, so ``l = κ`` for ``κ > 0`` and ``l = |κ| − 1``
   for ``κ < 0``.  A sign slip here silently swaps the two fine-structure
   partners in every radial integral.

2. **The spherical Pauli components.**  ``σ_0 = σ_z`` and
   ``σ_{±1} = ∓(σ_x ± iσ_y)/√2``.  The required identities are
   ``σ_μ† = (−1)^μ σ_{−μ}`` (which is what makes ``α_μ`` Hermitian) and
   ``Tr(σ_μ σ_ν†) = 2δ_{μν}`` (which fixes the normalisation, and hence the
   overall scale of every Gaunt matrix element).

3. **The rank-``L`` angular element.**  ``_clm`` must equal the closed-form
   Racah Gaunt coefficient

       ``⟨l m| C^L_q |l' m'⟩ = (−1)^{l−m} √((2l+1)(2l'+1))``
       ``    × 3j(l L l'; −m q m') · 3j(l L l'; 0 0 0)``

   with ``C^L_q = √(4π/(2L+1)) Y_L^q``, up to the module's overall ``(−1)^l``
   phase convention.  Note the ``(2L+1)`` from the Racah normalisation
   cancels against the ``√(2L+1)`` in the Gaunt coefficient — that is the
   step where a factor of ``2L+1`` usually gets lost or double-counted, and
   it is why the check is worth writing in closed form instead of tabulating.

4. **Completeness of the spherical harmonics.**  ``Σ_q |C^L_q(n̂)|² = 1``
   pointwise, hence ``Σ_q Σ_{l'm'} |⟨l m|C^L_q|l'm'⟩|² = 1`` for every
   ``(l, m, L)``.  This is a normalisation sum rule, not a fit: it fails for
   any wrong factor in ``_clm``, in either direction.

5. **Selection rules of the two-body operator.**  ``M = Σ 2m`` is conserved
   and the multipole is bounded by both triangle conditions,
   ``L ≤ min(j_a+j_c, j_b+j_d)+1``.

6. **Retardation is included.** The complete static operator contains
   Gaunt plus retardation; frequency-dependent transverse exchange remains
   outside this Hamiltonian. Independent radial/operator tests are in
   ``test_static_breit_retardation.py``.

Two properties were probed and are deliberately **not** asserted here,
because they are not true as coefficient-level statements: the term list of
``gaunt_angular_terms`` is decomposed over ``(L, q, μ)`` and the individual
``(L, X_a, X_c, X_b, X_d)`` terms are not separately parity-even, nor does the
term multiset of ``⟨cd|G|ab⟩`` equal that of ``⟨ab|G|cd⟩``.  Only the *summed*
matrix element has those properties, and it can only be formed once the radial
integrals are supplied.  Asserting them against the decomposition would be
asserting something false, so they are not asserted here at all.

Full static Breit closed-shell totals are checked against independent
DBSR_HF references in ``test_static_breit_retardation.py``.
"""

from __future__ import annotations

import itertools
import math

import pytest

from atomx.angular.wigner import wigner_3j
from atomx.methods.mcdhf.breit_ci import (
    RETARDATION_NOTE,
    BreitOperatorError,
    _clm,
    _sigma_mu,
    gaunt_angular_terms,
    l_of_kappa,
    retardation_terms,
    twoj_of_kappa,
)

#: The three Pauli matrices, in the same ``(m_s = +½, m_s = −½)`` ordering
#: that ``_sigma_mu`` uses.
_SX = ((0.0, 1.0), (1.0, 0.0))
_SY = ((0.0, -1.0j), (1.0j, 0.0))
_SZ = ((1.0, 0.0), (0.0, -1.0))


def _sigma_matrix(mu: int):
    """``_sigma_mu`` as a 2x2 matrix, indexed by ``(m_sa, m_sb)``."""
    return [
        [complex(_sigma_mu(mu, 2 * a + 1, 2 * b + 1)) for b in range(2)]
        for a in range(2)
    ]


def _spherical_pauli(mu: int):
    """The closed-form spherical components of the Pauli vector."""
    if mu == 0:
        return [[complex(_SZ[i][j]) for j in range(2)] for i in range(2)]
    sx = [[complex(_SX[i][j]) for j in range(2)] for i in range(2)]
    sy = [[complex(_SY[i][j]) for j in range(2)] for i in range(2)]
    sign = -1.0 if mu == 1 else 1.0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return [
        [sign * (sx[i][j] + 1.0j * mu * sy[i][j]) * inv_sqrt2 for j in range(2)]
        for i in range(2)
    ]


def _flat(matrix):
    """``pytest.approx`` cannot compare nested sequences; flatten a 2x2."""
    return [entry for row in matrix for entry in row]


def _racah_gaunt(l: int, m: int, L: int, q: int, lp: int, mp: int) -> float:
    """Closed-form ``⟨l m|C^L_q|l' m'⟩`` in the Racah normalisation.

    Written out from the standard Gaunt integral; ``wigner_3j`` is called with
    doubled arguments because ``atomx.angular.wigner`` takes ``2j``.
    """
    if m - q != mp:
        return 0.0
    if abs(m) > l or abs(q) > L or abs(mp) > lp:
        return 0.0
    w = wigner_3j(2 * l, 2 * L, 2 * lp, -2 * m, 2 * q, 2 * mp)
    if w == 0.0:
        return 0.0
    return (
        (-1.0) ** (l - m)
        * math.sqrt((2 * l + 1) * (2 * lp + 1))
        * w
        * wigner_3j(2 * l, 2 * L, 2 * lp, 0, 0, 0)
    )


# ---------------------------------------------------------------------------
# 1.  Dirac quantum-number bookkeeping
# ---------------------------------------------------------------------------


class TestKappaBookkeeping:
    """``2j`` and ``l`` recovered from ``κ`` must satisfy ``κ = (l−j)(2j+1)``."""

    @pytest.mark.parametrize("kappa", [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5])
    def test_kappa_from_l_and_j(self, kappa):
        """The defining relation, not a table of eight values.

        ``κ = (l − j)(2j + 1)`` with ``j = (2j−1)/2``.  For ``κ < 0`` this
        gives ``j = l + ½``, for ``κ > 0`` it gives ``j = l − ½``; a sign slip
        swaps the two fine-structure partners in every radial integral.
        """
        twoj = twoj_of_kappa(kappa)
        l = l_of_kappa(kappa)
        j = twoj / 2.0
        assert kappa == int(round((l - j) * (2 * j + 1))), (
            f"kappa={kappa} -> 2j={twoj}, l={l}: "
            f"(l-j)(2j+1) = {(l - j) * (2 * j + 1)}, expected {kappa}"
        )

    @pytest.mark.parametrize("kappa", [-1, 1, -2, 2, -3, 3, -4, 4])
    def test_l_and_twoj_are_non_negative_and_consistent(self, kappa):
        assert twoj_of_kappa(kappa) == 2 * abs(kappa) - 1
        assert l_of_kappa(kappa) == (abs(kappa) - 1 if kappa < 0 else kappa)
        assert l_of_kappa(kappa) >= 0
        # j = l ± 1/2 exactly, so 2j is odd and l is within 1/2 of j
        assert twoj_of_kappa(kappa) % 2 == 1
        assert abs(l_of_kappa(kappa) - twoj_of_kappa(kappa) / 2.0) == 0.5


# ---------------------------------------------------------------------------
# 2.  Spherical Pauli components
# ---------------------------------------------------------------------------


class TestSphericalPauli:
    """``σ_μ`` is fixed by ``σ_0 = σ_z``, ``σ_{±1} = ∓(σ_x ± iσ_y)/√2``."""

    @pytest.mark.parametrize("mu", [-1, 0, 1])
    def test_matches_the_explicit_pauli_matrices(self, mu):
        got = _sigma_matrix(mu)
        want = _spherical_pauli(mu)
        assert _flat(got) == pytest.approx(_flat(want), abs=1e-15), (
            f"sigma_{mu} = {got}, closed form = {want}"
        )

    @pytest.mark.parametrize("mu", [-1, 0, 1])
    def test_hermiticity_pairing(self, mu):
        """``σ_μ† = (−1)^μ σ_{−μ}`` — the identity that makes ``α_μ`` Hermitian.

        Without it the Gaunt operator is not Hermitian and the CI matrix is
        not symmetric, which shows up as complex eigenvalues rather than as a
        wrong number, so it is easy to miss.
        """
        got = _sigma_matrix(mu)
        partner = _sigma_matrix(-mu)
        conjugate_transpose = [
            [got[j][i].conjugate() for j in range(2)] for i in range(2)
        ]
        expected = [[(-1.0) ** mu * partner[i][j] for j in range(2)] for i in range(2)]
        assert _flat(conjugate_transpose) == pytest.approx(_flat(expected), abs=1e-15)

    def test_trace_orthonormality_fixes_the_normalisation(self):
        """``Tr(σ_μ σ_ν†) = 2 δ_{μν}``.

        This is the normalisation statement: the factor ``1/√2`` in the
        spherical components, and the overall scale of every Gaunt matrix
        element, are both pinned by it.  Getting it wrong scales the whole
        Breit correction while leaving every selection rule intact.
        """
        mus = [-1, 0, 1]
        for i, mu in enumerate(mus):
            for j, nu in enumerate(mus):
                a = _sigma_matrix(mu)
                b = _sigma_matrix(nu)
                # Tr(A B^dag) = sum_{k,m} A_{km} conj(B_{km})
                trace = sum(
                    a[k][m] * b[k][m].conjugate() for k in range(2) for m in range(2)
                )
                assert complex(trace).real == pytest.approx(
                    2.0 if i == j else 0.0, abs=1e-14
                ), f"Tr(sigma_{mu} sigma_{nu}^dag) = {trace}, expected {2.0 if i == j else 0.0}"
                assert abs(complex(trace).imag) < 1e-14


# ---------------------------------------------------------------------------
# 3-4.  The rank-L angular element
# ---------------------------------------------------------------------------


class TestRankLAngularElement:
    """``_clm`` is the closed-form Racah Gaunt coefficient times ``(−1)^l``."""

    def test_matches_the_closed_form_racah_gaunt_coefficient(self):
        """Exhaustive over ``l, l' ≤ 3``, ``L ≤ 4``, all ``m, q, m'``.

        ``_clm`` carries an overall ``(−1)^l`` phase relative to the standard
        coefficient; that phase is a convention (it cancels in the final
        matrix element) and is divided out here so that what remains is the
        physics: the ``(2l+1)(2l'+1)`` weight, the two ``3j`` symbols, and the
        ``m − q = m'`` conservation.
        """
        mismatches = []
        for l in range(4):
            for L in range(5):
                for lp in range(4):
                    for m in range(-l, l + 1):
                        for q in range(-L, L + 1):
                            for mp in range(-lp, lp + 1):
                                got = _clm(l, 2 * m, L, 2 * q, lp, 2 * mp)
                                want = ((-1.0) ** l) * _racah_gaunt(
                                    l, m, L, q, lp, mp
                                )
                                if abs(got - want) > 1e-12:
                                    mismatches.append((l, m, L, q, lp, mp, got, want))
        assert not mismatches, (
            f"{len(mismatches)} of the (l<=3, L<=4, l'<=3) angular elements "
            f"disagree with the closed-form Racah Gaunt coefficient; first: "
            f"{mismatches[0]}"
        )

    def test_allowed_multipoles_are_triangle_plus_parity(self):
        """``_clm`` vanishes unless ``|l − l'| ≤ L ≤ l + l'`` **and**
        ``l + L + l'`` is even.

        The parity half is not optional: it is the ``3j(l L l'; 0 0 0)``
        factor, which vanishes for ``l + L + l'`` odd.  ``l = l' = 1`` with
        ``L = 1`` satisfies the triangle and is still exactly zero — which is
        the case that catches an implementation that kept only the triangle
        and quietly dropped the ``3j`` with the three zero projections.
        """
        for l, lp in [(0, 1), (1, 1), (2, 0), (3, 2), (2, 2), (0, 2)]:
            for L in range(0, 6):
                allowed = abs(l - lp) <= L <= l + lp and (l + L + lp) % 2 == 0
                nonzero = any(
                    _clm(l, 2 * m, L, 2 * q, lp, 2 * mp) != 0.0
                    for m in range(-l, l + 1)
                    for q in range(-L, L + 1)
                    for mp in range(-lp, lp + 1)
                )
                assert nonzero == allowed, (
                    f"l={l}, l'={lp}, L={L}: triangle says "
                    f"{'allowed' if allowed else 'forbidden'} but the element is "
                    f"{'nonzero' if nonzero else 'zero'}"
                )

    @pytest.mark.parametrize("l,L", [(0, 1), (1, 0), (1, 1), (2, 2), (3, 4)])
    def test_completeness_sum_rule(self, l, L):
        """``Σ_q Σ_{l'm'} |⟨l m|C^L_q|l'm'⟩|² = 1`` for every ``m``.

        From ``Σ_q |C^L_q(n̂)|² = 1``, which holds pointwise on the sphere
        because ``C^L`` is ``√(4π/(2L+1))`` times a normalised spherical
        harmonic.  Summing the squared elements over a complete set of
        intermediate ``(l', m')`` therefore gives 1 exactly.  This is a
        normalisation sum rule: any wrong factor in ``_clm`` — including the
        ``√(2L+1)`` that the Racah normalisation cancels — breaks it, and no
        amount of phase convention can hide that.
        """
        for m in range(-l, l + 1):
            total = 0.0
            for lp in range(l + L + 1):
                for mp in range(-lp, lp + 1):
                    for q in range(-L, L + 1):
                        total += _clm(l, 2 * m, L, 2 * q, lp, 2 * mp) ** 2
            assert total == pytest.approx(1.0, abs=1e-12), (
                f"l={l}, m={m}, L={L}: sum of squared angular elements = "
                f"{total}, but completeness requires exactly 1"
            )


# ---------------------------------------------------------------------------
# 5.  Selection rules of the two-body coefficient list
# ---------------------------------------------------------------------------


class TestTwoBodySelectionRules:
    """``M`` conservation, the multipole ceiling, and reality of the terms."""

    #: A representative set of κ quadruples spanning s, p, d, f and both
    #: fine-structure partners on every leg.
    _KAPPAS = (-1, 1, -2, 2, -3, 3)

    def _all_terms(self):
        for ka, kb, kc, kd in itertools.product(self._KAPPAS, repeat=4):
            for m2a in range(-twoj_of_kappa(ka), twoj_of_kappa(ka) + 1, 2):
                for m2b in range(-twoj_of_kappa(kb), twoj_of_kappa(kb) + 1, 2):
                    for m2c in range(-twoj_of_kappa(kc), twoj_of_kappa(kc) + 1, 2):
                        for m2d in range(-twoj_of_kappa(kd), twoj_of_kappa(kd) + 1, 2):
                            terms = gaunt_angular_terms(
                                ka, m2a, kb, m2b, kc, m2c, kd, m2d
                            )
                            if terms:
                                yield (ka, m2a, kb, m2b, kc, m2c, kd, m2d), terms

    def test_M_conservation(self):
        """``Σ m`` of the bra must equal that of the ket.

        The Gaunt operator is a scalar under rotation, so ``M`` is a good
        quantum number; a nonzero element with ``ΔM ≠ 0`` means the ``Σ_{μq}``
        contraction dropped or doubled a term.
        """
        violations = []
        for args, terms in self._all_terms():
            ka, m2a, kb, m2b, kc, m2c, kd, m2d = args
            if m2a + m2b != m2c + m2d:
                violations.append((args, len(terms)))
        assert not violations, (
            f"{len(violations)} two-body elements have ΔM ≠ 0; first: "
            f"{violations[0]}"
        )

    def test_multipole_ceiling_is_the_triangle_bound(self):
        """``L ≤ min(j_a+j_c, j_b+j_d)+1`` on every term.

        The rank-1 alpha matrix multiplies ``C^L`` in the expansion ``1/r₁₂ = Σ_L r_<^L/r_>^{L+1} C^L·C^L``
        allows photon L one larger than the electronic pair triangle.  The expansion is also *complete*: the ceiling must be
        actually reached for at least one term, otherwise an allowed
        multipole is being truncated away.
        """
        violations = []
        global_max = 0
        global_ceiling = 0
        for args, terms in self._all_terms():
            ka, _m2a, kb, _m2b, kc, _m2c, kd, _m2d = args
            ceiling = min(
                (twoj_of_kappa(ka) + twoj_of_kappa(kc)) // 2,
                (twoj_of_kappa(kb) + twoj_of_kappa(kd)) // 2,
            )
            ceiling += 1  # alpha rank 1 permits photon L = j_a+j_c+1
            ls = {L for (L, *_rest) in terms}
            global_max = max(global_max, max(ls))
            global_ceiling = max(global_ceiling, ceiling)
            if max(ls) > ceiling:
                violations.append((args, max(ls), ceiling))
        assert not violations, (
            f"{len(violations)} elements exceed the triangle ceiling; first: "
            f"{violations[0]}"
        )
        # The bound need not be attained by every element -- for s shells the
        # parity factor kills every L > 0 -- but across the whole parameter set
        # the largest allowed photon rank (L = 6) must actually appear, or the
        # sum over L is being truncated below the selection-rule maximum.
        assert global_max == global_ceiling == 6, (
            f"the largest multipole that appears anywhere is L = {global_max} "
            f"while the largest triangle bound in this parameter set is "
            f"{global_ceiling}; the photon-multipole sum is being truncated"
        )

    def test_all_coefficients_are_real(self):
        """A real Hamiltonian has real matrix elements in a real spinor basis.

        A surviving imaginary part means the relative ``i`` between the ``PQ``
        and ``QP`` radial integrals leaked out of the radial part and into the
        angular coefficient, where it cannot cancel.
        """
        worst = 0.0
        for _args, terms in self._all_terms():
            for (_L, _xa, _xc, _xb, _xd, coeff) in terms:
                worst = max(worst, abs(complex(coeff).imag))
        assert worst < 1e-14, (
            f"the largest imaginary part in any Gaunt angular coefficient is "
            f"{worst:.3e}; the coefficients must be real"
        )


# ---------------------------------------------------------------------------
# 6.  Declared omissions stay declared
# ---------------------------------------------------------------------------


class TestStaticBreitContent:
    """Static retardation is present and dynamic exchange is declared absent."""

    def test_retardation_is_included_and_dynamic_exchange_is_declared(self):
        from atomx.methods.mcdhf.breit_ci import breit_terms, gaunt_terms
        labels = [(1,-1,1),(1,-1,-1),(2,1,1),(2,1,-1)]
        terms = retardation_terms(0,1,2,3,1.,labels)
        assert terms and all(key[0] == 't' for key,_ in terms)
        assert breit_terms(0,1,2,3,1.,labels) == gaunt_terms(0,1,2,3,1.,labels)+terms
        assert 'Gaunt plus frequency-independent retardation' in RETARDATION_NOTE
        assert 'Frequency-dependent transverse exchange is not included' in RETARDATION_NOTE

    def test_gaunt_terms_are_available_for_the_same_arguments(self):
        """The implemented path must actually produce terms.

        Guards the degenerate "everything raises" failure, in which the
        operator is nominally required but silently contributes nothing.
        """
        terms = gaunt_angular_terms(-1, 1, -1, -1, -1, 1, -1, -1)
        assert terms, "the Gaunt operator returned no terms for 1s²/1s²"
        assert all(len(term) == 6 for term in terms)
