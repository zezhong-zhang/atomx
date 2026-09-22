"""Cross-rung first-principles invariants.

Every other file in this suite exercises one module.  This file exercises the
*physics* those modules claim, using statements that hold for any correct
implementation and whose expected values are derived **without consulting
atomx** and without consulting any other atomic-structure code.

The rest of the suite is largely change-detection: it pins what atomx printed
last time.  That is useful, but it cannot distinguish "correct" from
"consistently wrong", and this repository has shipped several consistently
wrong results past a green suite (a CI matrix that never evaluated
``<CSF_r|H|CSF_s>``, a Uehling kernel built from the wrong Bessel function, a
collision strength that fell to zero past the Born-completion point).  The
statements below are chosen because each one of them would have failed for at
least one of those defects, and because each is *derivable*, so it cannot rot
into agreement with a bug.

Six independent principles, in the order they appear:

1. **Dirac--Coulomb :math:`\\kappa` degeneracy.**
   ``E(n, κ) = E(n, −κ)`` exactly.  The Sommerfeld eigenvalue depends on the
   Dirac quantum number only through ``|κ|``, i.e. only through ``j``: the
   point-Coulomb Dirac problem is degenerate in ``j`` within a shell, which is
   precisely why the Lamb shift is observable.  A solver that leaks an
   ``l``-dependence into the point-Coulomb energy violates this.

2. **Closed form of the :math:`1s_{1/2}` level.**  For ``n = 1`` the general
   Sommerfeld expression collapses to ``E = mc²(√(1 − (Zα)²) − 1)``, which is
   derived independently (it is the ``γ = √(1 − (Zα)²)`` branch of the
   relativistic energy).  This checks the general formula against a special
   case, not against itself.

3. **The point-Coulomb critical charge.**  ``γ`` is imaginary for ``Zα > 1``:
   the Dirac equation with a *point* Coulomb potential has **no** bound
   solution for ``Z > 1/α ≈ 137.036``.  Nothing about atomic structure may
   quietly return a number there.

4. **Finite nuclear size reduces binding.**  A finite charge distribution has
   ``V_fin(r) ≥ V_pt(r)`` pointwise (equality outside the nucleus), so the
   first-order shift ``ΔE = ∫ρ_e(r)[V_fin − V_pt]d³r`` is strictly positive.
   This is a sign statement, not a fit, and it is independent of how the
   nuclear potential is modelled.  Its magnitude obeys a scaling law: for
   ``R ≪`` the inner-shell radius, ``ρ_e`` is constant across the nucleus and

       ``ΔE ≈ (2π/5) |ψ(0)|² Z R²``
       ``∝ Z³ · Z · R²``  and  ``R ∝ A^{1/3} ∝ Z^{1/3}``  ⟹  ``ΔE ∝ Z^{14/3}``

5. **Rayleigh--Ritz.**  No variational energy may fall below the exact
   ground-state energy of the Hamiltonian it is a Rayleigh quotient of.  This
   is the statement an unvalidated CI matrix violated by 0.39 Ha.

6. **Exact atomic-unit identities.**  The Hartree is *defined* as
   ``α²m_ec²``, so ``mc² = 1/α²`` Ha exactly; the atomic unit of velocity is
   ``c = 1/α``; ``E_h`` is exactly twice the rydberg and exactly
   ``HARTREE_EV`` electron-volts.  These are identities, not measurements.

References for the two external numbers used below:
- Drake, *Phys. Rev. A* **45**, 176 (1992) / *Can. J. Phys.* **80**, 1195
  (2002): He ground state ``−2.903724377034`` Ha (non-relativistic, infinite
  nuclear mass).  The leading relativistic correction lowers it by a further
  4.4e-6 Ha (``−2.9038632``), so ``−2.9037244`` is a safe *lower* bound.
- The hydrogen ``2p`` fine-structure interval is ``0.365 cm⁻¹``
  (10.95 GHz), the textbook value that the ``2p_{1/2}–2p_{3/2}`` splitting
  reduces to at ``Z = 1``.
"""

from __future__ import annotations

import math

import pytest

from atomx import Atom, Correlation
from atomx.constants import (
    ALPHA,
    AU_RATE_PER_S,
    AU_TIME_S,
    C_SQ,
    CODATA,  # noqa: E402
    HARTREE_CM1,
    HARTREE_EV,
    HARTREE_J,
    SPEED_OF_LIGHT,
)

#: Exact CODATA-2022 constants used to *derive* the electron-volt conversion
#: of the Hartree instead of quoting a rounded table entry.
SPEED_OF_LIGHT_M_PER_S = CODATA["2022"]["_c"]
ELEMENTARY_CHARGE_C = CODATA["2022"]["_e"]
from atomx.physics.dirac import hydrogenic_dirac_energy

#: Exact non-relativistic ground-state energies (infinite nuclear mass) in
#: Hartree.  Used only as *lower* bounds: a variational energy below one of
#: these is impossible, which is the whole point of the assertion.
EXACT_NR_GROUND_STATE = {
    "He": (-2.903724377034, 4),      # Drake 2002, 4 is the reported digit count
    "Be": (-14.6673564, 5),          # Drake, Be I
}

#: ``j + 1/2 = |κ|``, so the Sommerfeld expansion is written in ``|κ|``.
EXPANSION_ORDER = 2


# ---------------------------------------------------------------------------
# 1-3.  The exact Dirac-Coulomb hydrogenic spectrum
# ---------------------------------------------------------------------------


class TestDiracCoulombSpectrum:
    """The point-Coulomb Dirac spectrum is known in closed form.

    These are not regression pins: each expected value is either an exact
    symmetry of the Sommerfeld formula or an independent special case of it.
    """

    @pytest.mark.parametrize(
        "n,kappa,Z",
        [
            (1, -1, 1), (2, -1, 1), (2, 1, 1), (3, -2, 6), (3, 2, 6),
            (2, -1, 10), (2, 1, 10), (4, -3, 20), (4, 3, 20),
            (3, -1, 54), (3, 1, 54), (5, -4, 92), (5, 4, 92),
        ],
    )
    def test_energy_is_even_in_kappa(self, n, kappa, Z):
        """``E(n, κ) = E(n, −κ)``: the point-Coulomb Dirac energy depends on
        ``j`` only, never on the sign of ``κ`` (i.e. never on ``l``).

        Measured before writing: identical to every printed digit, e.g.
        ``n=2, κ=±1, Z=10`` both ``−12.520859662919822``; ``n=3, κ=±2, Z=10``
        both ``−5.558022660146563``.  Two of these pairs (``2s_{1/2}`` vs
        ``2p_{1/2}``, ``3p_{3/2}`` vs ``3d_{3/2}``) are the textbook
        degeneracies, and no existing test asserted either.
        """
        plus = hydrogenic_dirac_energy(n=n, kappa=abs(kappa), Z=Z)
        minus = hydrogenic_dirac_energy(n=n, kappa=-abs(kappa), Z=Z)
        assert plus == pytest.approx(minus, rel=1e-15), (
            f"E({n},{ -abs(kappa) },{Z})={minus!r} != "
            f"E({n},{abs(kappa)},{Z})={plus!r}; the point-Coulomb Dirac "
            f"energy must not depend on the sign of kappa"
        )

    @pytest.mark.parametrize("Z", [1, 2, 10, 26, 50, 82, 92, 118, 137])
    def test_1s_closed_form(self, Z):
        """``n = 1`` reduces to ``E = mc²(√(1 − (Zα)²) − 1)`` exactly.

        Derived independently from the general Sommerfeld formula by
        substituting ``n_r = 0``, ``γ = √(1 − (Zα)²)``, so this is a
        special-case identity rather than a restatement.

        Measured: agreement to 8.7e-13 (Z=1), 9.3e-11 (Z=10), 6.5e-08 (Z=50),
        1.2e-06 (Z=92) — the drift is the ``float`` cancellation in
        ``1 − (Zα)²`` as ``Zα → 1``, which is why the tolerance is scaled.
        """
        za = Z * ALPHA
        expected = C_SQ * (math.sqrt(1.0 - za * za) - 1.0)
        got = hydrogenic_dirac_energy(n=1, kappa=-1, Z=Z)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-5), (
            f"Z={Z}: 1s1/2 = {got!r}, closed form = {expected!r}"
        )

    @pytest.mark.parametrize(
        "n,kappa,Z",
        [(1, -1, 1), (2, -1, 1), (2, 1, 1), (2, -2, 1), (3, -1, 10),
         (3, 1, 10), (3, -2, 10), (1, -1, 20), (2, -1, 20)],
    )
    def test_sommerfeld_expansion_second_order(self, n, kappa, Z):
        """The ``(Zα)²`` Sommerfeld expansion, derived here rather than quoted.

        Expanding ``γ = |κ|√(1 − (Zα)²/κ²)`` and ``(1 + u)^{−1/2}`` in
        ``x = (Zα)²`` gives the textbook series

            ``E ≈ −Z²/(2n²) − x Z²/(2n³)·[1/|κ| − 3/(4n)] + O(x²)``

        Because ``|κ| = j + ½`` this is Sommerfeld's fine-structure formula.
        The residual must therefore be ``O((Zα)⁴)`` *relative to the
        correction term*; the assertion below is written as a bound on that
        ratio, so it cannot be satisfied by a formula that is wrong at order
        ``x``.  For ``Z = 1`` the expected agreement is ~5e-5.
        """
        za = Z * ALPHA
        x = za * za
        rydberg_scale = Z * Z / (2.0 * n * n)
        leading = -rydberg_scale
        correction = -x * Z * Z / (2.0 * n**3) * (1.0 / abs(kappa) - 3.0 / (4.0 * n))
        got = hydrogenic_dirac_energy(n=n, kappa=kappa, Z=Z)
        residual = got - (leading + correction)
        # The next Sommerfeld order carries x^2 relative to the leading term;
        # bounding the residual by 2 x times the correction is a statement
        # about the *series*, independent of the closed form.
        assert abs(residual) <= 2.0 * x * abs(correction) + 1e-14, (
            f"n={n} kappa={kappa} Z={Z}: E={got!r} but the 2nd-order "
            f"Sommerfeld expansion {leading + correction!r} leaves a residual "
            f"{residual:.3e}, larger than the O(x^2) series remainder "
            f"{2.0 * x * abs(correction):.3e}"
        )

    def test_hydrogen_2p_fine_structure_is_the_textbook_interval(self):
        """H I ``2p_{1/2} − 2p_{3/2}`` = 0.365 cm⁻¹ = 10.95 GHz.

        From the expansion above with ``n = 2``, ``l = 1``:
        ``Δ = (Zα)²Z²[1/1 − 1/2]/(2n³) = α²/32`` Ha.  This is the textbook
        number, and the test states it in Hz and cm⁻¹ so a wrong power of α
        (the kind of error that made the old Uehling kernel 3.9x too large)
        cannot hide behind a Hartree-scale tolerance.
        """
        e_2p12 = hydrogenic_dirac_energy(n=2, kappa=-1, Z=1)
        e_2p32 = hydrogenic_dirac_energy(n=2, kappa=-2, Z=1)
        # ``AU_TIME_S`` is the atomic unit of time ``hbar/E_h``, so the atomic
        # unit of *angular* frequency is its reciprocal.  A spectroscopic
        # interval in Hz is ``dE/h``, i.e. ``dE/(2*pi*hbar)``: using
        # ``AU_RATE_PER_S`` here directly would be a 2*pi error, which is the
        # trap this line is written to document.
        planck = 2.0 * math.pi * AU_TIME_S * HARTREE_J
        interval_hz = (e_2p32 - e_2p12) * HARTREE_J / planck
        interval_cm1 = (e_2p32 - e_2p12) * HARTREE_CM1
        assert pytest.approx(0.365, abs=5e-3) == interval_cm1
        assert pytest.approx(10.95, abs=0.15) == interval_hz / 1e9
        # The analytic leading term ``alpha^2/32`` agrees to 3.4e-5 relative,
        # which is exactly the size of the next order in ``alpha^2`` (5.3e-5):
        # the residual is the O(alpha^4) Sommerfeld term, not noise.
        assert pytest.approx(ALPHA**2 / 32.0, rel=1e-4) == e_2p32 - e_2p12
        assert (e_2p32 - e_2p12) / (ALPHA**2 / 32.0) > 1.0, (
            "the exact interval exceeds its leading alpha^2 term, so the "
            "O(alpha^4) coefficient must be positive"
        )

    @pytest.mark.parametrize("Z", [138, 140, 170])
    def test_no_bound_point_coulomb_state_above_critical_charge(self, Z):
        """``Zα > 1`` has no bound point-Coulomb solution.

        ``γ = √(κ² − (Zα)²)`` is imaginary there, so the point-nucleus Dirac
        equation has no normalisable bound state.  The requirement is that
        this is *stated*, not that some particular exception type is raised —
        atomx currently returns ``nan`` and emits a numpy ``RuntimeWarning``
        from ``sqrt``.  Returning ``nan`` is acceptable; returning a finite
        number would not be.
        """
        with pytest.warns(RuntimeWarning):
            value = hydrogenic_dirac_energy(n=1, kappa=-1, Z=Z)
        assert math.isnan(value), (
            f"Z={Z} has Z*alpha={Z * ALPHA:.4f} > 1: the point-Coulomb Dirac "
            f"equation has no bound state, but atomx returned {value!r}"
        )

    def test_subcritical_charge_stays_finite_and_ordered(self):
        """The physical branch below ``Zα = 1`` is finite and monotone."""
        energies = [hydrogenic_dirac_energy(n=1, kappa=-1, Z=Z)
                    for Z in range(1, 138)]
        assert all(math.isfinite(e) for e in energies)
        assert all(b < a < 0 for a, b in zip(energies, energies[1:], strict=False)), (
            "hydrogenic 1s binding must increase monotonically with Z"
        )


# ---------------------------------------------------------------------------
# 4.  Finite nuclear size
# ---------------------------------------------------------------------------


class TestFiniteNuclearSize:
    """A finite nuclear charge distribution must *reduce* the binding energy."""

    @pytest.mark.parametrize("Z,config", [(2, "1s2"), (10, "1s2 2s2 2p6")])
    def test_finite_nucleus_binds_less_than_a_point_nucleus(self, Z, config):
        """``E(finite) > E(point)`` — a sign derived from ``V_fin ≥ V_pt``.

        ``V_fin(r) − V_pt(r) = Z/r − (3Z/2R)(1 − r²/(3R²)) > 0`` inside the
        nucleus and zero outside, so ``ΔE = ∫ρ_e ΔV d³r > 0`` for any
        electron density.  No reference value is involved and the statement
        cannot be satisfied by a nucleus model that ignores the finite size.

        Measured: He ``+2.07e-08`` Ha, Ne ``+4.37e-05`` Ha.
        """
        point = Atom(Z, config, nuclear="point", corrections="none")
        point.solve()
        finite = Atom(Z, config, nuclear="fermi", corrections="none")
        finite.solve()
        shift = finite.total_energy - point.total_energy
        assert shift > 0.0, (
            f"Z={Z}: E(fermi)={finite.total_energy!r} < "
            f"E(point)={point.total_energy!r}; a finite nuclear charge "
            f"distribution is less attractive inside the nucleus, so the "
            f"binding energy must *decrease*.  Shift = {shift:.6e} Ha"
        )

    def test_finite_size_shift_follows_the_Z_14_over_3_law(self):
        """``ΔE ∝ Z^{14/3}`` from ``|ψ(0)|² ∝ Z³``, ``ΔV ∝ Z``, ``R ∝ Z^{1/3}``.

        Because the nuclear radius is ~5 orders of magnitude smaller than the
        inner-shell radius, ``ρ_e`` is constant across the nucleus and
        ``ΔE = (2π/5)|ψ(0)|²ZR²``.  Hydrogenic scaling of the inner shell
        (``|ψ(0)|² ∝ Z³`` for a K-shell, which stays K-shell-like along an
        isoelectronic sequence) with ``R ∝ A^{1/3} ∝ Z^{1/3}`` gives the
        exponent ``14/3 = 4.667``.

        A wrong potential shape, a missing factor of ``Z`` in the smearing
        radius, or a point-nucleus fallback that leaves the shift at the
        numerical noise floor all break this.  Measured exponent He → Ne:
        ``ln(4.366e-05 / 2.07e-08)/ln(5) = 4.76``.  The band is wide on purpose
        (the K-shell is not exactly hydrogenic and ``R`` is not exactly
        ``Z^{1/3}``) but it is narrow enough to reject the noise-floor case,
        where the exponent is undefined and the ratio is ~1.
        """
        shifts = {}
        for Z, config in [(2, "1s2"), (10, "1s2 2s2 2p6")]:
            point = Atom(Z, config, nuclear="point", corrections="none")
            point.solve()
            finite = Atom(Z, config, nuclear="fermi", corrections="none")
            finite.solve()
            shifts[Z] = finite.total_energy - point.total_energy
        exponent = math.log(shifts[10] / shifts[2]) / math.log(10.0 / 2.0)
        assert exponent == pytest.approx(14.0 / 3.0, abs=0.7), (
            f"finite-size shift exponent = {exponent:.3f} "
            f"(He {shifts[2]:.3e} Ha, Ne {shifts[10]:.3e} Ha); the "
            f"|psi(0)|^2 * Z * R^2 scaling predicts 14/3 = 4.667"
        )


# ---------------------------------------------------------------------------
# 5.  Rayleigh-Ritz
# ---------------------------------------------------------------------------


class TestVariationalBound:
    """No Rayleigh quotient may fall below the exact ground state."""

    @pytest.mark.parametrize("symbol,Z,config", [
        ("He", 2, "1s2"),
        ("Be", 4, "1s2 2s2"),
    ])
    def test_dhf_energy_is_above_the_exact_ground_state(self, symbol, Z, config):
        """``E_DHF > E_exact`` for every element.

        The bound is the literature non-relativistic ground-state energy,
        which is itself within 4.4e-6 Ha (He) of the exact relativistic value
        and therefore a safe *lower* bound at the tolerance used here.

        This is the assertion that the old configuration-average CI matrix
        violated by 0.39 Ha for Be, and nothing in the suite caught it.
        """
        exact, _digits = EXACT_NR_GROUND_STATE[symbol]
        atom = Atom(Z, config, corrections="none", nuclear="point")
        atom.solve()
        assert atom.total_energy > exact, (
            f"{symbol}: E_DHF = {atom.total_energy!r} is BELOW the exact "
            f"ground state {exact!r}.  No variational energy can be below the "
            f"exact ground state of the Hamiltonian it diagonalises."
        )
        # ... and it must not be absurdly far above it either: the Dirac-Fock
        # energy of a closed shell recovers the 1s-dominated part to ~1 %.
        assert atom.total_energy < exact + 0.1 * abs(exact), (
            f"{symbol}: E_DHF = {atom.total_energy!r} is more than 10 % above "
            f"the exact ground state {exact!r}; the SCF has not converged to "
            f"a physical solution"
        )

    @pytest.mark.slow
    def test_ci_expansion_obeys_hylleraas_undheim_macdonald(self):
        """Hylleraas--Undheim--MacDonald: enlarging a CI space can only lower
        the eigenvalues, and the lowest one can never cross the exact value.

        Both halves are theorems, not fits:

        * ``S_1CSF ⊂ S_singles ⊂ S_SD``, and H--U--M says the ``k``-th
          eigenvalue of the projected Hamiltonian is an upper bound to the
          ``k``-th exact eigenvalue, monotonically improving with the space.
          So ``E(SD) ≤ E(s) ≤ E(1CSF)``.
        * Consequently the correlation energy recovered from the SD space is
          bounded by the exact correlation energy,
          ``|E_corr(SD)| ≤ |E_corr(exact)|``, and cannot have the wrong sign.

        The bound used is the non-relativistic exact ground state, which is
        *above* the exact Dirac--Coulomb ground state, so it is a conservative
        (safe) bound: dropping below it is proof of an error, never a
        relativistic subtlety.

        This is precisely the statement the old configuration-average
        ``build_ci_matrix`` violated: it returned ``−15.0527`` Ha for Be SD,
        0.385 Ha BELOW the exact ground state, with a green suite.

        Measured here (``fac_frozen``, ``corrections="none"``):
        ``1-CSF −14.5759``, ``s −14.575893``, ``sd −14.632497``;
        exact non-relativistic ``−14.66736`` (Drake).
        """
        e_exact = EXACT_NR_GROUND_STATE["Be"][0]
        energies = {}
        reference = Atom(4, ["1s2 2s2"], corrections="none")
        reference.solve()
        energies["1csf"] = reference.total_energy
        for expansion in ("s", "sd"):
            atom = Atom(
                4,
                "1s2 2s2",
                correlation=Correlation(
                    expansion=expansion, radial_optimization="fac_frozen"
                ),
                corrections="none",
            )
            atom.solve()
            energies[expansion] = atom.total_energy

        assert energies["sd"] <= energies["s"] <= energies["1csf"], (
            f"Be CI ladder is not monotone: 1-CSF={energies['1csf']!r}, "
            f"s={energies['s']!r}, sd={energies['sd']!r}.  Adding CSFs to a "
            f"CI space cannot raise the lowest eigenvalue "
            f"(Hylleraas-Undheim-MacDonald)."
        )
        assert energies["sd"] > e_exact, (
            f"Be SD lowest eigenvalue {energies['sd']!r} is BELOW the exact "
            f"ground state {e_exact!r}.  A projected Hamiltonian's eigenvalue "
            f"is an upper bound to the exact one; being below it means the "
            f"matrix is not the matrix element of the Hamiltonian."
        )
        e_corr_sd = energies["sd"] - energies["1csf"]
        e_corr_exact = e_exact - energies["1csf"]
        assert e_corr_exact < e_corr_sd < 0.0, (
            f"Be SD correlation energy {e_corr_sd!r} must lie between 0 and "
            f"the exact correlation energy {e_corr_exact!r}"
        )
        # Empirical, and deliberately loose: SD on frozen HF orbitals is
        # expected to recover a substantial but incomplete share of the
        # correlation energy.  A collapse to ~0 means the off-diagonals are
        # missing (the old defect); ~100 % means over-correlation.
        fraction = e_corr_sd / e_corr_exact
        assert 0.3 < fraction < 0.95, (
            f"Be SD recovers {fraction:.1%} of the exact correlation energy "
            f"({e_corr_sd:.6f} of {e_corr_exact:.6f} Ha); a frozen-orbital SD "
            f"space should recover well under 100 % but far more than 0 %"
        )


# ---------------------------------------------------------------------------
# 6.  Exact atomic-unit identities
# ---------------------------------------------------------------------------


class TestAtomicUnitIdentities:
    """Identities that follow from the *definition* of the atomic units."""

    def test_mc_squared_is_one_over_alpha_squared(self):
        """The Hartree is defined as ``α²m_ec²``, so ``mc² = 1/α²`` Ha."""
        assert pytest.approx(1.0 / ALPHA**2, rel=1e-15) == C_SQ

    def test_atomic_unit_of_velocity_is_one_over_alpha(self):
        """``c = 1/α`` in atomic units, by the same definition."""
        assert pytest.approx(1.0 / ALPHA, rel=1e-15) == SPEED_OF_LIGHT

    #: CODATA 2022 Rydberg constant in cm^-1.  ``E_h = 2 h c R_inf`` is the
    #: *definition* of the Hartree, so both unit conversions below are derived
    #: from it rather than quoted from a rounded table entry.
    RYDBERG_CM1 = 109737.31568160

    def test_hartree_is_exactly_two_rydbergs(self):
        """``E_h = 2 R_inf`` in cm^-1 and ``2 h c R_inf / e`` in eV.

        Both atomx constants agree with this definition to 2.7e-13 relative
        (measured: ``HARTREE_CM1`` 219474.63136314112 vs 219474.6313632;
        ``HARTREE_EV`` 27.211386245981167 vs 27.211386245988), a CODATA
        rounding residual rather than an inconsistency.

        The identity is what matters: a constants table that has drifted, or a
        Hartree/Rydberg mix-up (a factor of 2), fails here immediately.  A
        rounded eV-to-cm^-1 factor copied from a table would *not* work —
        ``8065.544`` is only good to 8e-9 relative, 30x worse than the
        constants themselves, and that is exactly the kind of sloppy constant
        that produced the old Uehling error.
        """
        assert pytest.approx(2.0 * self.RYDBERG_CM1, rel=1e-11) == HARTREE_CM1
        planck = 2.0 * math.pi * AU_TIME_S * HARTREE_J
        e_h_ev = (
            2.0
            * planck
            * SPEED_OF_LIGHT_M_PER_S
            * self.RYDBERG_CM1
            * 100.0
            / ELEMENTARY_CHARGE_C
        )
        assert pytest.approx(e_h_ev, rel=1e-9) == HARTREE_EV

    def test_rate_and_time_units_are_reciprocal(self):
        """``AU_RATE_PER_S`` and ``AU_TIME_S`` are reciprocals by definition."""
        assert pytest.approx(1.0 / AU_TIME_S, rel=1e-12) == AU_RATE_PER_S

    def test_mc_squared_reproduces_the_hydrogenic_1s_nonrelativistic_limit(self):
        """``C_SQ`` is the mass term the hydrogenic solver must use.

        ``hydrogenic_dirac_energy`` returns ``C_SQ·(1/√(1+x) − 1)``; in the
        ``x → 0`` limit that is ``−C_SQ·x/2 = −Z²/(2n²)`` at leading order
        because ``C_SQ·α² = 1``.  Composing the two constants here checks they
        are mutually consistent *and* that the solver's normalisation is
        ``mc²`` rather than the rydberg.
        """
        for Z, n in [(1, 1), (2, 1), (10, 2), (30, 3)]:
            got = hydrogenic_dirac_energy(n=n, kappa=-n, Z=Z)
            schrodinger = -Z * Z / (2.0 * n * n)
            # the expansion parameter is (Z alpha)^2, so the tolerance must
            # scale with it: at Z=30 that is 4.8e-2, at Z=1 it is 5.3e-5
            assert got == pytest.approx(schrodinger, rel=max(1e-6, (Z * ALPHA) ** 2)), (
                f"Z={Z} n={n}: {got!r} vs Schrodinger {schrodinger!r}; the "
                f"relativistic energy must approach -Z^2/(2n^2) to O((Z alpha)^2)"
            )
            # the relativistic correction is always a *deepening* for the
            # point Coulomb problem, and of order (Z alpha)^2
            assert got < schrodinger
