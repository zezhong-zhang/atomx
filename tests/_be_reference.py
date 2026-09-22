"""One place for every beryllium reference energy used in the test suite.

Before this module the same physical quantity was pinned to four different
literals in four files (``-14.66736``, ``-14.66736 - 0.005``, ``-14.6674``,
``-14.5734``) with no way for a reader to tell which protocol each number
belonged to.  Audit finding F7 (``docs/test_audit/S2_ci_mcdhf_correlation.md``).

The three quantities are *different physics* and are now named as such:

``BE_EXACT_NR``
    The **literature** non-relativistic Born-Oppenheimer ground state of
    beryllium at infinite nuclear mass, ``-14.66736`` Ha (Drake's
    Hylleraas-variation value).  This is the lower bound every
    non-relativistic variational calculation must stay above.

``BE_DC_LOWER_BOUND``
    The same bound lowered by the finite-nucleus Dirac-Coulomb shift.
    Relativity plus the finite (Fermi) nuclear size move the ground state
    **down** by ~2.5 mHa at Z = 4; 5 mHa of margin is kept so that a
    variational gate written against it cannot be tripped by the nuclear
    model.  It is a bound, not a measurement.

``BE_HF_1CSF``
    **atomx's own** single-configuration ``1s2 2s2`` energy in this suite's
    protocol (``Atom(Z=4, config="1s2 2s2", correlation=None,
    engine="bspline", corrections="none")``), measured on this machine:
    ``-14.575892`` Ha.  It is an upper bound to the exact ground state and
    a *reference*, not an external number; tests that have the solve in
    hand should prefer the live value over this literal.

For orientation, the numerical Hartree-Fock **limit** for Be is
``-14.573023`` Ha (the textbook value): atomx's finite B-spline basis sits
2.9 mHa above it, i.e. its 1-CSF energy is not converged to the HF limit.
"""

from __future__ import annotations

from functools import lru_cache

#: Literature non-relativistic Born-Oppenheimer ground state of Be
#: (infinite nuclear mass), Hartree.  Any non-relativistic variational
#: calculation must land above this.
BE_EXACT_NR = -14.66736

#: Finite-nucleus Dirac-Coulomb variant of the same bound: the physical
#: ground state is ~2.5 mHa *below* ``BE_EXACT_NR`` (relativistic
#: contraction plus the Fermi nuclear size at Z = 4), so a bound with 5 mHa
#: of margin is used wherever a relativistic solve is being gated.
BE_DC_LOWER_BOUND = BE_EXACT_NR - 0.005

#: Textbook numerical Hartree-Fock limit for Be (external, not atomx).
BE_HF_LIMIT = -14.573023

#: atomx's own 1-CSF ``1s2 2s2`` reference energy in this suite's protocol,
#: measured on this machine.  Kept for docstrings and for the few gates that
#: compare a *different* protocol against it; use :func:`be_1csf_energy`
#: when the test can afford the solve.
BE_HF_1CSF = -14.575892


@lru_cache(maxsize=1)
def be_1csf_energy() -> float:
    """atomx's 1-CSF ``1s2 2s2`` Be energy, solved once per test process.

    Measured rather than typed, which is what audit finding F7 asks for:
    the recorded literal drifted from the produced value by 0.18 Ha in
    ``test_sd_space_variational.py`` without any gate noticing.
    """
    from atomx import Atom

    atom = Atom(
        Z=4,
        config="1s2 2s2",
        correlation=None,
        engine="bspline",
        corrections="none",
    )
    atom.solve()
    return float(atom.total_energy)
