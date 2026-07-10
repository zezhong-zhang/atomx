"""Derived observables computed from an :class:`AtomicResult`.

Everything under this sub-package is **read-only over the result**:
it never mutates solver state or triggers a new SCF.  This is the
atomic-physics analogue of PySCF's ``pyscf.prop`` — a home for
oscillator strengths, hyperfine constants, isotope shifts, g-factors,
polarisabilities, parity-violating amplitudes, and so on.

Current modules
---------------
:mod:`atomx.properties.transitions`
    E1 / M1 / E2 / M2 matrix elements, oscillator strengths, Einstein
    A coefficients.

:mod:`atomx.properties.hyperfine`
    Magnetic-dipole A and electric-quadrupole B constants.

:mod:`atomx.properties.isotope_shift`
    Normal and specific mass shift, field shift, full isotope-shift
    compositions.

:mod:`atomx.properties.zeeman`
    Landé g_J factors and Zeeman energy shifts.

Future modules (Phase 7 scaffolds)
----------------------------------
:mod:`atomx.properties.polarizability`
    Static and dynamic dipole polarisability α(ω).
:mod:`atomx.properties.g_factor`
    Bound-electron Landé g_J beyond leading non-relativistic order.
:mod:`atomx.properties.pnc`
    Parity-non-conservation weak-charge amplitudes.
:mod:`atomx.properties.edm`
    T,P-odd Schiff-moment enhancement factors.
"""
