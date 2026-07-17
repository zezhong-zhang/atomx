"""Numbers quoted in the preface (docs/book/vol1/frontmatter/preface.md).

Run:      python docs/book/vol1/frontmatter/calc/preface_numbers.py
Enforced: tests/test_book_preface.py  (pytest -m book)
"""
import numpy as np


def analytic_dirac_1s(Z=1):
    """Point-nucleus Dirac 1s energy (Sommerfeld, n=1, kappa=-1), Hartree."""
    from atomx.constants import ALPHA, C
    return C ** 2 * (np.sqrt(1.0 - (Z * ALPHA) ** 2) - 1.0)


def digits_1s(numerics):
    """Digits of agreement of an engine's H 1s eigenvalue with the analytic value."""
    from atomx import Atom
    exact = analytic_dirac_1s()
    result = Atom(Z=1, nuclear="point", numerics=numerics).solve()
    e = result.orbital_energies["1s1/2"]
    return -np.log10(abs(e / exact - 1.0))


if __name__ == "__main__":
    exact = analytic_dirac_1s()
    print(f"analytic point-nucleus Dirac 1s: {exact:.13f} Eh")
    for engine in ("bspline", "shooting"):
        print(f"{engine:9s} engine: agrees to {digits_1s(engine):.1f} digits")
