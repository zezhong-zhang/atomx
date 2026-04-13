"""ODE shooting Dirac solver — Radial/Salvat-style.

Solves the radial Dirac equation by:
1. Outward integration from r=0 (power-series starting values)
2. Inward integration from r_max (WKB boundary conditions)
3. Matching at the classical turning point
4. Eigenvalue refinement from Q-discontinuity

Combined with local Slater exchange in an SCF loop, this achieves
the same accuracy as Salvat's DHFS code (~0.01-0.05%).
"""

from atomx.shooting.solver import ShootingSolver
