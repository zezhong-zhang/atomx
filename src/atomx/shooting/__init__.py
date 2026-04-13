"""ODE shooting solver — Radial/Salvat-style.

Implements both Schrödinger (SBOUND) and Dirac (DBOUND) bound-state solvers
using power-series integration, matching Salvat's RADIAL package exactly.

Combined with local Slater exchange in an SCF loop (DHFS), this reproduces
the same eigenvalues and total energies as Salvat's DHFS code.

Key algorithms (all from Salvat's RADIAL package):
- Power-series integration (DIR0/SCH0) with adaptive step control
- Natural cubic spline potential interpolation (SPLIN0/VINT)
- Non-uniform grid (SGRID): A*R + B*log(R+C) + D = I
- 6-point Lagrange quadrature (SLAG6)
- Eigenvalue refinement from wavefunction discontinuity (DBOUND/SBOUND)
- Total energy via Slater integrals + DCJ3 angular coefficients (EXPVAL)
"""

from atomx.shooting.solver import ShootingSolver, ShootingResult
