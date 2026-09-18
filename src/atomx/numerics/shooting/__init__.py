"""ODE shooting solver for bound-state atomic orbitals.

An independent solver that integrates the radial Schrodinger or Dirac
equation directly on a numerical grid (no basis set expansion).  This
approach matches Salvat's RADIAL/DHFS Fortran package exactly.

This module is self-contained — it does NOT depend on the B-spline
pipeline (bspline/, dirac/, scf/).  It has its own grid, spline
interpolation, and quadrature routines.

Submodules
----------
solver : ShootingSolver SCF loop with local Slater exchange.
grid : Non-uniform radial grid via SGRID formula: A*R + B*log(R+C) + D = I.
spline : Natural cubic spline interpolation for potentials.
quadrature : 6-point Lagrange piecewise integration (SLAG6).
dirac_bound : Dirac eigenvalue search via wavefunction matching (DBOUND).
dirac_integrator : Outward/inward ODE integration of the Dirac equation.
dirac_power_series : Small-r power series starter for Dirac.
schrodinger_bound : Schrodinger eigenvalue search (SBOUND).
schrodinger_integrator : ODE integration of the Schrodinger equation.
schrodinger_power_series : Small-r power series starter for Schrodinger.
constants : Physical constants in Salvat's RADIAL convention.

All performance-critical routines are @nb.njit compiled.
"""

from atomx.numerics.shooting.solver import ShootingResult, ShootingSolver
