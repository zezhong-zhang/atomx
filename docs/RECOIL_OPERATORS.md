# Nuclear recoil operators and isotope properties

The leading recoil Hamiltonian is

```
H_recoil = sum_i p_i^2/(2M) + sum_(i<j) p_i.p_j/M.
```

The second term defines the specific mass shift (SMS). For a closed-shell
independent determinant, its direct term vanishes by parity and its exchange
contribution is negative. Reduced angular coefficients already sum magnetic
substates: pair weights are occupation fractions `q_a/(2j_a+1)` times
`q_b/(2j_b+1)`, not the two electron counts again.

## Shared calculation

`corrections/specific_recoil.py` provides the Vinti momentum matrix and its
point-Coulomb Breit correction (GRASP RIS4 SMS2/SMS3). Both the energy-correction
stack and `properties/isotope_shift/sms_abinitio.py` use this implementation.
`recoil_radial_matrices` evaluates each spinor and radial derivative once,
then assembles the leading and corrected matrices. It accepts either a
B-spline workspace or the shooting grid's native quadrature.

The closed-shell property API is:

```python
from atomx.properties.isotope_shift import compute_sms_abinitio

sms = compute_sms_abinitio(orbitals, workspace, Z=Z)  # point-Coulomb Breit recoil
leading = compute_sms_abinitio(orbitals, workspace)  # p_i.p_j only
energy_shift = sms.for_mass(nuclear_mass_in_electron_masses)
```

Orbitals are `(P, Q, occupation, kappa, label)` tuples. Open-shell inputs are
rejected by this closed-shell API. Use the CSF recoil matrix for a coupled
state or configuration-average recoil for an explicitly averaged ensemble.
Correlation, finite-size recoil, radiative recoil and higher orders in
inverse nuclear mass are not included by evaluating this first-order operator.
Finite-nucleus orbitals alone do not supply finite-size recoil.

## Physics defects corrected in the audit

The former property implementation used `abs(epsilon_a-epsilon_b)*<a|r|b>`
as momentum. This is not canonical momentum for a Dirac Hamiltonian:
`[H_D,r]` gives the Dirac velocity, and nonlocal HF exchange also contributes
to the commutator. It further multiplied by full subshell electron counts,
double-counting magnetic degeneracies. Its neutral-Ne value near -117.84 Ha·m_e
must not be used. Direct leading recoil is near -15.00 Ha·m_e on the same
closed-shell mean-field model. Correlated experimental isotope shifts are
a separate comparison, not established by this number.

The relativistic SMS2 term also mixed tensor phases. Its reduced Pauli
matrix used the standard convention, while the C denominator used Atomx's
legacy reversed-3j convention. For unequal-j rank-one channels these differ
in sign. SMS2 now uses GRASP CRE / FAC ReducedCL, matching the Pauli tensor.
Independent spinor angular integrals check this ratio. The legacy CI tensor
convention is documented separately; it has not been changed globally.

## Evidence and remaining limits

- Analytic hydrogenic `1s^2 2p^6` spinors give
  `K_SMS = -2 [32 Z/(27 sqrt(6))]^2`; tests check the sign, scale and magnetic
  substate sum at Z = 1, 2, 20.
- A read-only `DumpVinti` diagnostic was added to an isolated sfac frontend.
  It calls the supplied, unchanged `libfac.a`. The modified frontend,
  build command, library/source hashes, input scripts and outputs are retained
  in [the live recoil panel](../benchmarks/runs/2026-09-29-recoil-radial-live/).
- Eight channels in both directions at Z = 1, 10, 26, 54, 82 give 80 radial
  comparisons. Leading and Breit-corrected magnitudes agree within 41 ppm.
  The original 4000-point Z=82 repeat increased allocation without refining
  FAC's inner spacing and was inconclusive. A subsequent Ne/Pb repeat with
  grid-density factor 4 and tightened orbital energies agrees within
  2.9 ppm for both leading and Breit-corrected integrals. This is improved
  evidence for those two charges, not sub-ppm validation of the full panel.
  See [reference controls](TRANSVERSE_INTERACTION.md#reference-refinement-and-common-spinor-diagnosis)
  and [raw refined recoil data](../benchmarks/runs/2026-09-29-recoil-precision-reference/).
- The focused isotope/recoil/correction suite passes 124 tests. The previous
  neon basis gate is replaced by a direct-momentum calculation at a tighter
  SCF tolerance and a 50/70/90 B-spline refinement. The original 1e-9 relative
  property agreement gate is retained for the resolved 70/90 pair.
- In a neutral-Ne timing probe, returning both matrices together takes about
  0.26 ms versus 0.45 ms through separate calls on B-splines, and 0.81 ms
  versus 1.53 ms on shooting. These are operator timings, not total SCF speedups.
- The subsequent [radial convergence audit](DF_SOLVER_CONTROLS.md#independent-orbital-accuracy-for-properties)
  resolved the Ne engine gap: the fixed Fermi B-spline first interval needed
  nuclear-scale resolution, and shooting needed an independent orbital
  update target for a first-order property. Refined engines agree within
  4e-10 relative in SMS. This does not certify default tolerances for all
  properties or the external DBSR orbital residual.

These checks validate selected radial operators and their assembly. They do
not complete the requested isotope, correlated-spectrum or all-property audit.
