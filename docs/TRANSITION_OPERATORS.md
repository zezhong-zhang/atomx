# Radiative multipole operators

Repository availability: paths marked “local evidence” are outside the tracked
checkout. They retain the original evidence location and are not GitHub links.

One-photon transition rates use finite-frequency electric and magnetic
multipoles. The operator retains the spherical Bessel functions of
`k r = omega r / c`; it does not truncate them to powers of radius.
This matters for heavy ions and for weak lines whose leading amplitude
cancels. The static operator remains useful for static response and for
explicit long-wavelength diagnostics.

## Physics and numerical boundaries

The [Sn12+ CI baseline](TIN_EUV_BASELINE.md) explicitly selects length-gauge,
long-wavelength E1. Its reusable `angular.one_body_tensor.one_body_tensor_matrix`
projects one sparse signed magnetic action across complete CSF blocks and
matches the original pairwise tensor terms. This batching changes evaluation
and storage, not photon physics; it does not replace the finite-frequency
operator used by other consumers.

Optional complete CSF-coordinate columns give `B.conj().T @ T @ K` by
contracting into the magnetic representation before the operator action.
All CSF contributions remain. A caller-owned `TensorProjectionCache` reuses
immutable projections within an explicit byte budget, keyed by ordered CSFs,
magnetic/orbital layout and all signed coefficient bytes. Changes to a radial
operator reuse only the angular projection, never the previous observable.
The tin consumer retains every CI root while separately declaring its
fold0–3 emission endpoint window; higher-fold configurations still mix into
those states. This window is not a CI-space truncation.

- `properties/transitions/multipole.py` owns the electric Babushkin and
  Coulomb operators and the magnetic operator, with their physical units.
- `numerics/special.py` evaluates the regular scaled Bessel functions.
  A series near zero avoids division of tiny quantities; it is a numerical
  evaluation of the full function, not a physical long-wavelength switch.
- `numerics/radial_workspace.py` provides the quadrature and spinors, either
  from B-spline coefficients or the shooting grid. The physics is shared.
- Transition, photoionization, radiative recombination and the orbital E1
  dielectronic-recombination width drivers supply the photon energy.

The electric kernels use the **signed** frequency `omega = E_f - E_i` in
Hartree. Reverse both the orbital order and the frequency sign together.
Magnetic kernels take the positive photon energy. Rate prefactors always
use the positive emitted energy and the upper state's degeneracy.

## Electric normalization

Let `F_L(r,k) = (2L+1)!! j_L(kr)/k^L`, with limit `r^L` at zero,
`d = kappa_i-kappa_f`, `rho = P_i P_f + Q_i Q_f`,
`s = P_i Q_f + Q_i P_f`, and `a = P_i Q_f - Q_i P_f`.
The length radial integral is

```
R_B = integral [rho F_L + k F_(L+1)/(2L+3) (d s/(L+1) + a)] dr.
```

The velocity radial integral is

```
R_C = integral [F_(L-1)/k (-d s + L a)
             + k F_(L+1)/((2L+1)(2L+3)) (d L s/(L+1) + L a)] dr.
```

Both multiply the same reduced spherical-harmonic coefficient and use the
same Einstein-rate normalization. These expressions are the normalized
forms of the supplied FAC `faclib/radial.c::MultipoleRadialFRGrid`, including
the extra velocity-form division by photon momentum in `MultipoleRadialFR`.
FAC's reduced amplitude and Atomx's reduced amplitude have different rank
normalizations; the comparison uses physical rates, not their raw squares.

For exact eigenstates of a common local Dirac Hamiltonian, the two gauges
agree. With truncated CI spaces or nonlocal orbital exchange, gauge
agreement is an accuracy diagnostic; equality is not forced in code.

## Callers and diagnostics

`atom.transitions(gauge="both")` retains both electric gauges and reports
`photon_approximation="finite_wavelength"`. The default reported rate is
Babushkin/length. Magnetic gauge entries contain the same operator and
are not independent checks.

At the radial-kernel level:

- `radial_eL(..., kappa_i=ki, kappa_f=kf, omega=E_f-E_i)` is finite frequency.
- `radial_eL(...)`, with no frequency, is the static electric multipole.
- `radial_eL_velocity(...)` requires kappas and nonzero signed frequency.
  `long_wavelength=True` explicitly selects its leading photon term.
- `radial_mL(..., omega=abs(E_f-E_i))` is finite frequency; omitted or zero
  frequency requests its static limit.

CI contractions use the **many-electron state energy difference** in every
orbital pair of a line. Replacing it with each orbital energy difference
would define a different operator. Angular coefficients and radial
integrals are cached separately; radial caches include the state frequency.

The determinant CI engine supports finite-frequency length gauge. Its
`CILineSet.trace` is `None` by default: a set of operators at different
line frequencies has no single mixing-invariant determinant trace.
`ci_line_strengths(..., photon_approximation="long_wavelength")` provides
the fixed-operator trace diagnostic and the corresponding static lines.

The lower-level `properties.transitions.ci_effective.finite_ci_electric_line`
also consumes two exact finite CI Bloch reductions across J/parity sectors.
It retains both wave operators and physical metrics, validates the native
CSF/spinor/nuclear/Hamiltonian identities and common orbital metric, and uses
one state-energy photon frequency for every orbital pair of each line.
Both electric gauges and explicit long-wavelength controls use the shared
radial kernels. Signed amplitudes, S, A and absorption f/gf retain their
upper/lower statistical weights; a fixed-frequency operator's other matrix
entries are not automatically different-frequency lines. Complete finite
Be+ determinant/state comparisons pass 16 controls at 5.78e-15 a.u. maximum
amplitude difference. This is an exact finite-space consumption check,
without basis/experimental accuracy or a production SD metric claim.
Details and complete arrays are local evidence under
`atomic_pci_sd_native_2026_10_09/finite_bloch_transitions.md`.
Four shared CSF/magnetic/tensor amplitude magnitude cutoffs have also been
removed; only exactly zero terms are omitted there. A normalized rotation in
a multiplicity-two 5g3 J=9/2 angular subspace retains a nonzero 2.78e-17
component. The broader affected angular/CSF/transition panel passes 240 checks
without skips. No performance ratio is inferred from those test runtimes.

## Verified scope

The initial live FAC panel at Z = 1, 10, 26, 54 and 82 uses one-electron point
nuclei, infinite nuclear mass, and no Breit/QED shifts. Its finite-frequency
E1–E2 rates agreed within 9e-6 relative; E1–E4 within 2.7e-4. These historical
bounds are superseded by the exact-frequency audit below. The Atomx electric
gauge rate ratio differs from unity by less than 4.2e-12 on this panel.
Raw inputs, outputs, source hashes and comparisons are retained under
`electric-finite-frequency-live` (local evidence: `../benchmarks/runs/2026-09-29-electric-finite-frequency-live/`).
Regression tests retain independent FAC output and common-spinor fixtures.

This verifies the one-body photon operator for these states. It does not
establish full correlated-spectrum or continuum accuracy. Two-photon decay
still uses its explicit dipole approximation; static polarizability and
other static-response operators appropriately retain the static kernel.
Photon retardation in a transition matrix element also does not implement
the frequency-dependent electron–electron interaction. That separate operator
is now available explicitly in frozen-orbital CI; see
[transverse exchange](TRANSVERSE_INTERACTION.md). Higher-order bound-state
QED corrections remain outside these operators.

### Reference grid and orbital convergence follow-up

FAC's allocated point count alone does not refine its near-origin ratio or
asymptotic spacing. The runner now exposes `--fac-grid-density` separately
and shares the control implementation with the recoil/transverse runners.
It also exposes `--fac-energy-tolerance` for the inner orbital solver;
this is independent of the FAC SCF convergence threshold.

The refined Ne/Pb repeat (local evidence: `../benchmarks/runs/2026-09-29-transitions-precision-reference/`)
uses density factor 4, 8,000 points and orbital-energy controls of 1e-12.
Finite-frequency E1/E2 rates agree within 0.9 ppm. Pb E3/E4 and M4 still
show discrepancies up to 263 ppm. Ne's weak 2s–1s M1 rate changes with the
reference orbital settings and differs by about 128 ppm in this run.
The following audit supersedes the unresolved photon-interpolation diagnosis
in that run. Its original outputs remain available.

### Exact line frequencies and shared spinors

FAC's `transition.c::SaveTransition` chooses its photon grid from all level
pairs passed to a transition-table call, before rejecting forbidden
transitions. When the relative photon-energy spread is below 0.001, it
evaluates the operator at one midpoint frequency. Thus even an angularly
forbidden partner can change the frequency used for the allowed line.
The radiative prefactor still uses the individual line energy. This is a
reference interpolation approximation, not a missing Atomx interaction.

`bench_reference_transitions.py` now defines resolved jj configurations
(FAC syntax `2[p-]1`, `2[p+]1`, etc.) and calls `TransitionTable` separately
for each upper state. No state indices or energies are guessed. The
resolved Ne/Pb table repeat (local evidence: `../benchmarks/runs/2026-09-29-transitions-resolved-frequency/`)
agrees with direct FAC operator rates within 0.31 ppm, consistent with the
printed table precision. The earlier Pb 263-ppm discrepancy disappears:
Pb E1–E4 agree with Atomx within 0.25 ppm in this repeat. Independent Ne
E4 spinors still differ by up to 2.9 ppm; resolving the frequency is not a
claim that all orbitals are exact.

The separate `bench_reference_photon_operator.py` diagnostic calls the
unmodified FAC `MultipoleRadialFR` at each exact line frequency and exports
full-precision spinors. Atomx evaluates its own operator on those spinors,
with the same photon energy. Integrals stop at the smaller orbital support
(`ilast`); zero padding must not enter the interpolant or create a false
tail. The common-grid integration uses quintic interpolation and eight-point
Gauss quadrature in each interval. A septic-interpolation regression also
checks the particularly cancellation-sensitive hydrogen case.

The 100-value panel (local evidence: `../benchmarks/runs/2026-09-29-photon-certified-panel/`)
contains Z=1,10,26,54,82, both electric fine-structure branches and gauges
through E4, and M1–M4. It uses point nuclei, infinite nuclear mass, no
Breit/QED shifts, FAC density factor 8, 16,000 allocated points and inner
energy controls of 1e-12.

| Comparison | Maximum relative rate difference |
|---|---:|
| Common spinors, all five charges | 3.44e-8 |
| Common spinors, Z=10,26,54,82 | 1.76e-9 |
| Independent spinors, Z=1 | 3.20e-3 (weak M1) |
| Independent spinors, Z=10 | 3.21e-5 (weak M1) |
| Independent spinors, Z=82 | 7.10e-7 |

This separates photon-operator agreement from orbital errors. FAC and
Atomx also use slightly different physical constants, so these differences
are not exclusively quadrature error. Tracked H/Ne/Pb fixtures retain raw
supported spinors, reference rates, settings and hashes. The common-spinor
regression gates are 5e-8 for H and 1e-8 for Ne/Pb; independent electric-rate
tests are tightened from 3e-4 to 1e-5. No production operator was adjusted
to match the reference.

### Weak M1 convergence

The forbidden nonrelativistic 2s–1s M1 amplitude is a small remainder of
larger radial contributions. Orbital-shape errors are amplified, especially
at low Z. The retained `m1-refinement.py/json` audit holds the Atomx photon
energy fixed while independently refining shooting and B-spline orbitals.
Across 2,000–16,000 shooting points the rate spread is below 0.11 ppm for H
and 0.004 ppm for Ne. B-spline bases of size 80, 120 and 180 agree with
these rates within those scales.

FAC's M1 discrepancy relative to the 180-spline result shrinks as follows:

| FAC density factor | H discrepancy (ppm) | Ne discrepancy (ppm) |
|---|---:|---:|
| 4 | 12,910 | 128.1 |
| 8 | 3,206 | 32.08 |
| 16 | 802.5 | 7.92 |

FAC photon-energy changes across these runs are far too small to explain
the rate changes. The approximately factor-four decrease per density
doubling is consistent with FAC's second-order near-origin small-component
recurrence (`orbital.c::DiracSmall`). Shared-spinor agreement and two
independent Atomx discretizations support a reference orbital-error
diagnosis, but do not prove that every residual arises from that recurrence.
The FAC weak-M1 values have not reached the common-spinor operator precision.
Recorded process timings include different work and are not speed ratios.
