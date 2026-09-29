# One-body recoil and correlated correction audit

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The correction stack now evaluates the one-body Breit recoil operator by
default when a nuclear mass is supplied. `nms_mode="virial"` explicitly
selects the old `−E/M` estimate. The historical full-table ledger has not
been relabeled or updated; generation checks reject mixing its completed
virial results with new operator results.

The radial expression follows GRASP RIS4
`~/code/grasp/src/appl/ris4/rinti_nms.f90`, with analytic B-spline
derivatives (cubic-spline derivatives for sampled shooting orbitals).
It includes the large/small-component derivative and angular terms and
the relativistic Coulomb recoil terms. Off-diagonal radial integrals are
retained, and MCDHF contracts them with the full CSF matrix, including
coherences between different radial orbitals. The derivation is in
[Gaidamauskas et al. (2011)](https://arxiv.org/abs/1106.1988).
The implemented correction is a first-order state expectation; it does
not diagonalize the correction inside an accidentally degenerate manifold.

This fixes a silent omission: MCDHF previously ignored `nuclear_mass_me`
in an injected correction stack. Another fix makes VP-only and SE-only
stacks actually select their requested QED components; the old MCDHF
helper automatically enabled both. QED still uses diagonal orbital
expectations, so it does not yet have the recoil operator's radial
coherence treatment.

## Physical and numerical checks

For a point-Coulomb hydrogenic eigenstate, the Breit recoil parameter is
`K = (c⁴ − E_D²)/(2c²)`, with rest energy included in `E_D`. The analytic
1s radial functions at Z=1,20,80 reproduce `K=Z²/2`. The tests also check
invariance of a density-matrix contraction under radial orbital rotations,
explicit CI coherences, and resolved hydrogenic energy shifts.

Derivative observables need their own radial convergence check. For the
Z=80 point nucleus, a B-spline grid with `he=0.2` gives the following
relative deviation of the 1s recoil parameter from its analytic value:

| Innermost step parameter hi | Relative recoil deviation |
| --- | ---: |
| 0.25 | −4.3173% |
| 0.001 | −0.1336% |
| 0.00001 | −0.00755% |

The refined analytic gate uses `hi=1e-8`; a converged total energy alone
does not establish a converged recoil expectation. These point-nucleus
tests do not measure the error of finite-nucleus recoil physics.

## H–Ca result

`scripts/benchmark_first20_recoil.py` recomputed both ground stages for
all 20 neutral elements with shooting seeds, DBSR_HF static Breit,
model QED, and isotope-specific one-body Breit recoil. Both stages were
repeated on a denser inner grid. The reconstructed energy change is
entirely the change in recoil to within the benchmark's 1e-7 Ha gate.

The largest ionization update from the historical virial estimate is
**2.5735e-7 eV** (Ne I); the largest grid change in the corrected
ionization is **1.5098e-7 eV** (Ca I). Across all stages, the largest
change in the recoil expectation itself is **1.3821e-10 Ha**. The
19 evaluated matched NIST references still have **0.979798 eV** RMS
residual. One-body recoil cannot explain the large correlation gap.

## Corrections on actual correlated orbitals

`scripts/benchmark_he_combined_corrections.py` evaluates the correction
operators on the converged He SD-MCDHF states and a separately calculated
He II state, using the same He-4 nuclear mass. The n≤2 space (17 CSFs)
gives **24.423034 eV**, **0.164355 eV** below evaluated NIST He I.
The n≤3 space (98 CSFs) gives **24.536529 eV**, **0.050860 eV** below
NIST, after 433 orbital iterations (369.5 seconds). It differs from the
earlier additive n≤3 estimate by **0.000707 eV**. The correlation-space
change remains far larger than this correction-consistency change.
This replaces the additive diagnostic for that space with an actual
correlated-orbital correction calculation. It remains a limited model:
CI includes Gaunt but omits static Breit retardation; QED is a diagonal
orbital model; two-body, finite-size and radiative recoil are absent;
the active space and radial grid have not been extrapolated.

The interactive comparison displays these results separately from the
historical ledger and the earlier additive correlation diagnostic.

## Two-body recoil source mapping (implementation pending)

The local GRASP RIS4 source establishes the next operator components:

- `vinti.f90`: derivative and kappa-dependent angular parts of V(a,c).
- `rint_sms2.f90`: cross large/small radial products weighted by SIGMA_1
  and SIGMA_2 and -Z/c.
- `rint_sms3.f90`: (Q_a P_c - P_a Q_c)/r weighted by Z/c.
- `ris_cal.f90`: V2(a,c) = V(a,c) + SMS2(a,c)/CRE(kappa_a,1,kappa_c)
  + SMS3(a,c), only for allowed rank-one angular couplings.
- `smsnew.f90`: applies the rank-one two-electron angular coefficient to
  -[V2(a,c)V(b,d) + V2(b,d)V(a,c)]/2 and contracts with CI amplitudes.

Thus adding a scalar mass correction to each occupied orbital cannot
supply SMS. Its implementation requires the correlated two-body density,
angular selection, derivative integrals and relativistic cross terms.
The current recorded recoil remains one-body only; none of these newly
located two-body terms has been applied to the benchmark energies.

## Leading two-electron momentum diagnostic

`corrections/specific_recoil.py` now implements the Vinti radial matrix
and its rank-one two-electron CSF contraction for sum(i<j) p_i.p_j.
This is the leading momentum piece only; SMS2/SMS3 remain unimplemented.
It is not enabled in the production correction stack.

Independent nonrelativistic analytic checks at Z=1,2,20 verify
V(1s,2p)=32 Z/(27 sqrt(6)), V(2p,1s)=-V(1s,2p), singlet/triplet
1s2p eigenvalues +/-V²/3, and zero contribution for a one-electron state
and the uncorrelated 1s² determinant. On saved correlated B-spline
orbitals, integration-by-parts antisymmetry holds within 9e-16.

`audit_he_specific_recoil.py` gives K=0.17036854 au (n<=2) and
0.16342505 au (n<=3), corresponding to neutral level shifts +0.635560
and +0.609657 meV for He-4. The He II shift is zero, so the ionization
shifts are negative. The independent n<=3 ground-block value differs
by only 4.1e-10 eV in its ionization shift. This term slightly increases
the existing negative theory-minus-NIST residual; it cannot explain or
remove the 0.05 eV finite-space gap. Numerical agreement between symmetry
runs does not establish convergence of K with active space.

Data: `output/nist_spectra/he_specific_recoil.json`. These shifts remain
separate diagnostics until relativistic two-body recoil is validated.

## Relativistic SMS2/SMS3 extension

The diagnostic now also accepts Z to include the GRASP point-Coulomb
Breit two-body operator. `relativistic_vinti_matrix` forms the VINT2
matrix from the Vinti momentum term and both relativistic radial terms.
The spin tensor uses a reduced 6j expression, independently tested against
explicit Clebsch–Gordan spinor coupling for 36 kappa pairs. A separate
analytic s1/2–p1/2 check verifies that SMS2+SMS3 cancels the Q_s P_p
contribution and leaves -4Z/c times the integral P_s Q_p/r, including sign.
The zero-small-component limit reproduces the leading momentum operator.

For n<=3 helium, K changes from 0.1634250462 to 0.1634084401 au. The
ionization shift becomes -0.000609595239 eV, a +0.061949 micro-eV change
from the leading term. Independent ground-block orbitals reproduce the
Breit recoil shift within 4.1e-10 eV. Radial VINT2 antisymmetry agrees
within 9e-16 on these saved B-spline states.

This extension supersedes the earlier statement that SMS2/SMS3 is absent
from the diagnostic. It remains absent from production benchmarks and
has not yet been validated against an independent compiled GRASP energy
or across heavy ions. Finite-size, radiative, and higher-order recoil
remain omitted. The source port and analytic checks do not justify an
experimental precision claim or release of the full-table accuracy gate.

## DHF configuration-average implementation

`configuration_average_recoil` contracts the rank-one exchange term with
jj pair occupations. The direct term vanishes by parity. Explicit jj
ensembles use their pair moments, avoiding the incorrect product of mean
occupations. A test ensemble containing 1s² or 2p1/2² has zero cross-shell
recoil even though its mean occupations would spuriously imply a nonzero
exchange term. Independent weighted traces over resolved 1s2p CI levels
reproduce both leading and relativistic configuration averages.

The reduced spin tensors also pass a direct transcription of GRASP
SIGMA_1/2 phase conventions using independent Racah coefficients. Together
with the analytic radial and spin-coupling tests, the recoil test module
now has 45 passing cases. A compiled GRASP end-to-end comparison remains
outstanding.

`benchmark_first20_specific_recoil.py` evaluates the two-body correction
on fresh shooting-seeded, B-spline-refined DHF states with static DBSR
Breit, model QED and one-body recoil. It repeats each neutral and first ion
on the denser inner grid and records the correction separately in
`first20_specific_recoil.jsonl`. These are finite-basis mean-field
comparisons; completing them is not evidence for missing correlation or
radiative recoil being negligible at experimental precision.

All 20 pairs completed. Largest IE correction: 0.00068200225957 eV (Ne I); largest grid change in that correction: 1.47587651822e-09 eV (Ar I). RMS residual over 19 matched evaluated references: 0.979522805 eV. The remaining error is far larger than this recoil update.

## Correction-stack integration

Two-electron Breit recoil is now selectable through
`CorrectionStack(nuclear_mass_me=M, include_sms=True)` and
`DiracFock(nuclear_mass_me=M, include_sms=True)`. It has its own `sms`
energy component and `_sms_provenance`; `nms` remains the one-body term.
The flag defaults to false, so historical calculations are not silently
reinterpreted. Database generation has not yet been switched to this flag.

The DHF adapter uses the explicit jj ensemble's pair occupations. The
MCDHF solver evaluates each CI state, then reorders the SMS shifts with
energies and amplitudes. The result adapter evaluates the degenerate
ground-state ensemble. Unsupported per-block radial sets raise rather
than silently using unrelated orbitals. Missing nuclear mass is rejected.

End-to-end Ne DHF and two-electron Ne 1s2p MCDHF controls compare applied
shifts with direct operator evaluations. Independent syntax, analytic,
angular, configuration-mixture and recoil regression checks also pass.
Finite-size/radiative recoil and correction-matrix diagonalization within
accidentally degenerate manifolds remain absent.

## Fresh database physics version

The generator now selects static `dbsr_hf` Breit and `include_sms=True`
for every accepted DHF solve, including its retry/seed routes. The shared
shooting-seed helper keeps its explicit diagnostic defaults; generation
passes the updated model explicitly. Fresh rows store `sms_ha` and the
full correction-component JSON alongside `nms_ha`.

`physics_version=dbsr_static_breit_model_qed_grasp_nms_sms_v1` is required
in addition to the one-body operator version before a worker may extend
a ledger containing completed rows. Missing or mismatched versions cause
rejection without rewriting historical energies. New generation still
requires a separate output database; the old full-table hold remains.
No full-table calculation has been resumed by this change.
