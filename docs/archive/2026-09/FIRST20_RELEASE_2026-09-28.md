# First-20 review and corrected full-table release

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The consistent SQLite snapshot recorded in `corrected_full_table_release.json`
contains 3,481 completed H–Ca configuration jobs, no unfinished endpoints, and
all 657 core-hole stationarity revalidations complete. Record consistency
checks passed. The retained NIST source audit passed with its documented
coverage limitations. `release_corrected_full_table.py` verifies these gates
before changing the queue scope and preserves previous metadata and the
review in `full_table_release_history`.

This releases the corrected DHF **baseline**, not a claim of experimental
accuracy. The matched 68 ionizations have RMS residual 1.253012779 eV.
Configuration averages, incomplete manifolds, unknown assignments, and
inverted configuration order remain visibly distinguished from resolved
line predictions. Missing correlation and precision-QED terms remain model
limitations.

The extended configuration-notation recovery initially found another 126
line records and 27 jobs beyond H–Ca. A subsequent runtime audit found three
impossible tungsten configurations containing `4l`: the source uses an
unresolved angular-momentum label, whereas a literal l orbital has ell=8 and
requires n>=9. The parser now enforces ell<n. The three job records and
affected line records were preserved in `invalid_quantum_number_*_history`;
the impossible jobs were removed and their lines marked unsupported. Source
NIST records are unchanged. Seven focused parser/generator checks passed.

Seven ordinary workers and one recovered-configuration worker were launched
for Z=21–118. Their results use DBSR static Breit, model QED, operator NMS/SMS,
the applicable one-electron mass remainder, local stationary core-hole
rotations, targeted diffuse-state eigenvector refinement, and corrected DIIS
sweep residuals. Further failures are audited as the table is generated.

## 3s-hole correlation diagnostic

The largest first-20 excitation discrepancies include Cl-like
3s3p6 configurations. [Froese Fischer et al. (2006), section 8](https://doi.org/10.1016/j.adt.2006.03.001)
describe their interaction with 3s2 3p4(1D)3d 2S. This is a physical reason
to test configuration mixing rather than assume further grid refinement
will remove the full discrepancy.

`probe_cl_like_s_hole_mixing.py` uses only those two configurations with the
Dirac–Coulomb Hamiltonian. Ar II and Ca IV converge to the target J=1/2,
even-parity symmetry, with energy lowering of 51,166.63 and 59,354.25 cm^-1,
respectively. The K III calculation terminates with a failed line search and
is explicitly unvalidated; its energy is not used as a correction.

These results are displayed in the interactive comparison as diagnostics.
The ground endpoints remain uncorrelated, corrections are not recomputed,
and there is no active-space extrapolation. No diagnostic value replaces a
production DHF result or contributes a new accuracy score. Nine comparison
tests passed and the generated JavaScript passed `node --check`.

An independent DBSR_HF check is also being retained. Nonrelativistic p-shell
configurations must use `term=LS` to expand both relativistic partners;
`term=AV` with a literal p(6) gives an invalid over-occupied jj subshell and
is not a valid reference. The tight-tail runs use `end_tol=1e-11` because the
default tail cutoff prevents the requested 1e-8 orbital tolerance.

The retained `dbsr_s_hole_reference/comparison.json` now records converged
ground and 3s-hole DBSR calculations. Matching both Fermi nuclear shape
parameters gives Ar II excitation energies 150625.792997 cm^-1 (DBSR) and
150625.793848 cm^-1 (atomx), differing by 0.000851 cm^-1. Both external
orbital residuals are below 1e-8. This comparison isolates Dirac–Coulomb
physics; it excludes Breit, QED, and recoil. The much larger corrected
NIST residual therefore cannot plausibly be removed by this radial-solver
difference alone.

The comparison also exposes a provenance issue for subsequent work: atomx's
default empirical radius uses A=45 for Ar, whereas the recoil isotope is
A=40. The external check deliberately matches that same radius to isolate
the electronic calculation. These default nuclear-profile parameters should
be made explicit in the database and linked consistently to isotope choices;
the electronic agreement does not certify isotope-radius accuracy.
