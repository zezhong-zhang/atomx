# Saved high-l block diagnosis

This is one predeclared, read-only saved-matrix experiment. It imports no atomx
module, builds no atomic operator, and changes no source, H/S, root, gate, or
historical failure. The two input archives are the unchanged failed outputs of
[002](../yb_high_l002/RESULTS.md) and [003](../yb_high_l003/RESULTS.md). Only their
already calculated kappa=4 channel is available; kappa=-5 was not reached.

The complete original electron span has 119 columns in 239 boundary coordinates,
with all 67 original roots below 50 Ha retained. The unused upper raw-H storage
is excluded exactly as in the native solver: its lower triangle defines H.
The contract was saved before array arithmetic. One launch performed exactly two
complete block Ritz solves, using ordinary and extended-precision projected
contractions. It used 76.2 MiB peak RSS and less than one second; this is not a
controlled performance measurement.

| Saved/candidate route | Whole-electron Gram error | Complete-band ordinary S-dual residual (Ha) |
| --- | ---: | ---: |
| Original 002 | 5.66e-15 | 1.8192200e-7 |
| Independent selected-root refinement 003 | 9.7054074e-10 | 1.4565980e-7 |
| Full-span block Ritz, ordinary projection | 9.62e-15 | 1.4565971e-7 |
| Full-span block Ritz, wide projection | 1.20e-14 | 1.4565969e-7 |

All routes fail the unchanged 1e-8 Ha full-band residual gate. Both block Ritz
routes pass the original Gram, projector, column-overlap, and root-shift controls.
Using wide Rayleigh energies also fails the residual gate. Original ordering and
all 67 band members were preserved; the other 52 electron roots were retained.

At original column 45 (2.9405921101964845 Ha), the repaired block residual is
1.45659694e-7 Ha outside the original electron span and about 1.5e-11 Ha inside
it. The saved independent refinement reaches an in-span residual below 4.1e-13
Ha over the entire band, while its out-of-span maximum remains 1.4565972e-7 Ha.
Repeating only the contractions in extended precision leaves that floor. The
003 Gram failure is off-diagonal: individual norm errors remain 2.45e-15.
These observations distinguish incorrect joint orthogonality from the remaining
full-space eigenpair residual. They do not establish a certified lower bound for
every possible arbitrary rotation of this span.

S has measured 2-norm condition number 3.341263364e8, and the retained electron
energies extend to 6.429124038e8 Ha. Ordinary projected-H contraction asymmetry is
3.07849e-7 Ha, versus 7.50732e-11 Ha with wide contraction. This large spectral
range explains why an otherwise accurate full FP64 block eigensolve need not
deliver the requested absolute accuracy for low roots. It does not diagnose an
omitted physical Hamiltonian term. `dirac_assembly._refine_eigenpair` minimizes
an FP64 Euclidean projected residual, not this full S-dual residual; 003 also
refined only 67 of the jointly retained 119 columns.

The smallest sound shared-kernel direction is residual correction in the full
239 active-coordinate generalized eigenproblem, followed by joint metric
orthogonalization and retained-root Ritz/identity checks. The correction space
must include residual directions outside the currently approximate 119-column
electron projector, while retaining the original physical branch and complete
band. A block correction satisfies both the eigen-equation residual and the
joint S-Gram constraint; per-column updates require a joint repair afterward.
Use the full original H/S and evaluate acceptance in the original full S-dual
norm. The broad spectral range warrants residual-controlled correction or
precision-aware block treatment; a global FP64 diagonalization and energy
stability alone are insufficient. This is a recommendation, not a tested kernel.

The diagnostic establishes neither current frozen-core DF stationarity nor g
spectral response, correlation convergence, experimental accuracy, or source
qualification. Original failures remain untouched. Detailed per-root arrays,
source/input identities, and checks are in `result.json`, `diagnostic_arrays.npz`,
and `receipt.json`. No new general insight entry is proposed; this local evidence
supports the existing rule to validate residuals in the actual native metric.
