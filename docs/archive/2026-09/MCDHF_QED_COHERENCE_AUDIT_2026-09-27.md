# MCDHF QED coherence audit — 2026-09-27

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Corrections and scope

Local Uehling and Wichmann–Kroll potentials now use the full same-kappa
radial matrix and the correlated one-particle density matrix. Orbital
occupations alone discard cross terms after correlation-orbital rotations.
The solver and correction adapters use the coherent expectation, including
block-specific radial sets. GRASP `src/appl/rci90/vpintf.f90` likewise
integrates cross-orbital products PF(IA)*PF(IB)+QF(IA)*QF(IB).

A separate MCDHF dispatch bug concatenated n and positive kappa: n=2,
kappa=1 became "21", parsed as principal n=21 by the self-energy backend.
Canonical subshell labels fix this. Existing DHF dispatch already used
canonical labels; this bug does not explain its correlation deficit.
Historical MCDHF cached predictions require revalidation.

## Recomputed helium control

SD-MCDHF orbitals are optimized for Dirac–Coulomb, followed by CI Gaunt,
coherent local VP, diagonal model self-energy and one-body Breit recoil.
The He-4 nuclear mass is 7294.299387111486 electron masses.

| Active space | CSFs | Ionization (eV) | Theory minus NIST (eV) | VP coherence change (micro-eV) |
|---|---:|---:|---:|---:|
| n <= 2 | 17 | 24.42302539782434 | -0.1643636131756594 | -8.6760 |
| n <= 3 | 98 | 24.536514047322903 | -0.0508749636770958 | -15.2888 |

Reference: archived evaluated He I ionization, 24.587389011 eV. Both
orbital optimizations pass their stationarity gates. Increasing the space
changes the result by 0.113489 eV; active-space convergence is not established.
The VP fix is much smaller than the remaining discrepancy.

Data: `output/nist_spectra/he_combined_coherent_vp.jsonl`. Numeric orbital
snapshots are saved alongside it with SHA256 recorded in each row. The
older `he_combined_corrections.jsonl` remains a historical diagnostic.

## Across-table self-energy dispatch controls

Same-orbital hydrogenic 2p1/2 checks isolate the label correction:

| Z | Backend | Corrected minus historical model SE (eV) |
|---:|---|---:|
| 2 | Mohr | -0.000000781822312 |
| 20 | Mohr | -0.005651009643 |
| 36 | Mohr | -0.026079431767 |
| 80 | Indelicato | +4.400690061739 |

These validate dispatch, not experimental accuracy or a full transition.
Data: `output/nist_spectra/mcdhf_self_energy_label_audit.json`.

## Remaining physical limits

Self-energy is still a diagonal orbital model. Full static Breit
retardation in CI, two-body recoil, finite-size and radiative recoil,
active-space/radial extrapolation and degenerate-subspace correction
diagonalization remain outstanding. Local VP and recoil are first-order
expectations. No full-table accuracy promotion follows from these fixes.

Regression coverage includes orbital-rotation invariance, kappa selection,
block-specific radial sets, coherent result adapters, principal-number
dispatch including two-digit n, and end-to-end positive-kappa SE shifts.

## Saved-state representation audit

`scripts/audit_he_qed_radial_rotation.py` loads and hash-verifies the numeric
snapshots, reconstructs the full ground-state 1RDM, and rotates the 1s/2s
radial basis together with that density matrix. Thus the physical density
is unchanged. The unrotated self-energy reproduces the recorded result.

At five angles from 0 to 1.1 radians, coherent VP is invariant within
2.4e-20 eV. The diagonal Mohr SE varies over 0.000649990 eV (n<=2) and
0.000683590 eV (n<=3). These sampled spans are representation diagnostics,
not physical uncertainty bounds: assigning hydrogenic principal-number
labels to arbitrary rotated correlation orbitals is outside the model's
spectroscopic interpretation. The result demonstrates why a full operator
or a validated orbital-mapping prescription is needed for precision SE.
It does not explain the 0.050875 eV residual or establish its entire cause.
Data: `output/nist_spectra/he_qed_radial_rotation.json`.

## Coherent local self-energy path

The optional `shabaev_local` backend now supplies its full same-kappa
radial matrix using its existing potential, amplitude calibration and
quadrature. MCDHF and the correction adapter contract this matrix with
the full correlated density. Other self-energy backends retain their
explicit diagonal approximation; the default backend is unchanged.
This is the local part only, not the full nonlocal Shabaev operator.

The snapshot audit now tests this SE matrix under the same five orbital
rotations. It is invariant within floating-point precision. Tests also
compare its diagonal with the original single-orbital backend and check
the independent correlated-density adapter integral. This fixes operator
contraction consistency without establishing experimental model accuracy.
The stored helium ionization benchmarks still use Mohr SE; the optional
local-operator diagnostic does not replace their physics provenance.

## Independent symmetry-space control and extension

`benchmark_he_combined_corrections.py --ground-block --max-n 4` restricts
CSFs to the complete J=0 even sector of each SD space. This is an exact
angular symmetry restriction for the helium ground state, not a removal
of configurations in that sector. Records and snapshots are separate
from the original all-symmetry calculation.

The n<=2 control uses 5 rather than 17 CSFs. Its DC energy agrees with the
original within 6.4e-12 Ha. The diagonal Mohr result differs by 1.063e-5 eV,
while replacing SE on both neutral and ion by the coherent local operator
reduces the difference to 5.2e-10 eV. This independently confirms the
importance of operator coherence on separately optimized radial sets.

`audit_he_symmetry_convergence.py` recomputes both endpoints for the local
model on the saved states, keeping other corrections fixed. The original
n<=3 local-model ionization is 24.5367861890 eV, still 0.0506028220 eV below
NIST. Changing this SE model does not close the correlation-space gap.
Larger ground-block runs are pending; read the JSONL for completed rows.

The n<=3 symmetry control has also completed: 14 CSFs instead of 98,
290.7 s versus 355.4 s for the prior all-symmetry run. DC energies agree
within 6.98e-11 Ha. Coherent-local-SE ionizations agree within 6.15e-9 eV,
whereas diagonal Mohr ionizations differ by 2.92e-5 eV. The ground-block
local-model ionization is 24.5367861952 eV, residual -0.0506028158 eV.
Both controls support the symmetry restriction; they do not establish
radial or active-space convergence. The n<=4 calculation is in progress.

## n<=4 false plateau and finite-orbital stability failure

The 30-CSF J=0 even n<=4 run finished after 493 iterations (1308.4 s).
Its DC energy is -2.9019746292548843 Ha, essentially the n<=3 value.
Seven of its sixteen natural occupations are below 2e-10: adding nominal
orbitals did not materially enlarge the occupied correlated space.

`audit_he_virtual_stability.py` reproduces the saved DC energy, constructs
compact f-orbital trial functions in the same nuclear positive-energy
subspace, then evaluates CI with the original Hamiltonian. A potential
V_nuc + a*r² is used only to generate the trial functions, never to score
energies. At a=0.01,0.1,1, the energy drops by 0.000003561, 0.000221687,
and 0.003675726 eV. The f-orbital norms remain unity to floating precision.

Thus the gradient-based convergence flag does not establish active-space
stability. The n<=4 plateau must not be used as an extrapolation or an
accuracy bound. The best finite trial and its CI amplitudes are retained
in `he_virtual_stability_restart.npz`; the source hashes and probe results
are in `he_virtual_stability.json`. The next optimization must populate
and optimize the weakly occupied virtual directions, then repeat this
finite-orbital check before claiming correlation-space convergence.

## Weak-virtual population remedy and checkpointed refinement

`mcdhf.virtual_stability.populate_weak_virtuals` searches near-empty
natural-density directions. Each natural-orbital rotation is explicitly
checked by a new CI calculation for energy equivalence; non-equivalent
rotations in truncated spaces are skipped. Compact candidates are projected
into the same nuclear positive-energy space, orthogonalized against all
retained same-kappa orbitals, normalized, and accepted only if they lower
the original CI energy. No NIST value enters this algorithm.

For the saved n<=4 helium state, 23 of 28 trials lower the energy, from
-2.9019746292548843 to -2.9028764547370987 Ha (0.024540 eV gain). This is
an improved initial state, not a converged spectrum. The restart and full
trial history are in `he_populated_virtuals.json` and the referenced NPZ.

`optimize_he_populated_virtuals.py` refines this restart with the unchanged
Dirac-Coulomb functional. The preconditioner occupation floor is exposed
as a numerical parameter (default 0.05 unchanged); this diagnostic uses
1e-4. It also uses the new optional optimizer progress callback to save
radial/CI checkpoints every 50 iterations and the final state even if the
iteration limit is reached. Energy and residual traces are written every
10 iterations. Convergence, virtual stability and physical corrections
must be reassessed on the resulting state.

A real two-electron f-space test verifies energy lowering under the
original Hamiltonian, unit orbital norms, retention of the nuclear
positive-energy projector, and checkpoint values consistent with the CI
energies and final optimizer iteration.
