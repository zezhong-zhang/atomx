# Balanced-family fixed005-orbital RCI discriminator

Complete allowed 4f/5p/5s pair families were evaluated with Atomx's native public
Dirac-Coulomb and complete static-Breit RCI consumers on the **same 24 converged
native005 MCDHF radial spinors**. The original five-state MCDHF stationarity
pass is preserved. These enlarged fixed-orbital RCI spaces have **not** undergone
MCDHF orbital reoptimization or an enlarged-space stationarity check.

The earlier selective 68-CSF test left the ground essentially unchanged. Here
complete 4f-family mixing lowers the ground by about 815 cm⁻¹, and the subsequent
5p family changes transitions by 304–808 cm⁻¹. Thus omitted same-pool physical
channels mattered before any larger radial cutoff. The final 531-CSF model still
misses nominal NIST transitions by 1303–2129 cm⁻¹. Its numerical and state-binding
gates pass; it is a physical diagnostic, **not a production spectrum or complete
core-valence calculation**.

## Prospective model and input identity

The [contract](contract.json) was written before execution. The
[frontier](../balanced_cv_frontier/README.md) had independently verified all
21 nonrelativistic configurations, 531 complete jj-CSFs, and resource bounds.
No model was reduced to fit the budget.

| Stage | Complete retained families | CSFs | Complete J/parity blocks | Largest raw M=J block |
| --- | --- | ---: | ---: | ---: |
| `stage_4f` | bare + 4f | 294 | 16 | 136 |
| `stage_4f5p` | bare + 4f + 5p | 468 | 16 | 209 |
| `stage_4f5p5s` | bare + 4f + 5p + 5s | 531 | 16 | 232 |

Each core-hole family contains all six unordered pairs from the same 6s/5d/6p
pool: 6s², 6s5d, 6s6p, 5d², 5d6p, 6p². Generating each family from each bare
reference retains parent provenance while deduplicating identical configurations.
Family sizes are bare 5, 4f 289, 5p 174, 5s 63 CSFs. “Balanced-family” describes
this common generation rule; it does not imply complete CV, core-core,
higher-rank correlation, or radial convergence.

All runs retain 69 electrons, Z=70, ns=123, ks=9, original 24 radial spinors,
all roots in every allowed J/parity block, and the original raw-determinant cap
2000. The Fermi nuclear profile uses the original default A=175 empirical rms
radius 5.246119777900239 fm. This is not a measured isotope-specific charge-radius
or mass claim. No QED, recoil, dynamical-Breit, empirical energy fitting, altered
EOL weights, or other software backend was introduced.

The source is the [native005 frozen source](../native_run/nuclear_eol005_projected_increment/source_identity.json),
all 353 Python files, map SHA
`9dff33cc24acecff9e821c4f9b836e02efca0d615291d6f92ff5a7014a726c3f`.
The shared mutable source was not edited. Original orbitals/workspace SHA is
`1e9195125c871492ec618872f70361d8d34016722ba942b71be90196d291bc30`.
Original MCDHF residual was 5.143009363125294e−8 in the original metric with
the original 1e−7 stationarity gate. [Inputs and all hashes](contract.json),
[full basis/coupling-tree identity](basis_identity.json), and
[exact stage preflight](preflight.json) are retained.

## Measured same-state transitions

All entries are cm⁻¹ above the **tracked 6s state in the same Hamiltonian and
space**. Independently recorded actual finite-H ground remains that 6s state at
every stage and for both operators. The full signed eigenvectors are retained.

| Model | D3/2 | D5/2 | P1/2 | P3/2 |
| --- | ---: | ---: | ---: | ---: |
| 4f DC | 24285.870 | 24863.044 | 24771.050 | 27833.828 |
| 4f static Breit | 24323.859 | 24859.022 | 24714.436 | 27752.894 |
| 4f+5p DC | 25058.434 | 25685.127 | 25052.413 | 28302.898 |
| 4f+5p static Breit | 25077.914 | 25666.913 | 25018.721 | 28246.682 |
| 4f+5p+5s DC | 25032.615 | 25661.484 | 25066.236 | 28316.786 |
| 4f+5p+5s static Breit | 25043.732 | 25635.584 | 25034.531 | 28263.092 |
| Original bound NIST nominal | 22960.800 | 24332.690 | 27061.820 | 30392.230 |
| Final static gap, theory−nominal | +2082.932 | +1302.894 | −2027.289 | −2129.138 |
| Final relative error | +9.072% | +5.355% | −7.491% | −7.006% |

The unchanged 1% nominal comparisons fail for all four final transitions.
NIST values here are bound to original005 inputs; they are not a new database
fetch or a complete isotope/environment-resolved experimental uncertainty model.
The final static-minus-DC transition increments are +11.116/−25.900/−31.705/
−53.694 cm⁻¹. Those are measured same-space operator changes, much smaller
than the remaining gaps. No inference that uncomputed QED is universally small,
or that a different historical QED budget transfers to these states, follows.

## Lower and upper shifts are separate

These static-Breit **absolute state-energy increments in Ha** compare each
stage with its immediate predecessor (the first predecessor is bare005).

| Added family | 6s | D3/2 | D5/2 | P1/2 | P3/2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4f | −0.003713980137 | −0.001132504336 | −0.000827650956 | −0.004940377206 | −0.003924413824 |
| 5p | −0.004843168164 | −0.001407443671 | −0.001162146862 | −0.003456741071 | −0.002593303458 |
| 5s | −0.000130013208 | −0.000285758015 | −0.000272758412 | −0.000057977684 | −0.000055242446 |

Differencing each upper increment against the **same-stage lower** produces:

| Added family | D3/2 transition | D5/2 transition | P1/2 transition | P3/2 transition |
| --- | ---: | ---: | ---: | ---: |
| 4f, against bare005 | +566.568 | +633.476 | −269.163 | −46.185 |
| 5p, against complete 4f | +754.054 | +807.891 | +304.286 | +493.788 |
| 5s, against complete 4f+5p | −34.182 | −31.329 | +15.810 | +16.410 |
| Full 4f+5p+5s, against bare005 | +1286.441 | +1410.038 | +50.933 | +464.014 |

These increments are conditional on preceding families and orbital identity;
they are not independent additive error bars. The complete-family 4f result
differs qualitatively from the earlier [68-CSF selective-family diagnostic](../cv_discriminator/README.md),
whose ground did not acquire these channels. The next 5p contribution is larger
than the tested 5s contribution on this pool, although both lower all absolute
target energies. D excitation gaps worsen while P gaps partly improve; adding
physically allowed channels is not guaranteed to monotonically improve
experimental errors.

| Static multiplet | 4f centroid / splitting | 4f+5p centroid / splitting | 4f+5p+5s centroid / splitting |
| --- | ---: | ---: | ---: |
| D, weights 4:6 | 24644.957 / 535.163 | 25431.313 / 588.999 | 25398.843 / 591.853 |
| P, weights 2:4 | 26740.074 / 3038.458 | 27170.695 / 3227.961 | 27186.905 / 3228.561 |

The final D splitting remains 780.037 cm⁻¹ below nominal 1371.89; P splitting
is 101.849 cm⁻¹ below nominal 3330.41. Centroid and splitting respond differently,
so a better P splitting does not certify the P centroid.

## Full state and numerical checks

The primary state tracker performs complete signed-overlap assignment in each
exact symmetry, starting with saved native005 DC states. Complete DC-to-static
assignment then binds static targets. Independent adjacent-static assignment,
bare-reference maximum overlap, full spectral distributions, and nearest
same-J/parity gaps are checked. All five target bindings remain unambiguous.

The D3/2 and P1/2 global energy order actually changes after adding 5p: static
P1/2 is global column 1 and D3/2 column 2. State identity therefore cannot be
transferred by an original global column number. The target roots remain local
rank 0 in their respective symmetries, but this is an observed result, not the
binding prescription.

Final static bare weights are 99.0275% (6s), 99.6235% (D3/2), 99.7166% (D5/2),
97.9764% (P1/2), 98.3087% (P3/2). Ground weights are bare 99.0275%, 4f 0.6089%,
5p 0.3582%, 5s 0.0054%. No same-symmetry near-degenerate clusters lie within the
predeclared 1e−8 Ha window; the closest target same-symmetry gap over stages is
0.12058894447000057 Ha. The full all-root CI character and nearest-gap audit,
including unselected high roots, are preserved in per-stage JSON/NPZ files and
[degeneracy audit](degeneracy_audit.json).

All **2706 execution checks** and **21707 independent replay/archive checks**
pass. Independent verification replays complete saved matrices with SciPy
`eigvalsh`, every exact-J block, Cauchy interlacing of successive principal
spaces, signed full-matrix reconstruction, left/right closure, unchanged
principal Hamiltonians, complete root/family character, symmetry-resolved
assignment, full spectral moments, and lower/upper/transition/centroid/splitting
ledgers. This independently checks the numerical solution of Atomx's saved H;
it does not independently certify the Atomx physical Hamiltonian or replace FAC/
GRASP/NIST comparisons. The source and physical inputs are hash-verified before
execution and during replay.

Observed execution was 75.334 s, peak RSS 1246.133 MiB, CPU one thread, within
4 GiB and 900 s. Replay observed 0.47 s. Initial machine load was about 10.7 on
32 CPUs. Times include cold/warm shared caches and archival matrix construction;
they are **not controlled software comparisons, scaling efficiencies or GPU
speedups**. No GPU was used. The small same-pool family test resolved an actual
physical distinction before a larger n/l scan.

## Files and next physically discriminating step

The [result](result.json) has `stages[].tracked_levels.static_Breit`, absolute
and transition increments, all-root provenance, family/scope identity, and
`card_summary`. The [verification](verification.json) binds its exact SHA and
all scientific evidence. [Presentation](presentation.json) supplies a source-bound
Chinese `card_summary_text` without rewriting the raw scientific result.
[Spectrum CSV](spectral_summary.csv) and [energy-increment CSV](energy_increments.csv)
contain both operators and all stages. Each stage directory preserves both full
matrices, all signed eigenvectors, all eigenvalues, all-root character, complete
adjacent/operator overlaps, and target bindings.

The supported next direction is genuine Atomx MCDHF optimization in the
declared balanced family space with the original EOL weights/targets and
stationarity gates, followed by same-orbital RCI and uncertainty-aware state
comparison. The measured 4f and 5p differential shifts make those families a
priority; the 5s increment supplies a useful smaller conditional scale. These
results do not certify new orbital stationarity, select a reduced physical
model without a new contract, or show that enlarged radial/correlation
truncations must eliminate the gap. Further core-core/virtual/triple channels,
orbital response and matched correction coverage still require separate
bounded physical experiments. No additional optimization run was started here.

Commands, from the repository root:

```bash
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/balanced_cv_replay/verify.py
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/balanced_cv_replay/make_report.py
```

`run.py` refuses to overwrite existing completed or failed execution evidence.
A successor calculation needs a new owner directory and prospective contract.

Documentation impact: this increment changes no shared source or public API.
The earlier 68-CSF archive remains unchanged and has its distinct selective-family
scope. Current interpretation, replay and input links live here; the parent
campaign incorporates this source-bound successor rather than silently rewriting
historical results. The durable inference is that complete same-pool physical
families and tracked lower-state response must precede a radial-cutoff diagnosis.
