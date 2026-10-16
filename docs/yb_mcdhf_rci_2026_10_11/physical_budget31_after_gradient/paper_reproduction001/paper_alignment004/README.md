# Literature alignment 004: fixed-orbital complete 5p core

Type: completed experiment/current input-alignment boundary; no production
source or default changes. The owning
contract is [contract.json](contract.json), executed by [align_core.py](align_core.py).
The immutable predecessor is [complete6p](../native_virtual_p003/README.md).

The user requested model alignment after comparing Yb+ transition energies
across programs. The primary target remains Sahoo's RCCSDT paper; AMBiT/GRASP
papers are separately identified comparison recipes. A program name or similar
experimental deviation does not identify a common Hamiltonian/model.

## Source-grounded alignment matrix

| Item | Current native model / this step | Published target / remaining difference |
|---|---|---|
| Species and observable | 69-electron Yb+, 6s S1/2 to 5d D3/2,D5/2, cm^-1 | Same electronic transitions; do not use neutral-Yb rows |
| Energy convention | Three attachment energies relative to one 68-electron closed core; both lines from their differences | Same EA arithmetic in Sahoo Table1; never compare a total69e energy with an EA |
| Nuclear input | Existing empirical A171 Fermi distribution, unchanged | Sahoo's actual radius/density input not published in the checked main paper; Hur GRASP uses172, so isotope inputs are not identical |
| Radial representation | ns180/order9 B-splines, R500, restricted kinetic balance | Sahoo:40GTO per relativistic symmetry, R500, kinetic balance; exponent lists unavailable. Equal radial extent is not equal radial basis |
| Correlated core | Previous18:5s2,5p1/2^2,4f14; this step adds complete5p3/2^4, giving22active/46frozen | Sahoo's actual frozen/core partition is unspecified;22 is our declared controlled closure, not a recovered paper input |
| Virtual angular/radial coverage | Same single6s/5d/6p layers,18magnetic spinors | Sahoo SDT throughg, h/i atSD; Hur AMBiT8spdf; Kawasaki AMBiT16spdf. All exceed this native virtual subset |
| Correlation equations | Complete finite EA-CCSDT/R3 tensor equations | Sahoo coreT plus valenceSv FS-RCCSDT ansatz; strict method/input equivalence not established by small-model agreement |
| Hamiltonian corrections | DC only, unchanged | Sahoo reports h/i and self-consistent Breit contributions separately. They must be recomputed in their own matched models, never added from the paper |
| Validation | Full retained equations and magnetic partners, physical input hashes | Numerical convergence is separate from radial/core/virtual/experimental convergence and physical exp(T) normalization |

Primary sources, checked2026-10-15:

- [Sahoo arXiv2504.01438v1, Table1 and Eqs1-3](https://arxiv.org/html/2504.01438v1).
- [Hur arXiv2201.03578v2, Supplemental SecIV and TableS7](https://arxiv.org/html/2201.03578v2).
- [Kawasaki arXiv2402.13541v2, SecIII and TableIII](https://arxiv.org/html/2402.13541v2).
- [NIST YbII electronic levels](https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable6_a.htm):
  D3=22960.80, D5=24332.69cm^-1, splitting1371.89cm^-1.

## Predeclared discriminator and resource admission

Hypothesis: opening5p3/2 changes differential correlation at fixed orbitals,
Hamiltonian, virtuals and equation ranks. Before any new solve, embed every old
amplitude by its particle/hole identities and verify the old signed reference
phases, Fock/ERI entries, common68e reference energy, all689589 predecessor
SDT equations and all127620 predecessor R3 equations for each of S,D3,D5.
New5p3/2 equations may be nonzero; that is the measured new channel.

Accept the new spectrum only if all1292379 core SDT equations and all191880
R3 equations for all12 target partners pass1e-10Ha, with J2/J+ and norm
gates unchanged. Experimental improvement is not an acceptance gate.
The first failed source, identity, resource or equation gate stops; preserve
partials and failures without changing ranks, orbitals, root selection or
tolerances.

The conservative22core/18virtual payload is12,583,357,392bytes. Local admitted
workspace increases explicitly from8192 to16384MiB; the24GiB process address
cap remains unchanged. One CPU27, one BLAS thread, load1<=8 and available
memory>=32GiB apply. Probe wall cap360s; solve wall cap2400s,60 core residual
calls and48 Davidson vectors per target. These are admission bounds, not
measured memory or a performance gain.

The next complete virtual f pair would give22core/32virtual and conservative
payload70,672,663,744bytes. It is rejected under this contract without
allocation. A scalable representation and separate physics/resource contract
are required before that expansion. Missing f/g coverage remains a physical
gap after core closure, with no assumed sign or transferable correction.

## Measured result

[Probe](probe001/result.json) passes1171 identity/sign/off-shell controls.
The common68e HF origin agrees within5.457e-12Ha, old Fock entries within
2.843e-14Ha, and all old ERIs exactly. The embedded old core energy differs
by2.843e-14Ha; all predecessor SDT and three R3 vector equations recover their
old converged residuals. Newly opened core equations reach0.05579529127Ha,
so this is a nontrivial new channel rather than relabeling the old result.

[New solve](actual001/result.json) passes531 live controls. Sixteen complete
DIIS evaluations converge all1292379 SDT equations to5.88234e-11Ha; core J+
is9.0033e-15. The S,D3,D5 targets use16/19/18 Davidson actions. All12 magnetic
partners pass their full191880-dimensional R3 equations, J2 and canonical
norm checks; maximum partner residual9.39456e-11Ha. No tolerance or root
selection rule changed.

| Model | S to D3 /cm^-1 | S to D5 /cm^-1 | D5-D3 /cm^-1 |
|---|---:|---:|---:|
| Previous18core/18virtual | 25164.078643 | 25572.326411 | 408.247768 |
| New22core/18virtual | **25979.234799** | **26667.252046** | **688.017247** |
| New minus previous | +815.156156 | +1094.925635 | +279.769479 |
| NIST electronic levels | 22960.80 | 24332.69 | 1371.89 |
| New minus NIST | +3018.434799 | +2334.562046 | -683.872753 |
| New relative error | +13.1460% | +9.59434% | -49.8489% |

The old line errors were+9.59583%/+5.09453%. Thus complete5p core closure
increases both transition errors in this fixed s/p/d virtual subset, although
the fine splitting moves closer. Neither observation is an experimental
convergence certificate or a numerical correction transferable to another space.

| State | New EA /cm^-1 | EA shift /cm^-1 | Total69e relative-energy shift /cm^-1 |
|---|---:|---:|---:|
| 6s | -92793.404355 | -888.632038 | -5010.145726 |
| 5d3/2 | -66814.169556 | -73.475882 | -4194.989571 |
| 5d5/2 | -66126.152309 | +206.293597 | -3915.220092 |

The correlated68e core itself shifts by-4121.513688cm^-1. An EA shift is
therefore not a total69e energy shift: all three total energies decrease,
while D5's EA increases. The core shift cancels from each transition. The
[comparison driver](compare_alignment.py) and [comparison data](alignment_comparison.json)
preserve this arithmetic, all three EAs, both lines and the splitting separately.
Published RCCSDT/h-i/Breit increments are reference-only, never applied.

The [retained iteration12 arrays](iteration12_core.npz) and
[rank diagnosis](limiting_equation_snapshot.json) show that this iteration's
largest off-shell residual was a4f-to6p triple,7.15454e-9Ha. This is a numerical
iteration observation, not the size of a physical missing-triples correction.

## Verification, resources and remaining work

[Current tensor regression](tensor_tests.log), [JUnit](tensor_tests.xml):
12PASS/0SKIP in14.58s at unchanged production source. Fresh saved-equation
replay is recorded by [review_saved.py](review_saved.py), under its
[predeclared contract](review_contract.json). The [saved replay](saved_review.json)
passes52 controls in162.513s, freshly reapplying the complete core equations
and all12 magnetic partner equations, norms, J2 and separate energy ledgers.
This replay uses the same production action and is not an independent full22e
FCI oracle. Physical exp(T) normalization, omitted-rank/full-H defects and
left/dressed properties remain uncomputed for this larger model.

The main solve takes1803.307s and peaks at2206.238MiB RSS; the initial probe
takes78.594s and peaks at2385.363MiB. These are observed scopes, not controlled
performance comparisons. [Load samples](resource_samples.json) later exceed
the initial load1<=8 admission guard (sample9.935); no paired speedup or
controlled timing claim follows. The declared memory and wall caps are retained.
[Solve admission](solve_admission.json) preserves the original1255s action-cost
estimate separately from actual measurements.

The [execution record](execution_record.json), [source/import origins](module_origins.json)
and [documentation impact](documentation_impact.json) bind the local experiment.
[Closing reviewer](verify_increment.py) checks unchanged source and sealed
predecessors, the numerical receipts, test JUnit, current local links and reaped
task sessions. Link checks cover the present working tree, including local
ignored/untracked evidence, rather than a clean-checkout publication.
Original362-file production source map
`ea402a15130e5041dab452940c288804b300008b8c44dafb6e7a7d1c3f53c373`
and61 sealed predecessor artifacts remain unchanged. Existing MCDHF→RCI
main-route failed gates remain unchanged; this RCC comparison does not admit
that optimizer. Current guides link this continuation without rewriting old
evidence. No new global rule is introduced.

The next input gap is balanced virtual angular/radial coverage, beginning
with complete f partners in a separately admitted representation. A core-only
expansion has not repaired the spectrum. Full40GTO inputs, nuclear parameters,
core/virtual/triples cuts and strict FS-RCC method equivalence are still open.
No full-paper replication or experimental-precision result is claimed.

Reproduction commands use the current .venv and the six thread variables
listed in the execution record. `align_core.py prepare`, then `probe`, then
`solve` require new empty output directories; they intentionally refuse to
overwrite retained evidence. `compare_alignment.py` and `review_saved.py`
consume the saved solution and do not rebuild SCF orbitals.
