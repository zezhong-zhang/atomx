# Five-state directional derivative result

**Closed scientific FAIL; all ten complete native points and ten state gradients finished.**
The unchanged ordinary-FD gate passes37/40 comparisons; three D3/2 line comparisons fail.
All20 absolute-state and4 literal EOL-callback comparisons pass;13/16 line comparisons pass.
The failed cases remain failed. No retry, step scan, model reduction or tolerance change followed.

The direction is **−checkpoint13.full_normalized_pg**, A’s complete normalized preconditioned tangent,
in the original3689-coordinate P/polar manifold. Steps were fixed at0.002 and0.001 before energies.
The model remains native frozen f043, literal13, A91NR/1595CSF andB100NR/2574CSF, all31radials,
Z70/A175 Fermi nucleus, five original signed physical UIDs and equal0.2 EOL weights.
Every point has its own complete retained H/CI owner and original531-reference tracking.

Derivatives below have units Ha per scalar displacement along this fixed direction.
They are numerical directional comparisons, **not spectral-energy error bounds**.
Only the weighted EOL is the original stationarity target; nonzero individual-state gradients
in the shared orbital bank are expected. This result does not certify full-space stationarity,
old StageB FD, the original optimizer, missing-correlation completeness or agreement with NIST.

The complete unedited evidence is [actual/result.json](actual/result.json), with the exact
[driver](run.py), [contract](contract.json) and [root execution seal](root_execution_seal.json).

## All forty ordinary-FD outcomes

Both steps retain `max(1e−8,1e−5*abs(analytic))`. EOL uses the actual binary64
`float(weights @ energies)` callback at each endpoint; it is not replaced by the wide sum
of individual-state differences. Lines subtract the same lower6s state.

| Model | h | Observable | Analytic | Ordinary central FD | Abs error | Gate | Result |
|---|---:|---|---:|---:|---:|---:|---|
| A | 0.002 | state 6s_1/2 | -0.166246919514 | -0.16624681939 | 1.00124e-07 | 1.66247e-06 | PASS |
| A | 0.002 | state 5d_3/2 | -0.167133419764 | -0.167133306604 | 1.13160e-07 | 1.67133e-06 | PASS |
| A | 0.002 | state 5d_5/2 | -0.168460411267 | -0.168460295299 | 1.15968e-07 | 1.68460e-06 | PASS |
| A | 0.002 | state 6p_1/2 | -0.169510704961 | -0.169510595697 | 1.09264e-07 | 1.69511e-06 | PASS |
| A | 0.002 | state 6p_3/2 | -0.169193791327 | -0.169193673628 | 1.17699e-07 | 1.69194e-06 | PASS |
| A | 0.002 | line 5d_3/2 | -0.000886500249661 | -0.000886487214302 | 1.30354e-08 | 1.00000e-08 | **FAIL** |
| A | 0.002 | line 5d_5/2 | -0.00221349175276 | -0.00221347590923 | 1.58435e-08 | 2.21349e-08 | PASS |
| A | 0.002 | line 6p_1/2 | -0.00326378544705 | -0.00326377630699 | 9.14006e-09 | 3.26379e-08 | PASS |
| A | 0.002 | line 6p_3/2 | -0.00294687181275 | -0.00294685423796 | 1.75748e-08 | 2.94687e-08 | PASS |
| A | 0.002 | EOL EOL | -0.168109049367 | -0.168108937942 | 1.11425e-07 | 1.68109e-06 | PASS |
| A | 0.001 | state 6s_1/2 | -0.166246919514 | -0.16624689215 | 2.73646e-08 | 1.66247e-06 | PASS |
| A | 0.001 | state 5d_3/2 | -0.167133419764 | -0.167133387549 | 3.22145e-08 | 1.67133e-06 | PASS |
| A | 0.001 | state 5d_5/2 | -0.168460411267 | -0.168460376699 | 3.45679e-08 | 1.68460e-06 | PASS |
| A | 0.001 | state 6p_1/2 | -0.169510704961 | -0.169510669366 | 3.55952e-08 | 1.69511e-06 | PASS |
| A | 0.001 | state 6p_3/2 | -0.169193791327 | -0.169193760485 | 3.08422e-08 | 1.69194e-06 | PASS |
| A | 0.001 | line 5d_3/2 | -0.000886500249661 | -0.000886495399754 | 4.84991e-09 | 1.00000e-08 | PASS |
| A | 0.001 | line 5d_5/2 | -0.00221349175276 | -0.00221348454943 | 7.20333e-09 | 2.21349e-08 | PASS |
| A | 0.001 | line 6p_1/2 | -0.00326378544705 | -0.00326377721649 | 8.23056e-09 | 3.26379e-08 | PASS |
| A | 0.001 | line 6p_3/2 | -0.00294687181275 | -0.00294686833513 | 3.47762e-09 | 2.94687e-08 | PASS |
| A | 0.001 | EOL EOL | -0.168109049367 | -0.168109017977 | 3.13893e-08 | 1.68109e-06 | PASS |
| B | 0.002 | state 6s_1/2 | -0.163876272208 | -0.163876177794 | 9.44140e-08 | 1.63876e-06 | PASS |
| B | 0.002 | state 5d_3/2 | -0.164635755719 | -0.164635645433 | 1.10286e-07 | 1.64636e-06 | PASS |
| B | 0.002 | state 5d_5/2 | -0.166011523776 | -0.166011412603 | 1.11173e-07 | 1.66012e-06 | PASS |
| B | 0.002 | state 6p_1/2 | -0.166950915688 | -0.16695081149 | 1.04198e-07 | 1.66951e-06 | PASS |
| B | 0.002 | state 6p_3/2 | -0.166510425515 | -0.166510320469 | 1.05046e-07 | 1.66510e-06 | PASS |
| B | 0.002 | line 5d_3/2 | -0.000759483510903 | -0.000759467639 | 1.58719e-08 | 1.00000e-08 | **FAIL** |
| B | 0.002 | line 5d_5/2 | -0.00213525156766 | -0.00213523480852 | 1.67591e-08 | 2.13525e-08 | PASS |
| B | 0.002 | line 6p_1/2 | -0.00307464348049 | -0.00307463369609 | 9.78441e-09 | 3.07464e-08 | PASS |
| B | 0.002 | line 6p_3/2 | -0.00263415330684 | -0.00263414267465 | 1.06322e-08 | 2.63415e-08 | PASS |
| B | 0.002 | EOL EOL | -0.165596978581 | -0.165596873558 | 1.05023e-07 | 1.65597e-06 | PASS |
| B | 0.001 | state 6s_1/2 | -0.163876272208 | -0.163876247825 | 2.43829e-08 | 1.63876e-06 | PASS |
| B | 0.001 | state 5d_3/2 | -0.164635755719 | -0.164635720466 | 3.52525e-08 | 1.64636e-06 | PASS |
| B | 0.001 | state 5d_5/2 | -0.166011523776 | -0.166011485817 | 3.79588e-08 | 1.66012e-06 | PASS |
| B | 0.001 | state 6p_1/2 | -0.166950915688 | -0.166950882885 | 3.28030e-08 | 1.66951e-06 | PASS |
| B | 0.001 | state 6p_3/2 | -0.166510425515 | -0.166510388226 | 3.72888e-08 | 1.66510e-06 | PASS |
| B | 0.001 | line 5d_3/2 | -0.000759483510903 | -0.000759472641221 | 1.08697e-08 | 1.00000e-08 | **FAIL** |
| B | 0.001 | line 5d_5/2 | -0.00213525156766 | -0.00213523799175 | 1.35759e-08 | 2.13525e-08 | PASS |
| B | 0.001 | line 6p_1/2 | -0.00307464348049 | -0.00307463506033 | 8.42016e-09 | 3.07464e-08 | PASS |
| B | 0.001 | line 6p_3/2 | -0.00263415330684 | -0.00263414040091 | 1.29059e-08 | 2.63415e-08 | PASS |
| B | 0.001 | EOL EOL | -0.165596978581 | -0.165596944498 | 3.40829e-08 | 1.65597e-06 | PASS |

The three failures are A D3/2 at0.002 (1.30353590977e−8), B D3/2 at0.002
(1.58719026533e−8), and B D3/2 at0.001 (1.08696817935e−8), each against1e−8.
Passing absolute-state tests use their larger relative tolerance and do not imply that
upper-minus-lower cancellation passes the stricter line gate. The two scales do not alone
uniquely separate finite-step truncation, endpoint assembly and reported-energy rounding.

## Ten complete gradients and aggregate closure

Each full signed `contract_bilinear(c,c)` is normalized by its exact binary64 Fraction
actual norm, then each coefficient is represented once. No1e−15 filter or amplitude cutoff
is used. All31×246 components are retained. Reference-metric norms below are diagnostics.

| Model | State | Action | Tangent norm | Preconditioned tangent norm | Raw-h quotient minus native action |
|---|---|---:|---:|---:|---:|
| A | 6s_1/2 | -0.166246919514 | 2.16695761809 | 0.162612127589 | 5.15105e-14 |
| A | 5d_3/2 | -0.167133419764 | 2.26346207533 | 0.146822487383 | 5.17155e-14 |
| A | 5d_5/2 | -0.168460411267 | 2.2505049542 | 0.156998686497 | 5.12292e-14 |
| A | 6p_1/2 | -0.169510704961 | 2.20834701317 | 0.164489346044 | 5.15421e-14 |
| A | 6p_3/2 | -0.169193791327 | 2.13402515281 | 0.173369881114 | 5.21661e-14 |
| B | 6s_1/2 | -0.163876272208 | 2.13693355972 | 0.234513279877 | 5.15023e-14 |
| B | 5d_3/2 | -0.164635755719 | 2.2277190488 | 0.212992623097 | 5.17073e-14 |
| B | 5d_5/2 | -0.166011523776 | 2.21696566237 | 0.212141859868 | 5.12228e-14 |
| B | 6p_1/2 | -0.166950915688 | 2.17083179318 | 0.23420126216 | 5.15650e-14 |
| B | 6p_3/2 | -0.166510425515 | 2.09088422336 | 0.23885247786 | 5.22109e-14 |

The separate raw-h/ket-norm correction changes these actions by approximately5.12–5.22e−14.
This observation cannot explain the measured1e−8 line mismatch. It does not replace the
native polynomial gradient, and its ambient radial derivative is a separately labelled route.

The0.2-weighted sum of five state gradients closes against each saved full aggregate:

| Model | Max radial difference | Max raw-coordinate difference | Max tangent difference | Max pg difference | Aggregate action difference |
|---|---:|---:|---:|---:|---:|
| A | 3.97904e-13 | 2.18807e-11 | 2.18798e-11 | 1.80272e-14 | 1.00184e-14 |
| B | 5.40012e-13 | 1.74419e-11 | 1.74419e-11 | 2.88797e-14 | 2.79713e-15 |

Radial/raw limits are1e−9, tangent/pg/action limits1e−10; all ten closure checks pass.

## Complete orbital cancellation accounting

All31 rows for every state and all four upper-minus-lower lines are retained in
[orbital_action_accounting.json](orbital_action_accounting.json), derived only from saved
[A state gradients](actual/A_R0_state_gradients.json) and
[B state gradients](actual/B_R0_state_gradients.json). The original full gradients are in the
corresponding saved NPZ files; no arrays were reloaded for this report.

| Model | Line | Signed action | Sum absolute orbital actions | Sum abs / abs net | Tangent norm | pg norm |
|---|---|---:|---:|---:|---:|---:|
| A | 5d_3/2 | -0.000886500249661 | 0.00922216590957 | 10.4029 | 0.262413458558 | 0.0646643104782 |
| A | 5d_5/2 | -0.00221349175276 | 0.0104232110397 | 4.7089 | 0.263108790189 | 0.0787689062502 |
| A | 6p_1/2 | -0.00326378544705 | 0.00915221405512 | 2.8042 | 0.131982340908 | 0.0432145513675 |
| A | 6p_3/2 | -0.00294687181275 | 0.0116674536231 | 3.9593 | 0.252173672218 | 0.0623284713985 |
| B | 5d_3/2 | -0.000759483510903 | 0.00903972430454 | 11.9025 | 0.260084412467 | 0.0711411261984 |
| B | 5d_5/2 | -0.00213525156766 | 0.0102160333818 | 4.7845 | 0.264316709389 | 0.0815625946925 |
| B | 6p_1/2 | -0.00307464348049 | 0.00779240145039 | 2.5344 | 0.119242880828 | 0.0440703161018 |
| B | 6p_3/2 | -0.00263415330684 | 0.00972638766445 | 3.6924 | 0.24503919901 | 0.0629212968882 |

The D3/2 orbital-action cancellation ratio is10.40 inA and11.90 inB. These ratios describe
this basis and this one direction; they are **not unique attribution of a physical error**,
correlation omissions, or a spectral uncertainty. Signed contributions from every orbital
are preserved rather than selecting a large-amplitude subset.

## Separate arithmetic diagnostics

Richardson was predeclared as `(4D(0.001)−D(0.002))/3`; it cannot retrospectively pass
an ordinary-FD gate. Its line discrepancies and ordinary EOL-callback discrepancy are:

| Model | D3/2 | D5/2 | P1/2 | P3/2 | EOL callback |
|---|---:|---:|---:|---:|---:|
| A | 2.12142e-09 | 4.32326e-09 | 7.92740e-09 | 1.22143e-09 | 4.71078e-09 |
| B | 9.20227e-09 | 1.25148e-08 | 7.96542e-09 | 1.36638e-08 | 1.04360e-08 |

The saved result also retains every Richardson state action and the separately labelled
weighted-state EOL route. The ordinary callback-minus-weighted-state central slope is:

| Model | h=0.002 | h=0.001 |
|---|---:|---:|
| A | 1.81899e-10 | -7.27596e-10 |
| B | 0.00000e+00 | 5.45697e-10 |

## Execution and limits

The root-owned one-use execution closed exit1 at2026-10-14 00:57:43UTC. It took
133.724s with peak RSS2,201,653,248bytes (2.050GiB),
on CPU27/one thread under8GiB AS and900s inner caps. This is uncontrolled timing, not
a software or parallel-efficiency comparison. No external backend or optimizer was called.

Actual work:10 H/CI,50 full block eigensolves,10 signed RDM contractions/gradients,
10 angular tables,43 observed early nullspaces,28 polar retractions,21 raw h/S builds,
694040 complete radial-key requests,7060 native gradient pair potentials,28240 gradient
component convolutions,11 tracker overlaps/110 projections,7 input decodes and12 output archives.
All entry attempts completed; all677 checks were evaluated,674 passed and the three above failed.

Complete-CI residual maximum is2.54658516496e−11Ha; maximum CI Gram defect4.4408920985e−15,
radial Gram defect3.66373598126e−15 and no-pair S-squared leakage6.44756677161e−30.
These are the measured control scopes, not estimates of the experimental spectral gap.

The stdlib-only [header preflight](header_preflight.json) preserves an earlier auxiliary
rejection of legitimate Fortran-stored CI arrays. Fortran order was never a driver gate;
only that unsupported auxiliary assertion was removed. No numerical producer ran there.
The earlier004 header failure, StageB ordinary FD, production stationarity and nested-byte
failures remain immutable. This completed experiment adds a local37/40 directional result.

No further numerical retry follows this closure. Root will select the next physical
discriminator; this narrow residual does not justify abandoning the correlation/state-balance
question or promoting a new physical error bound.

[Independent saved-result review](result_review.json) passes11 grouped verification checks
and confirms this completed scientific FAIL. It independently reproduces all40 ordinary
FD slopes/decisions from savedJSON (maximum slope discrepancy3.53e−18Ha). It does not
change any failed gate or use Richardson to rescue an ordinary-FD failure.
Review SHA256: `d03ef2d7a9cfcdfd668f73688016dd0a41f9324803988968a2d838a70e4446e2`.
