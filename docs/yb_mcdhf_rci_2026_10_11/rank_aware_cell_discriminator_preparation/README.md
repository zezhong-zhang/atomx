The candidate fixes two private inner quadrature orders; no shared source or
semiseparable matrix routing changes. The source is ordinary500ac plus
`ks+floor(k/2)` and `max(ksp,ksq)+floor(k/2)` for nonnegative Coulomb rank.
Spline products times `r^k` have degree `2s−2+k`; these Gauss orders cover that
inner polynomial. Negative ranks retain the original algorithm and remain
unqualified. Actual outer quadrature and the radial/model space stay unchanged.

This is a prepared experiment, with **0 numerical or physics calls so far**.
The [contract](contract.json), [runner](run.py) and
[840 source/input preparation controls](preparation_verification.json) need
one exact root execution seal before numerical use. Preparation PASS is not
quadrature or Yb qualification.

Three independent single-cell anchors use signed scale product −24 and all
PP, QQ, PQ and QP components. The linear case covers ranks0/1/6; ordinary
P=r4/Q=r5 and mixed RKB P-order5/Q-order6 cover rank6 with an exactly zero
unused P coefficient. Outer Gauss16 is fixed for these anchors and covers their
maximum polynomial degree20. Dense banks, factors and the unchanged actual CI
pair helper are checked against the analytic two-triangle integral. The earlier
[rank6 analytic failure](../correlated29_radial_reuse_qualification_preparation/RESULTS.md)
remains failed and immutable. Its linear-r rank6 case is mathematically
convergent, but not an angularly allowed s-s contribution; it alone does not
quantify a physical Yb error.

After anchors, two fresh isolated source processes consume the exact same
123-spline/29-orbital recovered point and complete five targetwise signed CI
tables. They evaluate every one of **7398 affected k≥2 radial keys**, sharing
333 rank/source pair potentials and278 density pairs per source. All 25568
signed targetwise contributions are divided by their own exact saved CI norm.
There is no coefficient screening. Unchanged h/k0/k1 operator changes are
algebraically zero because their code and inner order are identical; that
statement does not certify their absolute quadrature accuracy.

The output gives each of five states' fixed-CI δE and each upper-minus-ground
transition shift. These are expectation corrections on the original stored
CI states, without new H, CI, gradient or optimizer evaluation. They are not
rediagonalized spectra or optimized MCDHF results. The equal-weight EOL change
is only a diagnostic summary. Original stationarity, derivative and experiment
failures remain separate.

The total cap is CPU31/thread1,3.95GiB,shared300s,outer315s+5s kill;
**14 genuine h/S builds**,14796 public radial evaluations,3 basis setups and
3 small Gauss companion eigvalsh calls. H/CI/g/Dirac/optimizer/GPU calls are0.
Fresh processes and separate NumBa cache directories prevent cross-source
reuse of geometry-only bank/snapshot caches. Equal physical inputs do not
assert equal numerical operators. Stage times are uncontrolled observations.
Any failed anchor, identity, call or resource gate stops the experiment; later
phases are not run, and no retry or tolerance change is allowed.

Documentation impact: this private numerical discriminator does not change
shared production documentation or the existing solver's guarantees. A later
shared fix must update the old “ks Gauss points / DBSR convention” comments,
negative-rank scope and numerical qualification in its own reviewed epoch.
The present durable insight is limited to the positive-rank inner polynomial
order and the preserved measured failure. No performance insight is promoted.
