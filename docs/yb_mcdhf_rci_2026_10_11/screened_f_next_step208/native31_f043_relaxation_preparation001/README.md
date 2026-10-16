# Bounded native31 f043 orbital relaxation

This is a separately declared continuation in frozen Atomx source f043, not
adoption of the changed current source. Its own 31-orbital baseline has 438
passing live controls; the single combined 4f–5f direction has 730 passing
finite-response controls at the two fixed h values. Those measurements do
not certify the whole orbital space or a stationary solution. Current-source
regression failures remain unchanged.

The [driver](run.py) selects the immutable 356-module f043 package before any
Atomx import. CI, its real Hamiltonian owner, matrix assembly, signed-density
gradient and public optimizer all come from that package. The independent
1595-dimensional exact-acceptance helper imports NumPy only. No current
owner, foreign backend or GPU adapter is consumed. The [dependency review](dependency_review.json)
records the difference from the pending current-epoch qualification contract.

One public `optimize_orbitals` call may take at most 16 new accepted steps.
It uses the literal recovered31 point, original seven nuclear projectors,
3689-coordinate layout and inverse metric, five physical UIDs with equal
weights, complete 1595-CSF target blocks and ordinary reported EOL. Baseline
H, roots, signed C, owner, radials and whole g/pg must reproduce their own
f043 evidence. Default history_size=10 starts empty. No energy callback or
quotient objective is substituted.

Every baseline/accepted checkpoint additionally computes the full-minus-
retained and direct complete normalized signed-density gradients. Exact CI
norms and all nonzero signed terms are retained; the angular capacity is
19,589 terms. The raw-h/ket-normalization ambient correction is recorded
separately. This costs at most 51 weighted-gradient calls over 17 points.
Both full-gradient comparisons and original stationarity gates are checked
without asserting a uniform path bound or adding CI response to an HF partial.

Every candidate CI archives all 1595 roots and signed coefficients before
residual, Gram and physical UID checks. Every accepted point archives its
coordinates, full metric, gradients and exact original Armijo comparison.
The private helper preserves the previous arithmetic and inequality, with
1595-dimensional/5-million-pair limits. Complete symmetry supports give at
most 4,623,405 examined pairs over 17 points; no budget is reset and no small
coefficient is discarded. A Python-integer census rejects inconsistent limits
before numerical imports or loops. Its certificate concerns finite stored-H descent.

The [contract](contract.json) bounds work at 65 H/65 CI, 51 weighted gradients,
one optimizer, CPU31 with single native/BLAS threads, 3.95 GiB and 600 seconds
(615-second outer launcher; 570-second optimizer soft limit). These are stop
caps, not a promise to finish 16 steps. The first unqualified candidate and
the previous qualified checkpoint are preserved. Lower/upper changes, four
lines and D/P centroids/splittings are reported at every accepted point;
NIST never selects a branch or changes weights. EOL descent need not reduce
the experimental gap, and original stationarity thresholds remain strict.
Strict attempted/completed counters remain active for every radial source
pair; progress writes occur at complete-gradient boundaries rather than per
pair. Full checks are preserved at final return or first failure.

Preparation used source, AST and JSON only, with zero numerical imports,
NPZ reads/rehashes or scientific calls. No shared source, registry, website,
timer or earlier evidence was changed. A separate exact root execution seal
and foreground launcher are mandatory. The prepared command is recorded in
the contract; no optimizer has been executed by this preparation.
