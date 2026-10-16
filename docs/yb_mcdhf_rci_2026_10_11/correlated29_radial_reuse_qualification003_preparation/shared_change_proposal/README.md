This is a source-only proposal for the two source files already qualified in
private qualification003, plus four independent public dense-CI tests. Shared
source and tests have not been changed here. Root owns application and numerical
execution after checking the exact preimages in `contract.json`.

The dense helper removes the nearest integer Hartree of a block's first real
diagonal before the existing LAPACK solve and restores that scalar to energies.
It keeps complete `evd`, subset `evr`, the public signature, Davidson and the
retained original Hamiltonian. Matrix assembly uses the existing complete
semiseparable PP/QQ/PQ/QP factors at ns>=96. The gradient is unchanged.

The two proposed source files are byte-identical to the qualified003 candidate.
Its new ordinary centered epoch reproduced its own saved-H energies bitwise;
the complete 1046-state comparison passed 1126 gates. Comparison with the old
unshifted energy epoch is source evolution. Both earlier route failures remain
failed. Original stationarity, experimental gaps and GPU/performance status are
not certified by these numerical equivalence gates.

`changes.patch` contains only those two source files and the existing CI test
file. The old six testcases and tolerances remain intact. Four new cases compare
complete/subset CI for both signs of a small off-diagonal element against an
independent 2x2 angle/projector formula at a large negative energy origin. Root
can run `tests/test_ci_dense_orthogonality.py` after application; the prospective
counts and resources are declared in `contract.json`. No test has run here.

`before/` preserves the old file epoch, and `proposed/` preserves the reviewable
new bytes. Historic consumers must select their archived source/test paths rather
than treating a later shared file as the old identity. The current native run005
keeps its frozen f043 source and original 3451-coordinate metric.

The documentation update target and exact proposed text are in
`documentation_impact.json`. Controlled warm measurements need a separate root
seal; the qualification's cold/profile observations are not a speedup certificate.
