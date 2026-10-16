# Complex phase discriminator of the same finite triples case

This is one separately authorized fixed representation change of the exact
N3/seven-spinor toy from [closed 002](../reference_tensor_compare002/RESULTS.md).
The original direct H_eff failure remains failed. The complete 35 core and 35
attachment determinants, 34 R rows, nonzero T3, omitted R4 row, four active
model columns and four frozen roots are retained. No Ne4 case is run.

The seven orbital phases have fixed angles π×(0, 1/7, −1/5, 1/3, −1/4, 1/6,
−1/8). In the rephased orbital basis, h and F transform as D†hD; every signed
ERI has its complete two-bra/two-ket phase. Full determinant phases independently
transform both complete H matrices, T, model Ω, ket H_eff and physical states.
Independent ordered 1/4-tensor occupation-list H construction and complex full
FCI verify every matrix entry and both complete spectra before the external run.
Current real-only native CC/EOM is not invoked with complex data.

The same corrected source-bound EXP-T binary is the independent tensor
comparator. Actual complex core T1/T2/T3 and attachment S1/S2/S3 diagrams are
decoded completely using the existing signed bra-to-ket convention. Their
core equations, phase covariance, Ω normalization and actual wave-projected
ket H_eff are checked before considering a matrix-direction answer.

All four complete maps of the raw formatted H_eff—identity, transpose,
conjugate and adjoint—are independently tested against expected ket H_eff,
actual tensor-wave projection, retained/full Bloch residual and all four
physical projectors. Exactly one must pass the unchanged 1e−10 gate, and
every other matrix map must differ by at least 1e−6. Root eigenvalues alone
cannot distinguish these conventions. Source LAPACK column-major handling
also motivates a separate conjugation check for raw.T model coefficients.

Both cases use the same private immutable native source snapshot and the
same 307-file corrected tensor source manifest. Native complex solver calls
must remain zero. Original scratch/HEFF and diagrams, complete phases/H/FCI,
all competing maps and physical waves are saved. No shared/backend/parser
source, rank, root, Fov element, scalar tolerance or physical model changes.

CPU 28, all numerical threads 1, 8 GiB address-space, 110 s inner and 120 s
outer limits remain; one external launch has a 60 s timeout and 512 MiB cap.
The complete bound baseline archive is loaded only after source/resource
admission, with a 64 MiB decoded-array cap. First failure is preserved;
no automatic retry or extra case is authorized by this record.
Unpaired phase times imply no performance conclusion.

`run.py` defaults to source-only verification; execution uses the repository
`.venv` with the declared CPU/thread/time environment and `--execute`.
The result determines the tested public matrix convention only after full
wave/operator evidence; any public interface change is a separate root decision.
This remains a synthetic non-multiplet equation control, not an atomic/NIST
or Sahoo2025 EA reproduction. Root owns current guides and EARS integration.

004 preserves the closed 003 exact-Hermitian ABI rejection. Its sole change computes each h/F upper coordinate once and stores its exact conjugate below, with unchanged real diagonals. Full h/F and independent complete determinant-H covariance still require 1e-12; no averaging, physical/model/root/rank/tolerance/backend changes. Root explicitly authorized this successor; the next scientific failure terminates this experiment.
