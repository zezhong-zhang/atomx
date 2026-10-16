# Complete 4f12 fixed-orbital numerical preparation

This preparation retains all 82 Pauli-allowed NR configurations with two 4f
holes and three electrons in the existing seven-shell valence pool. It adds
them to A91/B100 without changing the 31 radial waves, original accepted13
coordinates, 69 electrons, Z70/A175 Fermi nucleus, positive-energy projector,
metric or unshifted Dirac–Coulomb Hamiltonian. It does not perform orbital
optimization or append Breit, QED or experimental corrections.

[Complete models](complete_models.json) contain every 173/182 NR string and
all sorted jj occupations in each target sector. An independent NR splitting
and integer magnetic generating function reproduces every earlier dimension;
the B1+ pattern hash also equals the measured complete008 graph.

| Sector | A full CSFs / raw-M | B full CSFs / raw-M | B ordered structural pairs |
|---|---:|---:|---:|
| 1/2 even | 5,575 / 91,761 | 5,717 / 92,715 | 43,720,175 |
| 3/2 even | 10,192 / 86,364 | 10,418 / 87,182 | 40,882,406 |
| 5/2 even | 13,196 / 76,206 | 13,432 / 76,800 | 35,054,314 |
| 1/2 odd | 5,561 / 91,394 | 5,704 / 92,377 | 43,261,341 |
| 3/2 odd | 10,183 / 86,031 | 10,415 / 86,878 | 40,517,908 |

The union needs **18 unique j^q cases and 63 cold null-space calls**. These
replace the 15/43 census of the smaller model. The full radial-key upper
bound is 384,706: 161 same-kappa one-body keys and 384,545 two-body keys.
Reduced-C triangle/parity rules allow at most 706 unordered rank/source-pair
potentials and 496 pair densities, within the native 1024/512 LRU capacities.
These are finite-space bounds, not measured cache usage.

The largest B determinant CSR has at most 43,720,175 stored entries, about
525 MB with FP64/int32 storage. Typed FP64/int64 build buffers have a 700 MB
payload bound. Symmetrization, graph-construction highwater, Python/cache
overhead, radial workspace and allocator retention must also fit. The known
1.917 GiB peak covers the B1+ graph only. The proposed 8 GiB/1200 s per-worker
limit is an explicit bounded trial, not a proven fit or time prediction.
[Detailed plan](plan.json) retains all per-sector bounds and stopping rules.

[The independent source review](source_review.json) finds the shared physical
angular functions unchanged. Streamed assembly contracts the same signed
terms directly into numeric CSR. A CSF column has support in a single
occupation pattern; its diagonal therefore needs only that pattern's
determinant principal block. The full matvec retains every cross-pattern
coupling. The review is not a substitute for numerical source qualification.

`worker.py` uses one complete B operator per sector. Exact full signatures
identify A as the restriction removing the nine explicitly B-specific NR
rows while retaining all 82 new NR configurations. Each arm uses native
Davidson with two selected roots, 1e-9 Ha solver residual tolerance and a
fresh 1e-8 Ha direct residual gate. No full angular coefficient table,
dense full H, partial public tracker or retained full-spectrum snapshot is
constructed. Signed selected amplitudes, complete basis signatures, nested
indices and all bindings are saved if executed.

Physical matching embeds the owned accepted13 A/B target vectors by full
signed CSF signature in the same orbital basis. Their original531 physical
tracking is retained as ancestry. Directly embedding the earlier531
coefficients would ignore orbital changes and is not permitted. Strict
squared overlap greater than one half, Gram, direct residual and observed
two-root-gap gates qualify numerical isolated-root transport. Bessel's exact
omitted-overlap statement is conditional on an isolated exact eigenstate;
the numerical gates do not certify an omitted-spectrum gap rigorously.

Default invocations of `prepare.py` and `worker.py` are SOURCE/integer/JSON
only. They import no Atomx or NumPy scientific branch, open no NPZ and build
no large CSFs, angular tables, H or CI. SOURCE checks and Ruff passed for
this preparation. Execution requires an exact root seal and the completed
PASS result of `../matched001` at the same current dirty source identity.
The root may dispatch at most three workers on CPUs28–30 with one thread
each. A first worker failure writes shared `STOP`; later work rejects it
and preserves partial results. No automatic retry or change of physical
model, tolerances or resource caps is authorized by this preparation.

The [completed physical execution and offline review](aggregate001/RESULTS.md)
now pass all five sectors:265/265 live and554/554 offline checks. The full
specified class energies and four paired excitation shifts are calculated;
the source-only preparation remains distinct from that measured evidence.
Worker `de8f2b68…` is the root-reviewed guard-alias successor to the preserved
pre-review worker `90aa9e18…`. That source-only guard change preceded every
scientific run and changed no underlying angular function or gate.
