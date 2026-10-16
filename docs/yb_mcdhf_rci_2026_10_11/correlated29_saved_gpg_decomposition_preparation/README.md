# Saved original-metric gradient decomposition

Prepared for one root-sealed offline run on global80 and the literal run005 last qualified checkpoint, global208. No arrays have been loaded during preparation. Run005 closed successfully, but both original stationarity gates still fail; this preparation does not change that result.

The driver loads only the two saved coordinate/gradient archives. It reconstructs `raw = P.T @ radial_gradient`, `g = T(raw)` and `pg = T M T(raw)` using the original seven projectors and saved per-column inverse blocks. `T(v) = v - Y sym(Y.T v)` is the producer's same-kappa tangent map. Layout groups use kappa order; radial rows use (n,kappa) order, with explicit key mapping between them. The legacy metric fingerprint, coordinates, direct P*x unpack, physical target UIDs and original thresholds are checked. No Atomx geometry, Hamiltonian, CI, density contraction, gradient producer, spectral solver or optimizer is called.

For each of all29 orbital columns and each complete kappa group, the result records `||pg_j||²` and signed `g_j·pg_j`, then checks their sums. The19 core,5 bare-valence and5 new-virtual partitions also mask the *raw* coordinate input separately and apply the complete `T M T` map. Those legal tangent driving forces can occupy columns outside the input mask. All nine signed cross-work terms and all nine residual cross terms are retained; isolated group norms are not additive whole-residual contributions.

Only exact saved geometry/source lineage and direct unpack are established here. Physical-S positivity and the nuclear positive-energy span were validated by the original producer and are not solved again. These coordinate diagnostics and the producer's energy indicator are not strict bounds on state energies or transition errors. Large core energy does not imply core-dominated residual. The measured decomposition must precede a decision about preconditioning or staged orbital optimization.

Resources: CPU30, one BLAS thread,1GiB address-space limit,30s inner/35s outer cap. The consumer refuses execution without an exact once-only root seal matching the frozen driver, contract and input hashes. It preserves first failure/partial results and cannot choose another point, retry or improve producer certification. Runtime algebra remains unexecuted until that seal.

The new files are local prospective evidence. Historical run005, previous scientific proofs, shared source, website and current root reports remain unchanged.
