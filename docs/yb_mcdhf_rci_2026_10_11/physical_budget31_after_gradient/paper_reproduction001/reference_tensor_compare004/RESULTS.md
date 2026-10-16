# Closed complex tensor convention discriminator

004 closed exit 0: **36/36 PASS**, one external tensor launch. The unchanged
N3/seven-spinor synthetic C1 toy retains all 35 core/35 attachment determinants,
34 projected attachment rows and four fixed model roots. It is a capability and
convention test, not an atomic closed-core or paper EA reproduction.

The unique physical ket matrix is **`raw_HEFF.conj().T`**. The bound backend's
column-major eigensolver solves `raw_HEFF.T`; its right model coefficients must
be conjugated before the physical ket wave-operator lift. The raw public reader
and shared scientific source were not changed. The coefficient projector control
uses an independent eigensolve of `raw_HEFF.T`; actual `MVCOEF01` is preserved
but its rows were not decoded by this discriminator.

| Complete matrix conversion | Maximum error vs independent ket H_eff (Ha) |
|---|---:|
| Adjoint | 6.82790e-15 |
| Identity | 6.83795e-3 |
| Transpose | 1.36398e-1 |
| Conjugate | 1.36321e-1 |

Adjoint also matches the actual tensor-wave H_eff at 4.32987e-15 Ha,
retained Bloch at 3.21808e-14 Ha and all four physical projectors at
1.99386e-13. Its four root energies differ from the fixed native baseline by
2.43555e-15 Ha. The full core residual is 1.96347e-14 Ha and core energy
error versus independent full FCI is 1.77636e-15 Ha. Connected T3 remains
active in N+1 (maximum 7.98145e-4), with phase covariance error 8.16547e-16.

The finite-rank omission remains measurable: omitted 4p3h Bloch residual
6.11746e-4 Ha and physical full-H residual 5.28717e-4 Ha. These are diagnostics,
not passing full-FCI or experimental accuracy certificates. EOM and FSCC are
compared through their actual equations and model waves, not presumed identical.

003 failed before H/FCI/backend because the native integral ABI requires exact
Hermitian storage. Root authorized 004 to compute each h/F upper entry once,
store its exact conjugate below and keep the real diagonal unchanged. Full h/F
covariance errors were 2.23803e-16 / 6.51905e-17, with unchanged 1e-12 gates;
complete core/attachment H covariance errors were 4.44430e-15 / 8.88186e-15.
No averaging or gate/model/root/rank/backend change was made; failures001–003
remain intact.

Elapsed 0.83517 s; external call 0.22031 s; peak RSS 173,920,256 bytes,
one pinned CPU and one thread. This unpaired correctness run establishes no
performance advantage. All six preparation files, 358 private native sources,
307 bound tensor sources, declared input/runtime hashes and 77 actual native
imports were rechecked unchanged. No real-only native complex CC was called.

[Result ledger](actual/result.json), [complete phase arrays](actual/complete_phase_comparison.npz),
[raw tensor output](actual/fscc/cc.log), [contract](contract.json),
[freeze binding](execution_binding.json), [hash closure](handoff.json).
