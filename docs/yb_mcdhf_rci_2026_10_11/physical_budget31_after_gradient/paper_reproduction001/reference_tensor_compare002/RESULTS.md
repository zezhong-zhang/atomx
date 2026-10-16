# Closed real-case tensor comparison and orientation diagnosis

The unchanged N3/seven-spinor toy reached nonzero T3 action in N+1 and one
actual corrected full-tensor FSCC solve. The original matrix-direction gate
failed and remains [closed](actual/result.json). All native/core/tensor data,
original nine decoded diagrams, raw backend scratch and full arrays are retained.

| Independent quantity | Observed maximum |
|---|---:|
| External core energy minus complete N3 FCI | 8.88e−16 Ha |
| Complete tensor core residual | 1.95e−14 Ha |
| Native/tensor complete T1/T2/T3 lift difference | 2.32e−14 |
| Genuine T3 action in N+1 | 7.98145e−4 |
| Four attachment root difference | 2.45e−15 Ha |
| Native/tensor intermediate-normalized model wave difference | 2.15e−13 |
| Direct stored H_eff minus wave-projected H_eff, failed | 6.83795e−3 Ha |

The [saved-array discriminator](postfailure_matrix_orientation.json) shows that
the stored real H_eff is the transpose of the ket-wave projection: difference
4.48e−15 Ha. With that orientation, retained Bloch residual is 3.21e−14 Ha.
The separate [physical-wave diagnostic](postfailure_physical_wave_orientation.json)
recovers all four physical projectors within 1.99e−13. Its full-H residual
5.28717e−4 Ha matches native; the omitted 4p3h Bloch row remains 6.11746e−4 Ha.
This is a finite-rank limitation, and selected roots differ from full N4 FCI
by 4.08e−7 Ha. No global EOM/FSCC or atomic accuracy claim follows.

The original direct-orientation physical-wave residual 8.67e−3 Ha is confounded
by the same direction choice and is superseded for interpretation by the saved
wave diagnostic; its original receipt is unchanged. No tensor equation defect
is demonstrated. Real data cannot distinguish transpose from adjoint; a separate
complex phase-covariant case is required before changing a public convention.

The first [001 metadata failure](../reference_tensor_compare001/actual/result.json)
also remains frozen. It preceded every H/CC/EOM/backend call. This 002 successor
only corrected the packed ABI count 231→266, including conjugate duplicates;
all operators, ranks, active states, gates and caps were unchanged.

Total inner time was 0.879684 s, peak RSS 169,308,160 bytes. Native+independent-FCI
phase took 0.125527 s and external call 0.220249 s; these scopes are unpaired,
so they imply no efficiency comparison. Fourteen guard/patch tests passed.
Private 358-file Atomx snapshot and all 307 corrected tensor source files remain
unchanged. [Handoff](handoff.json) binds source-line and complete proof identities.

Receipt SHA: `9d3a4b5ff605446dd42cf727d2c1d266df6e2be490a358b79e9a14452d6999cd`.
Complete arrays SHA: `b348a45330ef25dc6d619269695e2b1c0c48df0973308a93415c6ff0daa9ad83`.
The Ne4/eight-spinor native atomic seam has not been launched.
