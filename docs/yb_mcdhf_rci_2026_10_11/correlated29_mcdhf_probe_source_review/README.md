# Correlated 29-orbital probe: independent source review

This review binds the final prepared native consumer, not an executed numerical
qualification. It imports no NumPy, SciPy or Atomx module, reads no NPZ arrays,
and executes no Hamiltonian, CI, radial gradient, Dirac solve or optimizer.

Two material draft findings were closed before the final source freeze:

- The original tracker groups follow first-seen CSF symmetry order
  `6s,D5,D3,P3,P1`; the target labels use `6s,D3,D5,P1,P3`. The old positional
  zip would mislabel D/P energies and multiplets. The final consumer requires
  singleton reference UIDs, unique complete matching and the selected physical
  symmetry before attaching each original label.
- Geometry validation and repeated tracking needed explicit accounting. The
  final cap includes 21 resolutions, 22 overlaps including initialization,
  105 rank-one SVDs, 36 geometry eigenvalue checks, 37 SciPy eigensolves,
  28 shared polar retractions and at most 28 NumPy fallback eigensolves.
  Native transient Dirac retries consume the same SciPy cap; they are not free.

The full scientific budget remains 6 H / 6 complete CI / 1 full radial gradient,
7 validation Dirac solves and at most 77 h/S assemblies. The six physical points
are the exact canonical anchor, its separately declared recovered-coordinate
point and four signed finite trials. Four mixed-increment diagnostics add
28 h/S assemblies and eight tracker resolutions. No optimizer is invoked.

The raw one-body matrices are checked against raw assemblies separately.
`L(h)=tril(h)+tril(h,-1).T` is checked against the functional and saved logical
operator; arbitrary raw matrices are not declared equal to L(h). This source
review does not bound their bilinear discrepancy in the new broad directions.
Actual signed directional gates, including CI response, must resolve that scope.

The coordinate gradient follows the native convention: `P.T @ radial_gradient`,
then Stiefel tangent projection, the declared copied SPD inverse blocks, then
tangent projection again. The expanded 3451-coordinate metric is explicitly a
new epoch. Preserving seven P arrays and old24 inverse columns does not transfer
old24 stationarity to 29 orbitals. The external-new5 and old/new rotation norms
are gradient diagnostics, not omitted-physics or experimental error bounds.

Both reported-root and retained-Rayleigh central differences use the declared
`h=1e-6` and unchanged `max(1e-8,1e-5*abs(analytic))` gate. Ordinary subtraction
of large total energies is reported with its ULP/h resolution, not used as the
sole rejection criterion. One gradient and two directional checks do not certify
a whole-gradient independent replay, complete MCDHF convergence, all 2163 parent
CSFs, a global ground state or experimental precision.

[verification.json](verification.json) records exact final hashes and source-only
checks. The 424 native preparation controls are an independently bound input;
they are not counted again as this review's controls. Root execution remains a
separate authority. Shared test `045ee541…` and all scientific source stay intact.
