# Native sparse Hamiltonian and Davidson qualification

This is a fixed-basis qualification of existing native machinery before a
larger 4f¹² correlation calculation. It does not add a correlation class or
change the shared-orbital objective.

The frozen f043 implementation builds complete native sparse determinant
Hamiltonians and exact sparse CSF projections for A (1595 CSFs) and B (2574
CSFs), at accepted point13 with identical 31 radial orbitals, original finite
A175 nucleus, nuclear positive-energy projector and radial metric. All 69
electrons and all five original symmetry sectors are retained. The complete
owned dense A_R0/B_R0 results from decision004 execution002 are read-only
oracles; this experiment never rebuilds a dense Hamiltonian.

The discriminating checks are:

- Every dense reference eigenvector is acted on by the native sparse operator
  in complete batches of at most32 columns. Full action and residual limits
  are1e−8 Ha; diagonal values, exact cross-sector zeros and bilinear symmetry
  are also checked.
- The existing native Davidson solver finds the lowest2 states separately in
  every complete symmetry block. All five owned target UIDs are root0 in the
  frozen dense controls; that fact is checked before execution. The selected
  residual and energy limits remain1e−8 Ha, signed wave/projector1e−8 and
  Gram1e−10. Phase alignment uses signed full-row overlaps.
- The original tracker and state-average APIs explicitly require full CI
  spectra. They are not called on the selected iterative result. No iterative
  Hamiltonian snapshot or changed-basis tracking certificate is fabricated.

[contract.json](contract.json) fixes all source/input hashes, field shapes,
scientific gates, resource bounds and first-failure stopping. The default
[run.py](run.py) performs only stdlib SOURCE/JSON checks, including356 frozen
source hashes and the original complete43-call angular nullspace census. It
performs zero NPZ loads and zero science. A root-created exact driver/contract
seal is required for one execution; an existing `actual/` prevents reuse.
CPU27, one BLAS/OMP/Numba thread,8 GiB address space and900 s inner/915 s outer
limits apply. Timing is diagnostic only; no comparative performance claim is
made. Larger4f¹² energies, stationary MCDHF and experimental precision remain
outside this qualification.

Documentation impact: shared source, APIs and defaults are unchanged. This
report is the experiment-specific account; parent status/navigation is owned
by root. Existing `ci_solve`/snapshot/full-tracker limits remain unchanged.
No new general insight is promoted before actual measurement.
