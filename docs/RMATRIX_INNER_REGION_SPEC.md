# R-matrix Multi-Channel Inner Region — measured gap + design (2026-10-01)

## Measured state (probes this session, scripts/rmatrix_burke_abinitio.py)

The package (scattering/r_matrix/, ~3000 lines) is real plumbing: channel
table, direct/exchange CC potentials, Bloch B-spline inner region,
outer matching, eigenphase machinery, Breit-Wigner fitter.  23 tests
green.  BUT the "Burke-1971 benchmark" is a REPRESENTATION test of
empirically injected poles (l2_pseudostates.py states this itself):
with include_l2_pseudostates=True the fit returns the tabulated values
to 5 decimals; with it FALSE the computed eigenphase is FLAT across the
pole windows:

| block | pure-CC eigenphase variation over +-8 Gamma | needed |
|---|---|---|
| 1Se(2s2) | 0.106 rad | ~pi rise |
| 1Po(2s2p) | 0.005 rad | ~pi rise |

The inner region as built (projectile in the static+exchange potential
of ONE frozen hydrogenic target orbital) is distorted-wave grade: it
has no dielectronic correlation, cannot bind the (2s2) quasi-state,
and its coupling to the open channel carries no Feshbach structure.
NO ab initio observable has ever validated the CC Hamiltonian.

## Design: a real (N+1)-electron inner region from existing machinery

The repo already owns every ingredient of an RMATRX/DARC-style inner
region:

1. **Box basis**: the Dirac B-spline workspace IS the R-matrix basis
   (cavity splines; first-spline-dropped convention already matches).
2. **(N+1) expansion**: Psi_k = sum_ij c_ij A[Phi_i(N) u_j] + sum_j
   d_j chi_j(N+1) — the m-scheme sparse CI (NIST-validated today on
   Ce3+/Pr3+/Eu3+/Nd3+ multiplets) assembles exactly this once the
   continuum spinors are in the spinor table; chi_j = box-bound
   (N+1)-electron configurations supply the dielectronic correlation
   that the current model lacks.
3. **Correlated targets**: Phi_i from the CI machinery (He 1s2 1S,
   1s2s/2p 1,3 L), not frozen hydrogenic orbitals.
4. **Surface amplitudes**: w_ik = <Psi_k | r=a> per channel — the
   existing surface.py generalised from one channel to blocks.
5. **Outer region**: cc_outer.py already multi-channel.

Gate plan (physics gates BEFORE any anchor use, per session discipline):
  G-A  He 1s2s 3S / 1s2p 3P threshold positions vs NIST (target quality)
  G-B  1Se(2s2) pole AB INITIO within ~0.02 eV of Burke -0.77816 Ha
       (pi-rise must appear in the PURE-CC eigenphase, no injected poles)
  G-C  1Po(2s2p+), 1De(2p2) widths within 30%
  G-D  eigenphase unitarity + Buttle-corrected R-matrix continuity

Cost estimate: the (N+1) CI for e-He with l_c<=2 and ~40 splines is
~1e4-1e5 determinants per LSpi block — sparse assembler scale, minutes.

## Decision

The empirical-pole path stays (labelled honestly); the inner-region
upgrade is the next HYP-grade increment.  External review of this
design BEFORE implementation (workflow).
