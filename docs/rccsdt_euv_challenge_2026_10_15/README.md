# Tin EUV single-core challenge design and historical RCCSDT pilot

The latest user revision reduces the challenge to one Sn12+ atomic-spectrum
problem: saturated 4p/4d/4f particle-hole CI and complementary core-valence
correlation. CI+MBPT is the recommended reproduction route; RCCSDT is an
optional method comparison. The current author design is
`/home/zzhang/playground/tasks/atomx-rccsdt-euv/submission/赛题设计.md`, with
the concise brief in `submission/core-task.md` and design-only archive
`atomx-tin-euv-core-design.zip` at the same workspace root. This is not a
new validated platform task. A [native runnable CI baseline](../TIN_EUV_BASELINE.md)
now solves the paper-defined CI space through{6}, retaining115715 roots
and signed E1 lines. The highest bare-CI row agrees within0.0567eV;
source-matched paper core MBPT, correlated operators and experiment remain pending. Published Table I centroids and convergence
rows are primary theory references; NIST Sn XIII identity and theoretical
threshold are auxiliary. Matching experimental data, complete reference
calculations and resource calibration remain release prerequisites. The
[current source guide](../EUV_SOURCE_MODEL.md#native-model-qed-operator-qualification)
records subsequent native signed rates, complete-block orbital stationarity
and source-qualified CC comparisons. Its actual joint all36-core/full-n7
model-QED FSCC splitting13143.815225cm^-1 still fails the experimental
13179(8)cm^-1 gate by-35.184775cm^-1. These diagnostic improvements do not release the
challenge or certify a calibrated source spectrum/CE.

The subsequent native finite research operator retains all induced zero-
through-three-body terms and passes independent finite P/Q and signed
native controls. Full fold6 J=2/J=3 sectors are consumed without diagonal
shift substitution. With static Breit/model QED, n5/n6 external Q gives
9671.231640/9281.979114cm^-1 against9786(30): both fail, and the larger-space
failure remains. Full coherent response locates the change in cross terms.
The source guide distinguishes this Hermitian P/Q construction from original
AMBiT Brillouin-Wigner reference/diagram conventions. No new archive or
reference score is released.

Original AMBiT MBPT source now consumes the same native radial integrals in
a separate diagnostic. Its complete Sigma1+2 coefficients pass independent
particle-hole/physical normal-order, phase and angular controls; full fold6
J=2/J=3 consumption gives10361.703834/10390.559989cm^-1 at n5/n6, both
failing experiment. Original Sigma3 directional values and configuration/
triangular/leading-selection rules remain unresolved as a complete operator.
These diagnostics retain the declared isotropic H0 and do not establish the
paper-specific reference, full MBPT, corrected EUV spectrum or CE.

The complete native n5 nonperturbative RCCSDT comparator now converges with
unchanged amplitude, raw-equation and complete-multiplet gates. Its splitting
13067.002698cm^-1 is111.997302cm^-1 below experiment and still fails8cm^-1.
The measured same-space triples change is-9.399703cm^-1; it is not added to
the full-n7 CCSD value. A private exact block-matching optimization permits
the1332.19s run after the original3600s timeout; paired controlled timing and
production builder integration remain separate from this physical result.

The native statistical-jj AOC-reference audit is now complete. It preserves
all original n4 radials and passes an independent physical38-electron
P/Q matrix check, but full fold6 induced0..3body n5/n6 gaps8753.338132/
8765.259249cm^-1 still fail9786(30). A same-point induced-rank2 diagnostic
also fails; omitting Sigma3 cannot be selected as an experimental repair.
Original AMBiT and native statistical averaging fields differ and remain
distinct declared models. A private two-valence FSCC comparator passes
complex finite FCI/Bloch and real Be C1/axial/gauge full-matrix controls.
Native Sn12 complete45-pair n4 CCSD/CCSDT give9408.777175/9397.227492
cm^-1 (both FAIL30), rank increment-11.549683cm^-1. A separate all36-core,
full244virtual n7 two-valence CCSD run gives9709.739955cm^-1, error
-76.260045 FAIL30; its complete core/one-attachment prerequisites replay
the prior Sn13 result bitwise. Independent phase-bound LS assignment checks
five measured levels: only3F4 meets its own50cm^-1 D2 gate; the others fail.
Paper-model P-space and self-consistent-QED differences remain explicit.
Public0h2p support is not released.
See the owning source guide for full point, resource and root gates.

The v1.1.0 five-layer design, archives and receipt below are historical and
unchanged. Its finite-model scores do not evaluate the new core task. The
previous author documents are preserved in `versions/1.1.0/design_snapshot`.

The user requested a challenge package combining atomx and ~/playground. The
current authoring workspace is `/home/zzhang/playground/tasks/atomx-rccsdt-euv`.
Its canonical task is `task/`; author design, references and feasibility live
in `submission/`; actual generation, image logs and numerical proof live in
`build/`. No atomx production equation or sealed Yb evidence was changed.

Historical version1.1.0 follows the user's explicit NIST/publication standard:
14 external observables are frozen with source/uncertainty/state identity.
The final score is code_score times external physics_score; absent values
keep a fixed denominator. Two offline runs give code1.0, physics0.05 and
coverage1/14; full scientific acceptance fails. The original1.0.0 code-only
ceiling is preserved in the version archive and is not a physical pass.
NIST's four ionization limits are theoretical values with2.5–3eV one-sigma
uncertainties. Published experimental levels/lines and2026 theoretical gA
centroids are separate reference groups. The2026 final table values can be
used despite the absence of its unpublished supporting line data/software.

Scientific corrections: Sn14+ has36 electrons and a Kr-like closed core;
Sn13+ is a4d electron attachment, not a4d hole. Sn12+/Sn11+ require two/three
valence particles. The2026 Reitsma paper is particle-hole CI+MBPT, not CCSDT;
its published data-availability statement says supporting research data and
software are not publicly available. Generic AMBiT source availability does
not supply the missing paper-specific input/output.

The executable pilot uses actual unmodified native Dirac-Coulomb integrals:
120Sn Fermi nucleus, ns90/order9/R60,32 frozen and4 correlated electrons,
complete4p3/2 occupied and4d3/2+4f5/2 virtual multiplets. The core has790
canonical SDT coordinates; attachment has910 R3 coordinates. Independent
fermion exponentials and native tensor recipes agree off shell, and finite
CI supplies a same-integral anchor. The model is neither radially converged
nor a full Sn spectrum. Supplied DHF oscillator strengths are kept separate
from correlated EA roots; full dressed CC properties remain a research gate.

The current native production tensor backend already implements complete
nonperturbative SDT and projected EA-R3 actions. The challenge exposes blank
student seams and an independent teaching scaffold; it does not claim a new
production RCCSDT implementation was necessary or completed here.

The historical package includes fixed-image Docker build material, a bounded verifier,
reference solution, adversarial probes, physical invariant checks and the
five-layer research design. Real experimental matching, multivalent full
Sn, measured MPI efficiency and remote Bohrium image publication remain
explicit release gates. The local pilot score cannot certify those gates.

Documentation impact: production API, physical methods and current Yb
measurements are unchanged, so their guides remain authoritative and are
retained. This note records the new challenge consumer and applicability.
Durable lesson: verify ionic electron counts and exact reference provenance
before defining quantitative scientific scoring; never grade unknown paper
inputs by invented numerical targets.
