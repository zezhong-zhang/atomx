# Sn12+ EUV CI baseline

The single-core challenge now has a native bare-DC calculation through the
paper-defined {6} space, with115,715 complete CI roots.
Complementary core MBPT and complete physical/experimental acceptance remain
pending. The owning source-bound report and full local outputs live under
`/home/zzhang/playground/tasks/atomx-rccsdt-euv/build/sn12_blocked`.
Earlier source/results remain in `build/sn12_baseline`; its original delivery
is archived in `versions/ci3-baseline`.

## Physical contract and explicit spaces

Z=50,120Sn Fermi nucleus, statistical VN DHF `[Kr]4d2`,ns90/ks9/R60bohr,
Dirac–Coulomb, complete4p/4d/4f fine-structure partners, length/long-wave E1.
No core MBPT, Breit or QED contribution is applied, and no published correction
constant enters the solver. Radial/external-virtual convergence is separate.

Promotion order is m=(6-N4p)+N4f. Independent angular counts and enumeration
agree on22 NR configurations/115715 levels at{6},5294 through{3},19019
through{4} and52517 through{5}. This is the paper expansion, not the
unrestricted eight-electron p/d/f Fock space. Every declared CI root remains.
Only the emission endpoints have a separate explicit `emission_max_fold=3`
window: higher-fold configurations still contribute to every wavefunction.
States carry J/parity, all fold weights and leading CSF. Dominant fold assigns
bands without target-energy matching; ambiguous assignments are reported.

## Reproduction and measured physics

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python tools/run_tin_euv.py --storage blocked --ci-engine spinor \
  --max-fold 3 --max-levels 6000 --max-determinants 6000 \
  --timeout 300 --output output/sn12-fold3
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python tools/run_tin_euv.py --storage blocked --ci-engine spinor \
  --max-fold 6 --emission-max-fold 3 --max-levels 116000 \
  --max-determinants 80000 --memory-gib 64 --timeout 1800 \
  --output output/sn12-fold6
```

Output paths must not exist. Counts and resource admission precede allocations;
caps reject without silently dropping states. The64GiB address-space cap is
explicit. Full{6} CI assembly has a conservative36.38GiB planning estimate,
not a whole-workflow RSS guarantee. The producer timeout covers DHF/CI/E1 and
streamed lines; final array/JSON serialization is measured in the end-to-end
wall time. Inputs record actual source hashes and thread environment; changes
to the Python-source file set or content reject certification.

Highest completed bare-CI row, {6}:

|Band|Native/eV|Published bare CI/eV|Signed difference/eV|
|---|---:|---:|---:|
|1→0|94.78670464|94.73|+0.056705|
|2→1|93.56209542|93.55|+0.012095|
|3→2|92.45236116|92.46|-0.007639|

[Reference: Reitsma et al.,PRA114,L020802(2026),TableI](https://doi.org/10.1103/v557-qqql).
These are matching published theory rows, not final MBPT results or experiment.
The2→1 gA centroid changes by-5.415682eV from{3} to{6}.
CI convergence is nonmonotone: the3→2 centroid changes about−4.80eV between
{4} and{5}. It is unsafe to infer saturation from one stable band.
The centroid is sum(gA*photon energy)/sum(gA); the plotted32eV distribution
separately uses gA*exp(−Eupper/32eV). Its chosen0.03nm Gaussian width is not a
paper parameter or plasma broadening model;0.01/0.03/0.10nm sensitivity is
retained. This distribution does not supply opacity or conversion efficiency.

The separate [reduced source model](EUV_SOURCE_MODEL.md) now consumes these
native levels/lines as absolute emission and Einstein absorption, with an
all-level fixed-charge LTE partition, slab transport, Drude laser deposition
and pressure-work dynamics. It retains preheat controls and an energy ledger;
it does not confer multi-charge or experimental CE acceptance on this CI run.

All1,628,730 computed nonzero E1 lines are retained. Three forward
adjacent bands contain1,614,500 lines versus
the paper's1,613,349. This discrepancy remains unresolved; reverse-band and
3→0 lines remain explicit. No strength cutoff or relabeling forces agreement.

## Exact storage and numerical reuse

`methods.ci_blocks.solve_ci_blocks` stores separate J/parity H/C blocks,
retains all roots and checks full H residuals and coefficient Gram defects.
The streamed path shares the ordinary `CIRadialSnapshot`. The optional
`physics.csf.spinor_ci` path consumes native h_eff/g/E_frozen and compiled
`numerics.slater_values`, with at most63 real active spinors. Exporter closed
reference keys are a normal-order partition, not the target electron count;
reference Fock and reference energy are not consumed. Frozen electrons
contract exactly; explicitly opened4p holes remain active.

Magnetic vectors always use sorted orbital keys. Converting into exporter
order requires both bitmask remapping and the exterior-product phase. The
initial Be prototype omitted that conversion and failed18/100 H entries
(maximum0.25871626Ha); the failure and exact repair are retained. Native
complete-H, opened-core-hole and orbital-phase controls now pass.

Sequential Clebsch–Gordan construction carries exact integer M labels rather
than rescanning masks, and skips only final components unable to reach M=J.
The shared one-body tensor supports exact early B†TK root projection and
caller-owned bounded immutable projection reuse. Cache identity contains
ordered CSFs, magnetic/orbital layout and all signed coefficient bytes;
radial operators are freshly applied. No weak amplitudes are screened.

The complete{4} before/after runs both use8 BLAS threads: all186 saved
arrays are bitwise equal. Single measured wall times are769.59/77.48s,
E1 times534.36/25.71s. This is a source-qualified single observation with
concurrent-work/cache effects, not universal scaling or MPI efficiency.
The original global{3} peak was5.13GiB; blocked storage measured0.63GiB.
Highest completed run: 1322.50s including serialization,
peak RSS15.812GiB, explicit cap64GiB.
Global{6} H alone would require107119689800bytes; block H+C payload is
13491842768bytes, before solver/CSF scratch.

## Verification, remaining work and documentation impact

Highest complete-H residual is1.091e-11Ha;
coefficient Gram defect1.221e-14.
A separate same-source angular audit checks every115715 CSFs over3832
configuration/J groups: metric defect2.31e−14 and highest-weight leakage
7.54e−14, with exact J-span counts. Every saved highest-run photon/rate/f/
amplitude and centroid is independently replayed. All199 arrays of the
current frozen{3} run are bitwise equal to the prior spinor gate.

Two affected panels pass300 checks with3 optional EXPT-runtime skips and1
existing strict `fac_frozen` helium radial-protocol expected failure. The
portable source's37 checks pass. The original shared-cache HFS serialization
failure and isolated-cache revalidation remain recorded; no physics tolerance
was relaxed. Own-file Ruff passes. The shared dirty tree's tracked-only
documentation check still fails; changed working-tree links are checked
separately without staging unrelated work.

Remaining physics: complementary-core one/two-body MBPT, necessary effective
three-body/subtraction terms and explicit P/Q double-counting exclusion;
consistent correlated E1, radial/virtual convergence, matched Breit/QED and
experimental validation. The finite one-body `ci_mbpt_solve` interface is
not a completed particle-hole CI+MBPT implementation. The reference ZIP is
not a platform-accepted full-paper solution; Docker remains an unbuilt recipe.

Owning CI/transition/architecture/testing/research guides are updated for
this increment. General CC/SD/MBPT and sealed Yb results retain their physical
contracts. Trigger: dominant angular construction → carry exact quantum
labels and reuse bounded signed projections → preserve every state and
qualify full matrices before timing. No new general policy is introduced.
