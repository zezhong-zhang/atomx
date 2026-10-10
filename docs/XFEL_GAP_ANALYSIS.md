# XFEL Capability Gap Analysis — atomx vs XATOM / FAC / Cowan (2026-10-02)

Repository availability: paths marked “local evidence” are outside the tracked
checkout. They retain the original evidence location and are not GitHub links.

Goal: enable atomx for XFEL (x-ray free-electron laser) calculations —
sequential multiphoton ionization of atoms under intense pulsed x-ray
irradiation, transient-configuration spectroscopy and scattering, and
the observables of single-particle imaging.

Grounding: the vendored XATOM source (xraypac, /tmp build) is read at
module level; atomx claims are taken from `src/` and the 2026-10
validation campaign (see trace.md and the gates).

当前 Ne q/r 谱位见完整3200Ha尾部、Breit约定与极点判别 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)：同模型全修正 q=908.193003810、r=908.065718918 eV，计算−各自实验 -8.896190/-13.881082 meV。完整1600→3200Ha三态rank1+rank2使两线变化 +5.914650/5.874690 meV；全部新singles为 0.014556/0.014465，再补全部新rank2为 +5.900094/5.860225 meV。三态完整CI、旧态、singles、DC/DCB及五算符/Q谱隙独立重放通过原1e−10Ha残差及有符号身份门；旧271轨道逐位保持。局部1meV增量门仍未通过，测前p=1.5预测的实际门状态保留，条件代理不作尾项界。同空间实际Breit增量相对2017印刷分项 -4.018936/-4.490156 meV，纯DC分项差 3.872315/2.097007 meV；跨模型差不是唯一物理误差预算。f/g计数预检不等于能量，完整高κVP覆盖仍须先修复；>3200Ha、n>6、径向密度、正能/高n rank3、轨道/投影SCF和开放共振尚未全部闭合。旧核/同位素、点模型极点及尾项代理不加到新谱位；下方编号节保留各自历史模型。

本轮Breit与复极点判别 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)的完整3200Ha固定CI Breit非线性跃迁响应 +0.241308/0.291141 meV已含；23项静态/复算符和45项有限频率精确源核验通过。DCB态Breit期望与开关增量定义的差为 0.241245/0.292472 meV，两个定义独立保存；2017印刷分项取值态未完全核实，不因更接近而选择处方。同原生DF完整n=2核空间的频率−静态实际响应+0.040845/+0.042963meV及独立全矩阵重放通过；2012参考组态处方在完整3200Ha态的一阶响应+0.045086/0.046889meV，指定嵌入算符有序本征值界0.071999/0.070425meV，不足以在该处方内解释约4–5meV分项差，不约束一般QED/SCF/投影或跟踪根排序。R4完整小池rank2使束缚锚失败，关掉Breit仍失败；R4/8/12的N2 DC及R12 DCB四态恢复束缚门，支持外盒/有限谱干预而非纯墙面误差界。R12八个完整谱与对称角度独立重放通过，q/r电子宽度4.379/5.665meV，角度复能跨度0.000101/0.000104meV；rank1→rank2跃迁仍移动−31.499/−32.138meV，物理极点未认证。实验另报c.g.的权重/估计量未验证，不当作已匹配中心误差；q−r中共同基态严格抵消。

关于加QED后是否再迭代，当前完整800→1600Ha、QED重解与f/g预检 (local evidence: `xfel_uv_tail_continuation_2026_10_06/README.md`)在完整1600Ha空间直接解QED-off/on，并核对QED-off態的一阶期望：非线性q/r响应+0.039504/+0.039272meV已含在当前谱位。轨道及正能投影保持冻结，此控制不是完整轨道/密度QED-SCF或负能支/投影反馈上界。原CI响应审计 (local evidence: `xfel_qed_response_audit_2026_10_05/README.md`)中的旧n4/5约+0.017673/+0.017567meV保留原模型范围；每次完整空间新增均重新求解全部五算符H，不能用小CI响应或残差认证轨道驻值。 原生与当前QED应用矩阵差的有限空间Weyl跃迁界为0.004915755meV，只约束同模型数值应用，不能替代QED模型或自洽/投影误差界。

另有直接QED单电子轨道探针 (local evidence: `xfel_qed_orbital_probe_2026_10_05/README.md`)：全部221原轨道内保持Fock势与无对投影固定，完整同κQED矩阵入哈密顿量重求单电子本征态。参考占据组合的上态减基态非线性响应q/r为+0.017853/+0.017850meV，λ=0.5/1与二阶和相容；不支持简单冻结Fock轨道重求补250meV。该量不是相关三态能量，不可再加到现有CI谱位；密度反馈SCF、截断CI轨道优化及投影外方向仍未测，完整QED轨道SCF上界仍未知。

13项因素筛选 (local evidence: `xfel_factor_screening_2026_10_05/README.md`)保留旧n4/0固定态的一阶控制：点核−Fermi跃迁约+0.522meV，半径同形±5%约0.05meV；条件自然同位素反冲质心约+0.19meV，纯22Ne约+2.03meV，属于旧状态控制。束缚rank3/n5/n6 (local evidence: `xfel_bound_rank3_n6_2026_10_05/README.md`)直接约束小束缚rank3与n5/n6；按n6校准的n⁻³核心穿透模型给n≥7两线条件代理约+7.764/+7.809meV，渐近起点未认证，不是界。第一性1s²电子对高k/高L代理提出高能双虚轨道候选，后由完整200→400→800Ha与新单激发对照 (local evidence: `xfel_high_energy_pair_cutoff_2026_10_05/README.md`)直接测量；旧百meV代理与形式高L约55meV不是可相加修正。两步条件尾部预测经完整800→1600Ha、QED重解与f/g预检 (local evidence: `xfel_uv_tail_continuation_2026_10_06/README.md`)的新实际增量检验；p=1.5和有效p的>1600Ha数字仍为条件代理，未认证有限s/p/d渐近性或径向密度，不据此修正谱位。旧一般rank1的100→200Ha约−42meV及旧单4f约−0.2meV不能约束双虚轨道尖点尾部。 最新1600Ha态的内部反冲期望一阶质量控制见完整800→1600Ha、QED重解与f/g预检 (local evidence: `xfel_uv_tail_continuation_2026_10_06/README.md`)：条件自然质心+0.190833/+0.192324meV，纯22Ne相对20Ne为+2.031936/+2.047816meV；未含同位素半径和非线性CI响应，实验组成未知，不加到生产谱位。

第一性原理诊断 (local evidence: `xfel_first_principles_diagnosis_2026_10_05/README.md`)完成同 n4 的 DC→Breit→QED→反冲三态路径与15态完整矩阵重放。相对2017印刷分项，纯 DC 已差约−273.5/−274.2meV，Breit增量再差约−18.5/−18.8meV，支持优先定位相关/空间表示；这些跨模型差不是唯一可加物理误差预算，也不足以把全部共同偏差归到基态。该分项对照对应旧n4/5空间；新谱位见上方最新完整三态对照，复极点仍未认证。

核心电子对与条件零点定位 (local evidence: `xfel_first_principles_localization_2026_10_05/README.md`)在同轨道/同纯DC类氦核补入0–5Ha完整双外部空间，能量下降88.689meV，完整Q谱隙88.012Ha，属于闭通道相关。最新n4三态的相对论文偏差随有限类氦比较零点同移121.664meV，跃迁偏差严格不变，因而不能唯一分配给基态。此低电子数控制指导了后续三电子直接干预 (local evidence: `xfel_positive_rank2_closure_2026_10_05/README.md`)；类氦能移本身不作为三电子谱位修正。

## 1. What XFEL physics needs

1. **Sequential multiphoton ionization cascade** — during one pulse an
   atom absorbs N photons; after every absorption/decay event the
   electronic structure is re-solved for the new configuration and all
   rates (PI, Auger, fluorescence) recomputed.  This is XATOM's
   `track_elec_config` core (Young et al., Nature 466, 56 (2010);
   Son–Young–Santra, PRA 83, 033402).
2. **Electron-by-electron Monte Carlo** over the pulse for fluence-
   dependent charge-state/spectral yields (`xatom_mc`, `xatom_mc_fly`).
3. **Rate-equation occupation dynamics** over the configuration graph
   (`yes_occupation` mode).
4. **Secondary channels**: direct double ionization / shakeoff
   (`xatom_shakeoff`: sudden orbital overlap; independently sourced direct-double data required), electron-impact ionization by
   photoelectrons (`impact_ionization`), Compton (`compton.f90`),
   two-step / REXMI resonant multiphoton absorption (`xatom_twostep`).
5. **Transient-config imaging observables**: elastic scattering
   (f0(q), anomalous f1/f2) and spectra per transient charge state
   (`xatom_imaging`, `spectra.f90`).
6. **Pulse model**: spectral photon fluence, temporal envelope.

## 2. Historical capability matrix (2026-10-02; current status in §7–9)

| Capability | XATOM | FAC | Cowan | atomx today | Gap |
|---|---|---|---|---|---|
| Structure (avg config) | HFS | DFS/DHFR | HFS+fit | **DHF/MCDHF/CI** (stronger) | — |
| Subshell PI σ(ω) | ✓ | ✓ | — | ✓ validated vs FAC 4-23% (E1+finite-ω) | near-edge resonances (Fano) partial (`fano.py` exists) |
| Auger rates | LS chain | jj/J-resolved | UTA tools | **3 tiers** (LS-exact 1.0000 vs XATOM; UTA-0 vs FAC; determinant-CI 0.89-1.11 of FAC) | manifold-H dressing (10%) |
| Fluorescence A | ✓ | ✓ | ✓ | E1 validated; M1/E2 partial | M1/E2 vs FAC panel |
| RR / DR | — | ✓ | — | ✓ validated | — |
| EII / CE | ✓ | ✓ | — | DW + BEB | high-Z DW basis limits |
| Shakeoff / direct double | ✓ sudden-overlap shakeoff; direct-double dataset unverified | — | — | **missing** | new |
| Compton profile | ✓ | — | — | planned (matrix TODO) | new |
| Two-photon / REXMI | ✓ | — | — | two_photon tests exist (bound-bound) | XFEL sequential-resonance driver |
| **Sequential-PI cascade (on-the-fly re-solve)** | ✓ `track_elec_config` | — | — | **missing** — `plasma/cascade.py` is a DECAY cascade (FAC SetCascade analog), not a photoionization cascade | **the core gap** |
| MC over pulse | ✓ `xatom_mc(_fly)` | — | — | missing | new |
| Occupation rate equations | ✓ | CRM (Te/ne) | — | CRM chassis ✓; **PI rates use Kramers/hydrogenic placeholders** — the real `scattering/photoionization` σ(ω) is not wired in | wiring |
| Charge-state/scd tables | ✓ | ✓ | — | CRM produces them | wire real rates |
| Elastic scattering f0/S(q) | ✓ | spinor integral | — | ✓ (in flight, Waller-Hartree) | per-transient-config |
| Anomalous f1/f2 | — | via PI (same formalism) | — | ✓ **2026-10-02**: f2=ωσ/(4πα), f1 KK, TRK gates | discrete lines into tabulation |
| Imaging (per-config diffraction) | ✓ | — | — | ingredients only | driver |
| Speed per configuration | ~ms (HFS+analytic) | s | fast | **seconds-minutes** (B-spline SCF) | fast path + caching |

## 3. Historical gap ranking (2026-10-02)

1. **Photoionization cascade driver** (the XATOM core): configuration
   graph, per-configuration rate table (PI per subshell + Auger +
   fluorescence), on-the-fly structure re-solve with memoisation,
   pumped by a photon fluence spectrum; outputs = charge-state
   distribution, electron/photon spectra, transient populations.
   Everything atomx needs for the RATES already exists and is
   validated — this is an integration increment, not new physics.
2. **Wire real cross sections into the CRM chassis** (replace
   `_kramers_photoionization_sigma_cm2`-class placeholders with
   `scattering/photoionization` + today's Auger tiers + einstein_A).
3. **Fast configuration-solve path**: engine="shooting" + local
   exchange is the natural XATOM-speed analog; needs a per-config
   timing benchmark vs XATOM (target: ≤10× XATOM for the same config).
4. **Shakeoff / direct double ionization**: local XATOM uses the sudden
   orbital-overlap approximation. Its `xatom_shakeoff_dat` generates data
   from SCFs; it is not a bundled fitted direct-double dataset. Any
   additional direct-double table needs explicit source attribution.
5. **MC over pulse + occupation dynamics** on the cached graph.
6. **Transient-config observables**: per-config f0/f1/f2 (both modules
   already parametrised by orbital sets — plug the cascade's transient
   configs in) → imaging/diffraction signals.
7. **REXMI / two-step resonance absorption** (uses the DR/manifold
   machinery from today's campaign).
8. Cowan: no gap for XFEL per se — atomx's CI/spectroscopy subsumes
   Cowan's level/rates role with relativistic physics; keep Cowan as
   an external cross-check anchor only.

## 4. Historical first-increment proposal (2026-10-02)

`src/atomx/xfel/cascade.py` — deterministic (rate-equation) sequential-
PI cascade over an explicit configuration graph:

* input: element/config, photon fluence Φ(ω) (integrated over the
  pulse), channel rate providers (PI per subshell σ(ω); Auger LS-chain
  tier; E1 fluorescence);
* loop: populate configuration graph breadth-first — each config:
  solve structure ONCE (memoised), tabulate rates, branch on
  (photoabsorption | Auger | fluorescence);
* outputs: charge-state distribution vs fluence, total electrons,
  per-config Auger/photon spectra, transient config table;
* validation anchor: reproduce XATOM's published Ne charge-state
  yields vs fluence (Young 2010 Fig. 1 shape; regenerate numbers with
  the vendored binary `-occupation`-style run), gated at O(20%)
  initially (orbital-model difference), tightening later.

Defer to later increments: MC trajectories, shakeoff, REXMI, pulse
time structure, imaging observables.

## 5. Honest risks

* **Per-config cost**: atomx SCF ≈ 10²-10⁴× XATOM's HFS.  Mitigations:
  shooting-engine fast path; frozen-core Δ-SCF after the first solve
  of a charge state; graph caching (many configs share the solve);
  subshell-rate reuse across configurations that differ by one
  electron.
* **Near-edge resonances**: XFEL pulses sit ON resonances (REXMI);
  the smooth-σ cascade underestimates there — the resonance-aware
  increment is required for photon energies near thresholds.
* The Kramers placeholders in `crm.py` must not silently coexist with
  real rates — route the CRM through the same providers.


## 6. Complete native Ne rate graph (2026-10-03)

The actual-rate rebuild and threshold report (local evidence: `xfel_real_rates_2026_10_03/README.md`) now provides 134 independently relaxed Gu jj configurations, including zero-occupation hole orbitals, with native daughter energy differences. All 378 occupied PI edges and 250 nonzero Auger edges close on the graph. `build_atomic_rate_table` is the production entry point for this model; the older skeleton-sharing provider is a different approximation.

Nine neutral-Ne PI points at 1028/1714/3428 eV agree with archived FAC references within 0.0343%; native Ne KLL totals are 1.0521 times a fresh XATOM reference. At F=1000 photons/a0², native Delta-SCF MC gives mean charge 4.8285 versus archived XATOM 5.9322. Changing only PI thresholds to orbital energies gives 5.9846: the dominant discrepancy is the near-edge threshold convention, with a concrete Ne4+ K threshold above 1 keV in the native model and below it in XATOM. This is a diagnostic, not a fitted replacement for native energy conservation.

The compiled MC reproduces the original seeded bin-wise observables on the new real table. At this increment finite Auger lifetimes, radiation competition, shakeoff and resonances remained separate physics work; their subsequent implementation is described in §7–8.

## 7. Finite pulses, sourced secondary channels and imaging (2026-10-03)

The extended XFEL implementation and live checks (local evidence: `xfel_extended_2026_10_03/README.md`) now add sparse finite-lifetime population dynamics, native E1 fluorescence, exclusive single-shake photoionization, explicit resonant excited-state pumping, and a transient atomic imaging driver. The native Ne graph has 135 states including the bare ion, 264 radiative channels and 603 shake channels. At 100fs and F=1000 photons/a0² its mean charge is 4.90643, compared with 4.83421 for the finite-Auger-only graph. This does not remove the threshold-model discrepancy identified in §6.

CRM defaults to an explicit native PI cross-section provider, uses that same provider in Milne RR, and fails when it is absent. Kramers requires an explicit model selection; native plus Kramers input is rejected. Resonance pumping covers isolated incoherent lines with bandwidth and weak-pumping gates, not strong coherent Rabi dynamics or Fano interference. The live native H test checks sequential resonance-assisted ionization. At this historical increment the resonance graph was supplied manually; §8 replaces that step with native E1 graph preparation within the supplied orbital basis.

Imaging computes mean(|f|²) and its population variance. The live Ne bank uses actual per-jj f0, optical-theorem f2 at the exact photon energy and continuum-only f1 from analytic piecewise-linear principal-value integration. Bound-bound dispersion is explicitly absent in that live bank. Direct-double ingestion checks attribution, units, graph closure, thresholds, duplicates and overlap with shakeoff; no unverified fitted dataset is enabled. XATOM exported radial-wavefunction checks distinguish transcription error from the Gu/HFS orbital-model difference.

## 8. Native preparation, broadband response and reuse (2026-10-03)

The production API report (local evidence: `xfel_native_api_2026_10_03/README.md`) moves native
resonance graph construction and state/spectrum imaging into
`atomx.xfel.NativeXFELModel`. It checks archived SCFs, retabulates true PI
over threshold-split Gaussian spectra, splits single-shake branches with
energy-resolved double thresholds, and reverses native E1 edges with
configuration statistical weights and both endpoint lifetimes. Imaging
includes signed discrete causal poles and separately integrates spectral
mean amplitude and intensity. These capabilities supersede the manual
graph and continuum-only imaging limitations of §7 for the supplied basis.

Real Ne9+ preparation gives 5 states and 4 E1 lines. At the native
1022.129845 eV K resonance, 100fs/1eV/F=1e15 photons/cm² gives bare probability
5.22634e-8 versus smooth-model zero; doubled fluence scales 3.94435. Spectral
quadrature order 32→64 closes q=0 ground intensity to 6.03e-13 relative.
Fixed-model fluence scans reuse spectral banks: measured 0.150–0.180s per
cached run versus 6.20–6.27s recomputing, with exactly equal paired outputs.
231 regression tests pass; one slow test is deselected.

Gu remains a configuration-average control rather than the canonical
MCDHF CI Hamiltonian-average relaxation. Missing higher orbitals, Fano
interference, coherent strong-field dynamics and time-dependent spectral
chirp remain explicit gaps. The driver does not infer an unsourced
direct-double fit. No new external FAC/XATOM accuracy claim is made here.

## 9. Explicit magnetic state coherence and occupation correction

The coherent XFEL report (local evidence: `xfel_coherent_2026_10_03/README.md`) adds
`CoherentGaussianPulse`, `CoherentSystem` and a native one-electron
adapter. A common orthonormal orbital basis supplies signed E1 amplitudes;
the same amplitudes determine the radiative jump operators. Native
continuum PI now resolves magnetic sublevels for a specified spherical
polarization. The configuration-average incoherent solver retains its
strong-pumping gate.

Real Ne9+ gives 11 magnetic states. At 1022.129845eV, 10fs, 1e17 photons/cm²,
the peak Rabi energy is 0.423603eV and the bare fraction is 2.43309e-4.
The density trace closes to 3.22e-15; half-time-step refinement changes
populations by at most 7.33e-14. Independently relaxed 1s/2s overlap
6.89495e-4 is replaced by a common orthonormal basis. K radiative rates
agree with the native graph; the tiny 2s→2p1/2 rate differs by 0.01569%
because its source orbital basis changes.

The §8 retabulator had an extra occupation multiplier on an already
occupation-weighted PI provider. It is removed and q=2 is tested.
Recomputing all 134 native Ne configurations/378 PI edges at1keV exactly
matches the original archive. The earlier q=1 Ne9+ example is unchanged.
272 tests pass and one slow test is deselected at this historical increment.
The multielectron bound-state increment is described in §10. Continuum
Fano/memory effects, SASE ensembles and strong-field imaging remain
separate work; near-threshold Markov PI is explicitly
rejected by the new native coherent adapter.

## 10. Signed MCDHF CI coherence

The multielectron CI report (local evidence: `xfel_ci_coherent_2026_10_03/README.md`) adds
`MCDHFCoherentModel` and signed MCDHF transition amplitudes with explicit
CI column indices and J values. One shared orthonormal orbital basis
defines the drive and E1 emission. Exact degenerate emission paths share
a bath; unresolved distinct lines and inconsistent rotating-frame jump
ladders fail explicitly. The caller must choose `bound_only=True`.

Actual two-electron Ne8+ gives five CI levels and 13 magnetic states.
The stronger E1 line is 922.873585 eV with A=9.40146e12/s. At 10 fs and
1e17 photons/cm² its peak excited population is 0.980506. Full-density
step refinement closes to 9.13e-12 and CI phase covariance closes exactly.
The orbitals use the existing correlated ground-CI optimization; no new
excited-state EOL relaxation or FAC accuracy result is claimed.

This resolves the signed multielectron bound-state Hamiltonian gap.
Multielectron PI/Auger on the same resolved CI basis is still required
for coherent REXMI ionization. Configuration-average loss is not mixed
into this adapter. M1/E2, Fano/memory, SASE, coherent imaging and general
nonsecular reservoirs remain outside its scope.

## 11. Resolved two-electron CI continuum losses

The CI loss report (local evidence: `xfel_ci_losses_2026_10_03/README.md`) adds
`TwoElectronCoherentModel`: N=2→1→0 on one magnetic density basis, with
signed Dyson PI and full Dirac–Coulomb Auger amplitudes (including h_D
when a bound spectator is unchanged). Occupation and J weights are not
applied again. The parent uses MCDHF CI optimization; the ion uses frozen
common orbitals, and the continuum uses its spherical direct Hartree
field projected out of the retained bound space. This explicit model
requires `corrections='none'`, far-from-threshold central-frequency PI
and resolvable secular electron baths.

Actual Ne8+ 1s2/1s2p resonance at 923.301084 eV produces a final Ne9+
probability 2.36342e-4 at 10 fs / 1e17 photons/cm². Auger and subsequent
1s-ion PI are closed in that example. A separately labelled frozen 2s2
control has a full-H Auger width 1.44508e14/s vs live FAC 1.8063e14/s:
**20% external difference remains unresolved**, despite 2.01e-9 continuum
mesh convergence. Omitting the one-body term gives a misleading closer
value 1.92487e14/s and is retained only as a rejected diagnostic.
Ground-optimized correlation 2s creates an unbound ionic level at
+26.1871 Ha, which fails explicitly instead of entering the bound graph.

This supersedes the N=2 continuum gap in §10; general N-electron PI/Auger,
excited-state EOL/relaxed-ion bases, interchannel/Fano/memory, Breit/QED,
SASE and coherent imaging remain outstanding. The implementation does
not claim a generally certified REXMI yield or reuse configuration-average
loss tensors as coherent amplitudes.

## 12. Shared Dirac operator and batched continuum preparation

The reuse/performance report (local evidence: `xfel_ci_reuse_2026_10_03/README.md`) moves the
CI adapter's h_D action into the shared physics Dirac object and native
B-spline evaluator. It replaces per-wave cubic interpolation with the
existing Dirac propagator dense-output sampler and collects wave requests
per final-ion field. The physical Hamiltonian and frozen ionic approximation
remain those of §11. Shared Gram matrices and per-snapshot energy caches
avoid duplicate work without mixing atomic identities.

Interleaved nine-trial actual Ne CI preparation benchmarks give 0.257→0.054s
(MCDHF, 4.73×) and 0.477→0.115s (frozen Auger control, 4.17×). These are
channel-preparation measurements; structure and time propagation are excluded.
Same-sampler scalar and batched amplitudes are identical. The strongest full
Ne8+ density trajectory changes by 8.05e-13 vs the historical cubic archive.
The independent differential/Galerkin Dirac, CI sign, occupancy and conservation
anchors pass. This does not resolve the 20% external Auger residual in §11.

## 13. Fixed CI level-average orbital relaxation

The state-average report (local evidence: `xfel_state_average_2026_10_03/README.md`) adds
`CIStateAverage` to the shared B-spline MCDHF optimizer, for arbitrary
electron number. It reuses the exact CI Hamiltonian angular weights and
analytic orbital gradient, accepts symmetry-local root selectors and records
ensemble populations separately from ground-state densities. The default
ground-only density trajectory replays against §12; no fractional-occupation
or XFEL-only radial optimizer is introduced.

For Ne `1s2 / 1s2p / 2s2`, shared level-average orbitals keep the projected
ion bound and open actual full-H Auger channels. Equal-level ns=100 width is
2.28777e14/s (ns=80→100 change 5.83e-6). Statistical-level width is
2.08514e14/s vs freshly rerun FAC 1.8048e14/s: 15.5% external residual remains
unisolated. These are distinct variational objectives, not a fit to FAC.

The N=3 K-hole space `1s2 2s / 1s 2s 2p` converges under the new objective;
the strongest K resonance changes from 910.041 to 899.031 eV. Line-specific
diagnosis identifies false rejections from using an unrelated global maximum
width: the actual limiting pair's frequency/participating-width ratio is
10.6055 at ns=100. The driver now compares natural FWHM
hbar*(Gamma_upper+Gamma_lower), retaining the separation threshold of 10
and the rejection of genuinely unresolved pairs. Its 26 magnetic-state
bound-only pulse runs with a 0.85974 resonant peak occupation and 1.87e-13
full-density refinement error. This is resonant bound excitation, not a
REXMI ionization yield. At that stage, general N→N−1 loss tensors, shared multi-charge relaxation,
continuum exchange/coupling and nonsecular dynamics remained open (current
finite-space chain and explicit cluster approximations are described in §15). This update
partially addresses the EOL item in §11 and does not certify general REXMI.

## 14. Explicit CI preparation and the atomic-state design

`CILevel(J2, parity, root)` and `CILevelPopulation` now share symmetry/root
resolution and degenerate-subspace checks with the orbital EOL objective.
`MCDHFCoherentModel(initial_state=...)` accepts either type; the joint
`TwoElectronCoherentModel` accepts the same parent preparation. Physical level
weights are distributed over M without an extra statistical factor. Preparation
does not re-optimize orbitals or alter loss/drive operators; the optimization
ensemble cannot be passed as the initial population.

[Atomic state design](ATOMIC_STATE_DESIGN.md) separates CI space, optimization
objective, property state, dynamical preparation and energy reference. It also
specifies remaining state-view, multi-charge relaxation and basis-transformation
work. The existing CI continuum is explicitly frozen final-ion direct-Hartree
Dirac DW followed by retained-bound-space projection; no nonlocal exchange,
coupled-channel continuum or complete self-consistent scattering is implied.
Current general-N progress and outstanding physical restrictions are in §15.


## 15. General-N signed CI resonance/loss chain (2026-10-04)

`MultiElectronCIChannels` reuses complete fixed-N jj CSF enumeration, the existing
Slater–Condon Hamiltonian and radial kernels. Every lower electron sector is
complete over the retained parent orbitals; signed Dyson PI and full h_D+V
Auger include spectators and magnetic interference. `MultiElectronCoherentModel`
assembles explicit all-charge magnetic density dynamics and consumes prepared
channel snapshots to reuse angular/DW caches. Parent EOL and frozen residual CI
share orthogonal orbitals; this is not multi-charge orbital optimization.

Opt-in photon/electron cluster reservoirs support nearby frequencies while
keeping exact Hamiltonian splittings and explicit 1e−3 spectral/tensor gates.
The geometric photon option preserves individual Einstein rates but omits the
radiative Hamiltonian renormalization. Distinct final ionic levels remain resolved
electron reservoirs, so their loss-generated coherence is omitted. Default
secular gates, threshold and pulse/table identity checks remain.

Ne7+ N3 at 899.0278 eV, 10 fs and 1e17 photons/cm² now shows actual REXMI charge
transfer: ns100 final charge 7.18726049 versus 7.00180303 without E1 driving,
using 82 magnetic states. Charge flow closes to 1.13e−8; full-density time
refinement is 1.01e−11 and CI phase covariance is exact. Generic N=2,3,4
independent fermionic anchors and actual N4 PI dynamics pass. However the
four-electron K resonance probe still rejects a low-energy radiative cluster
whose spectral coefficient error is 0.00175453. This restriction is preserved;
arbitrary-N resonance dynamics are not certified.

Fresh FAC N3 AITable ratios span 0.5769–1.4322 after matching J/parity/energy root
and level-rate conventions. Different optimization objectives and continuum
models remain unisolated. Full results, replay, sources and failure probes:
general-N CI REXMI archive (local evidence: `xfel_multielectron_2026_10_04/README.md`).
Outstanding: jointly relaxed charge sectors, tracked shell/root identities,
nonlocal exchange/coupled/Fano/threshold memory, broader vacuum reservoirs with
Hamiltonian shifts, coherent secondary losses/imaging and large-space scaling.

## 16. Absolute reservoir budgets and sparse magnetic operators (2026-10-04)

Explicit `geometric_bounded` radiation and `clustered_bounded` electron options
now bound finite-time generator substitutions against specified same-cluster
references, with a combined 1e-3 trace-norm budget rechecked at every solve.
Legacy gates are unchanged. Ne N3 full density exactly reproduces the archive:
radiative bound1.61e-8, combined bound1.54e-4. These are conditional Markov-model
comparisons, not bounds on microscopic bath physics, Lamb shifts or memory.

N4's first old relative electron failure is a6.52% change of an almost-zero PI
tensor (squared norm7.07e-58 cm²). The absolute mode keeps that branch but still
rejects the complete 201-M-state chain: conservative combined bound0.865985>0.001,
dominated by strong Auger clusters. It is not measured trajectory error. N4 K
resonance remains uncertified; no budget change/channel removal is justified.

Exact CSC jumps and diagonal+sparse E1 action activate above128 M states. Actual
N4 operator buffers fall from2.451GB to3.273MB; paired Hamiltonian-action-only
microbenchmark is8.89x, maximum difference6.66e-16. Small-basis historical
arithmetic and full-density replay remain unchanged. Derivation, independent
oracles, numeric diagnostics and scope:
finite-time reservoir archive (local evidence: `xfel_radiative_bound_2026_10_04/README.md`).

## 17. N4 residual comparison and exact coordinate closure (2026-10-04)

The N4 conservative failure in §16 remains the default outcome. An explicit
trajectory-dependent comparison now evaluates the signed original-energy
reservoir residual on the occupied density, including the bounded arithmetic
radiation correction. Its analytic inequality is evaluated with two time
resolutions; observed density/integral changes must each stay below1e-8 before
applying the unchanged1e-3 budget. This numerical convergence check is not an
interval enclosure of global ODE error or a microscopic bath certificate.

The N4 input also needed correction: assigning photon1 to every excited parent
root erroneously includes low-valence lines after adding the hole partner.
Production still rejects that mapping under the existing RWA window. Correct
parent K/valence ladders, explicitly undriven daughters and all retained losses
give ns100 final charge6.5585018714 versus6.0029241964 without E1 pumping.
Reported conditional bound6.59242e-7, sampled CP original-energy reference
trace-norm difference7.93e-10, and signed charge-flow closure~2e-13. CI phase
covariance is exact; radial and DW refinements pass in this same retained space.

Boolean nonzero generator closure reduces this initial state's40,401 density
coordinates to98 without amplitude truncation. Optional BDF uses a complete
real Jacobian for the nonholomorphic residual norm. Output density/positivity
cost remains quadratic; default small-basis DOP853 arithmetic is preserved.
197 relevant tests pass; complete historical N2/N3 densities and inherited
reference hashes are checked separately. This supersedes the absence of a
validated N4 trajectory in §16 for this specific parent-pumping input only.
Joint-charge orbital relaxation, nonlocal exchange, Fano/memory/vacuum shifts,
coherent secondary losses, active-space and daughter-pumping validation remain.
See the residual comparison archive (local evidence: `xfel_residual_certificate_2026_10_04/README.md`)
for the input correction, proof, numerical limits, data and replay commands.

## 18. Shared-orbital charge-sector CI optimization (2026-10-04)

The joint-charge item in §17 now has an explicit implementation:
`CIChargeStateAverage` uses fixed charge/level weights, shared orthonormal
B-spline orbitals and the existing CI/RDM/Dirac gradient optimizer. Parent
physical density, optimization mean electron number and pulse preparation
remain distinct. Independently rebuilt ionic spectra agree exactly.
Separate charge-specific orbitals, overlap/biorthogonal operator transforms
and dynamic orbital relaxation remain open.

Equal-charge/equal-root N3/N4 ns100 examples yield final mean charges
7.01112950/6.67985377, with unchanged residual gates and conditional bounds
8.25e-7/2.33e-6. N3's previous0.1 excess-charge target fails explicitly:
the strongest resonance's complete Auger rate drops5.49e12→2.37e11/s,
despite stronger E1/PI, through increased relative one-/two-body cancellation.
The objective changes the orbital model and is not fitted to rates or yield;
this sensitivity is not evidence of improved experimental accuracy.

Prepared-spectrum RDM+gradient medians6.187→3.676ms and9.337→4.011ms agree
with independent sector gradient sums to1.78e-15. This excludes CI builds,
DW, SCF and dynamics and establishes no overall XFEL speedup. Default N2/N3
full-density replay and inherited archive/reference identities are separately
checked. Nonlocal exchange, Fano/memory/vacuum shifts, coherent secondary
channels, active-space and all-daughter validation remain.
See joint charge averaging (local evidence: `xfel_charge_average_2026_10_04/README.md`).

## 19. Cross-charge CI snapshot reuse and retained-space sensitivity (2026-10-04)

Joint-charge trials and continuum providers now share the existing signed radial
CI evaluator within immutable orbital/geometry/nuclear snapshots. External
mutable inputs retain exact content and nuclear-value guards. A generic sampled
nuclear operator batches identical quadrature multiplication; it is not an
interpolated continuum field. Shifts/Breit, weak amplitudes and defaults retain
separate contracts. Historical N2/N3 and joint N3/N4 ns60 full densities replay
bitwise, with inherited evidence read-only.

The n=1,2 complete N3 parent has34 CSFs/120 magnetic states. A same-kappa1s/2s
rotation changes its spectrum by1.56e-13Ha, versus0.433Ha in the restricted
8-CSF space. Keeping the same orbitals but expanding CI changes the strongest
fixed-K-window Auger rate2.37e11→1.64e13/s; full-space joint reoptimization gives
1.39e13/s. These comparisons distinguish space expansion from relaxation and do
not imply root-character tracking or improved experimental accuracy.

Paired ns100 all-charge CI trial medians show1.024x for N3 with8/9
pairs faster and1.494x for N4 with all9 pairs faster. This includes snapshot preparation,
all matrices and diagonalizations, excludes SCF/DW/dynamics, and supersedes the
initial slow cache implementation. Complete-space XFEL dynamics, higher-n
convergence, nonlocal exchange, Fano/memory/vacuum shifts, coherent secondary
losses and general heavy-atom/daughter-pumping validation remain open.
See the snapshot and space archive (local evidence: `xfel_ci_snapshot_2026_10_04/README.md`).

## 20. Complex DC/DCB prerequisites and spectroscopy gates (2026-10-04)

The separate complex CI path now combines actual dilated DKB P/Q, an explicit
positive nuclear-Dirac spectral projector, complex Coulomb Slater/Yk integrals
and the complete signed fixed-N CSF operator. Explicit `include_breit=True`
adds both Gaunt and static retardation through shared physical kernels, not a
scalar shift. Actual N1/N2/N3 representation and physical pair checks pass;
this is not yet a general-N certified open-channel resonance model.

Analytic low-energy spinor-derivative and local-Dirac current-continuity gates
needed simultaneous origin/grid refinement. Tiny eigensystem residuals alone
missed these errors. All original physical thresholds and failed diagnostics
remain archived. The graded DC40 candidate passes the measured angle/count
checks but its bound-state imaginary anchor still fails. Point-nuclear DCB
has its own retained-count sequence and cannot be compared as if it were the
finite-core literature Hamiltonian.

The static Ne7+ q/r experimental discrepancy, complex-pole stationarity,
radial/box/angular/continuum-coverage convergence, finite-core alignment,
frequency-dependent interaction, QED/recoil and measured XFEL/detector response
remain outstanding. No new experimental, Auger-width, yield or SOTA certificate
is issued. Full evidence: complex static Breit (local evidence: `xfel_complex_breit_2026_10_04/README.md`).

## 21. Exact physical symmetry reduction (2026-10-04)

Static real/c-product radial permutation identities now compile into the
shared signed CSF tables, with explicit opt-in for generic evaluators and
default use in the owned complex DC/DCB model. Coulomb/Gaunt product and
coordinate exchange, signed retardation-current reversal and exact diagonal
zeros reduce radial columns; only one determinant triangle is compiled.
Transpose completion preserves complex symmetry, not Hermiticity. No state,
small amplitude or Hamiltonian term is discarded, and cache/resource caps stay.

Actual 108/234-CSF static DCB assembly medians improved 2.041/2.382 times,
with full matrix differences at most 1.11e-16 Ha and all matched eigenvalues
within 9.95e-14 Ha. These are assembly timings with fresh radial caches,
excluding radial solve/eig/SCF/XFEL, not whole-run or SOTA benchmarks.
157 regressions pass and one existing strict xfail remains; 15 strengthened
symmetry tests independently pass before final evidence sealing.

The shared RCFP/recoupling modules do not yet eliminate the determinant
intermediate from the general-N complete Hamiltonian compiler. Directional
XFEL response needs explicit magnetic/tensor density and polarization;
existing imaging is spherical. Wigner-D orientation reuse and a certified
non-Hermitian target-pole solver remain future work. All prior spectroscopy,
stationarity, continuum and experimental gates retain their original status.
Evidence: physical symmetry (local evidence: `xfel_physical_symmetry_2026_10_04/README.md`).

## 22. Controlled partial complex spectrum (2026-10-04)

The complete retained static complex CI now exposes explicit numerical
target-spectrum solving. Backend is required: matrix-free projected
determinant/GMRES or dense-CSF shifted LU, both with non-Hermitian Arnoldi.
Partial results retain every CSF/radial state but only requested vectors;
full_spectrum=False is explicit. Shift/nearby roots do not certify physical
poles, experimental assignment or across-angle identity. Inner original
linear residual, original eigenpair residual and c-product gates reject
bad/partial/exceptional results; there is no silent dense fallback.

Actual 234-CSF assembly/eigen measurements show matrix-free is slower and
LU saves only the tiny eigen stage, with no total-stage improvement. A
2460-order fixed matrix reconstructed from old archived C/E gives a 10.37x
LU target solver speedup, excluding reconstruction/assembly/SCF/XFEL.
Its old full Gram defect means this control is not an independently
reassembled physical Hamiltonian. The first shifted-LU linear gate failure
is retained; bounded original-system refinement passes the unchanged gate.
88 focused tests pass; previous physical archives remain unchanged.

Remaining gaps: direct general-N reduced CSF compilation, near-degenerate
full/target subspace treatment and across-angle identity, actual stationary
open-channel poles and continuum/nuclear/experimental convergence. Numerical
target capability does not close those physical requirements. Evidence:
target spectrum (local evidence: `xfel_target_poles_2026_10_04/README.md`).

## 23. P0 local derivatives and stationarity metric identity (2026-10-04)

Explicit complete real-CI cluster Rayleigh--Ritz refinement now removes the
common energy in extended precision. It preserves all roots, the supplied
Hamiltonian and the declared EOL weights; default off. On the same n4 q block,
driver slope spread decreases from 1.3854e-5 to 6.60e-12. Strict physical
S-tangent checks of independent directional matrices and RDM gradients pass
at initial/old endpoint (1.02e-14/1.11e-11 relative). The centered affine
two-root finite-difference panel is not full nonlinear radial certification.

Residual comparisons require the original spectral preconditioner identity:
endpoint restart rebuilds initial gaps and occupations. Same-metric replay
of the old 12-step endpoint now passes the unchanged comparison gate with
1.324e-10 absolute residual difference, without the new CI refinement. New
optimizer diagnostics fingerprint that metric; no update or gate changed.
Original failed different-metric comparisons remain archived.

Continued n4 optimization still fails: 28 steps, line_search_failed, residual
1.54913, estimated energy error 2.39656Ha, q principal overlap squared
0.5000024. Independently tested small descent steps lose majority support;
a larger step changes root but increases the declared energy. This observed
path hits the frozen-reference boundary; it does not prove that no interior
stationary point exists. Do not replace it silently by equal-weight clusters
or a coherent/diabatic target. 67 tests pass; eig-stage refinement adds about
40.8% measured cost, with concurrent optimization/no dedicated CPU.

P0 global path/target continuity and full n4 stationarity remain open, followed
by correlation, genuine open-channel poles/continuum, corrections and joint
experimental forward/statistical validation. No new Ne q/r, width or XFEL
experimental certificate. Evidence:
P0 local gradient audit (local evidence: `xfel_local_gradient_2026_10_04/README.md`).

## 24. Precision literature and operator coverage audit (2026-10-04)

The primary Togawa Table I separates two predictions: its new complex-scaling
theory gives Ne q/r=908.200(4)/908.071(4)eV and splitting128(1)meV;
the Yerokhin2017+erratum column gives908.194(4)/908.069(4)eV and125(1)meV.
Assigning the first column to2017 is a literature-identity error. The downloaded
PDF table page is visually verified and archived; differences and covariance
must not be reconstructed from independently rounded line centres.

The2017 route balances the discretized continuum for dedicated real RCI;
the2024 route explicitly uses complex scaling. Real-line precision therefore
does not require one unique resonance method, and a complex solver does not
certify a physical pole. A balanced-basis same-Hamiltonian comparison would
provide an independent diagnostic; the algorithm is not yet implemented.
FAC supports resolved relativistic CI and optional configuration averages;
GRASP/FAC do not have universal fixed-meV precision grades. AMBiT and pCI
publication restrictions exclude treating their bound-state accuracy as a
ready-made high-precision continuum benchmark.

Current atomx real correction modules exist, but shabaev_local implements only
the local model self-energy part, with nonlocal projectors deferred. Existing
finite complex DC/DCB capability and real correction APIs do not certify the
Ne panel or full QEDMOD coverage. Record requested/supported/applied/skipped
operators, avoid duplicate Breit and cross-Hamiltonian borrowed shifts.
No physical calculation or gate changed in this audit; P0 stationarity, Ne
q/r residuals and full width/yield/experimental requirements remain open.
Evidence: literature audit (local evidence: `xfel_precision_literature_audit_2026_10_04/README.md`).

## 25. Local reference programs for source and executable diagnosis (2026-10-04)

Five official reference sources are newly available under ~/code: AMBiT,
pCI, grasp-rci-qed, QEDMOD v7 and RATIP2012. Existing FAC, GRASP and XRAYPAC
are preserved. Official archive checksums match; all 52/250 extracted regular
files of QEDMOD/RATIP match their archives. Git commits, local modifications,
source anchors and readiness are recorded; no generic code claims the exact
Yerokhin/Togawa specialized Ne resonance calculation has been reproduced.

QEDMOD potgen/potuse1/potuse2 build from pristine source in an isolated
directory. Z10 point-nucleus potgen and potuse1 run, yielding nine states and
the package inversion/calibration checks at printed precision. Output units
are dimensionless F(Z alpha). Underflow/denormal flags remain archived;
package self-calibration is not independent Ne7+ physical precision. FAC
startup succeeds using the existing private Fortran runtime. Other newly
downloaded programs are not built; prior XATOM is not rerun this turn.

Source comparison identifies concrete follow-up entry points: QEDMOD/pCI
nonlocal self-energy, GRASP rmcdhf90's selected-root-index policy versus
atomx's frozen-reference majority contract, and RATIP continuum exchange.
These are operator/target/model differences, not established causes of the
current energy discrepancy. Static exchange does not supply full channel
coupling; bound-state real-symmetric CI solvers do not certify complex poles.
P0 stationarity, q/r errors and all width/yield/experimental gates remain
unchanged. Evidence: reference programs (local evidence: `reference_programs_2026_10_04/README.md`).

## 26. Live GRASP build and frozen n2 radial-input control (2026-10-04)

GRASP44 serial programs now build in an isolated ~/code directory, with6
upstream tests passing; MPI omitted. Pristine source and previous download
report are preserved. Predeclared point-nucleus Ne7+ N3 n2sp control selects
seven equally weighted levels in four J/parity blocks,28CSF; it is not the
original n4 multi-charge frozen-reference objective.

Original G92RWF tabulated P/Q and nodes feed atomx's shared GridCIWorkspace
without interpolation, projection or reoptimization. Seven matched energy
roots differ from GRASP by at most0.000216729meV; radialGram defect9.08e-13.
Small c/quadrature differences remain explicit, and matching this finite
spectrum cannot certify every signed matrix/amplitude or physical model.
Fresh-directory rerun reproduces six physical output files byte-for-byte.

GRASP stops after4SCF cycles while reportedSCNSTY reaches2.255e-6, above
defaultACCY5.96046e-8. It also allows a relative-energy-change stop; this
is not atomx's projected-gradient/estimated-energy-error stationarity test.
No frozen-reference n4 identity boundary was crossed or repaired here.

The controlled n2 panel provides no evidence of a low-levelCI discrepancy
of order0.22eV. Original n4 basis/target/projector/optimization, correlation,
resonance/continuum and correction/experiment gaps remain open. Next compare
the original n4 identical radial snapshots/fullCSFspaces before interpreting
independent optimization changes. Evidence:
GRASP diagnosis (local evidence: `grasp_diagnosis_2026_10_04/README.md`).

## 27. Reference theory reproduction remains failed; new operator controls (2026-10-04)

FAC nine full n2/n3/n4 DC/DCB/QED attempts and repeat runs complete. Signed
native LS recoupling (sympy 6j) selects unique triplet-parent q/r above50%;
n4 DCB+QED903.454987/903.337235eV does not reproduce author theory.
GRASP point-nucleus three-level equal-weight SCF is a different objective
from atomx's original multi-charge frozen-reference contract. Setting only
rwfnestimate's fine grid was reset by defaultSCF; actual N590/H.05/ACCY1e-10
300cycle n2/n3 controls reach iteration limits, n3SCNSTYmax.08846. n4 fails
2s node counting (2 vs1), IMPROV method2. Native frozen n3 DCB+VP+screened
hydrogenicSE+recoil907.577279/907.454359eV is not a stationary prediction.

2017 printed multiplet components are in Ry: Breit absolute shifts roughly
-289/-293meV, QED-125meV, netrecoil-22meV. Only the differential effects are
small(meV). Earlier claims excluding these corrections from a0.22eV common
shift were incorrect for absolute centers. Prior atomxDC versus publishedDC
actually differs by-191.411/-217.650meV. Do not borrow the paper increments
onto atomx or compare mismatched Hamiltonians to assign the remaining error.

RCI internally interpolates G92RWF onto a different grid. New directR92RES
spinor/potential/Jacobian controls retain Gram defects9.35e-10/3.13e-9:
DC q/r differences0.00003–0.00006meV, DCB differences36.66–49.06meV persist.
Open-shell Gaunt/retardation/radial/angular matrices remain to be compared;
no codebug attribution yet. Zero/tiny-frequency1e-6 maxleveldifference
1.24209/.911930meV fails the declared1e-5meV limit gate; threshold unchanged.

QEDMOD phase import is calibrated with analytic pointDirac1s, requiring
Q_QEDMOD=-Q_GRASP. CorrectF_total4.404974 matches the package's printed
anchor; wrongphase-488.692827 is retained. Diagonal orbital injection of
local+nonlocalSE expectations still omits offdiagonalCI QED, so it cannot
certify fullmodelQED. A first R92RES parser control produced zeroLS weights
because unsorted keys changed template determinant order; identity assertion
now rejects it and the original invalid output is retained.

Dedicated author2017/2024 inputs/executable were not obtained. ExactDKB
continuumbalance/complexscaled precision and fulluncertainty budget remain
missing from these generic reference controls. No productionphysics edit,
SOTA/experimental widths/rates/yields certificate or speedup claim.
Evidence: reference theory attempts (local evidence: `reference_theory_reproduction_2026_10_04/README.md`).

## 28. Open-shell Breit implementation gap closed on native n2/n3 controls (2026-10-04)

The previously unassigned 37–49meV DCB same-input line discrepancy is now
attributed to atomx mixing its legacy Coulomb reduced-C phase with standard
CG spinor Breit elements. Independent exact Sympy512 identities give the
two-body conversion (-1)^sum|kappa|. Gaunt and retardation now share it;
Coulomb, radial input, target weights and signed LS/root selection unchanged.
Original GRASP was not patched: an isolated attempt adding its IFAZFRCS to
Breit branches failed (up to99.03meV) and remains archived.

Complete native n2 four-block Breit matrix difference65.441916meV reduces
to0.000006759meV; n3 full368CSF control reaches0.000035202meV. All same-grid
n2/n3 q/r differences are below0.00006meV after repair, DC lines unchanged.
Native compressed full-matrix/radial fixture adds an open-shell regression.
These are finite frozen-input implementation controls, not numerical theory
uncertainty or stationary physical predictions. All author reproduction,
original n4 identity/stationarity, continuum/pole, fullnon-diagonalQED and
experiment/rate/yield gates remain open. Existing sealed records preserved;
new Breit spectra and losses must be rebuilt consistently before use.
Evidence: open-shell Breit diagnosis (local evidence: `xfel_breit_open_shell_diagnosis_2026_10_04/README.md`).

## 29. Original n4 identity response and feasible descent controls (2026-10-04)

The old majority-boundary line-search failure does not imply that every
feasible descent direction is exhausted. With the exact original reference,
ensemble, nuclear positive projector and gates, five-kappa and orbital-column
controls lower the joint objective by0.112229/3.865309meV and raise the reference
q-label overlap to0.500120738/0.568652903. These are ensemble energies, not
certified spectral shifts. Public optimizer restarts again stop naturally at
the boundary; original-seed metric residuals1.549087/1.484203 and energy-error
estimates2.396498/2.502496Ha fail the unchanged1e-7/1e-10Ha stationarity gates.

An isolated real-root diagnostic computes the identity derivative from
singular-safe determinant cofactors plus the complete same-symmetry CI
reduced resolvent. CI mixing dominates3p/4p identity changes, so freezing CI
amplitudes omits the main term. Full-space analytic projected descent lowers
the objective1.990564meV atstep.01, preserving original tracking/Armijo.
Independent nonlinear and fixed-amplitude radial differences check both
terms; small-step cancellation and failed large steps remain archived.

The current nearest q-block gap0.026888419Ha is not a machine-precision
near-degeneracy. Distinguish the earlier close-root eigensolver problem from
the current frozen-reference majority boundary. The prototype remains in
the report, with no production change; general complete-degenerate-subspace
and complex-pole responses remain missing. Feasible-direction progress is
not stationary physical spectroscopy or published-theory reproduction.
Evidence: identity response diagnosis (local evidence: `xfel_identity_boundary_descent_2026_10_04/README.md`).

## 30. Shared complete real CI subspace response; original n4 remains nonstationary (2026-10-04)

Public overlap cofactors and signed bilinear Hamiltonian weights now provide
fixed-reference real principal-matrix response with only selected-to-external
energy denominators. Complete internal degeneracies cancel; repeated minimum
principal values require a derivative matrix, not one arbitrary eigenvector.
Complex pole response and multiple active identity inequalities remain open.
Optional non-recording weight gradients preserve original energy diagnostics.
74 tests pass; three full N1/N2/N3 default overlap controls are bitwise equal
before/after. Public n4 slopes reproduce the sealed prototype within3.695e-13.

Original frozen reference/charge-and-level ensemble/projector/seed metric and
actual majority>.5/1e-7 residual/1e-10Ha energy-error gates are unchanged.
Diagnostic feasible steep20 and L-BFGS60 lower jointE128.295058meV but residual
.898685/error1.390570Ha still fail. Identity-normal squared alignment.9999987265
at support.586899 identifies possible overrestriction by the .6 activation.
Only numerical activation is narrowed to.5001 for another40 steps; jointE falls
35.777685meV, support.5000760065, residual1.113298/error2.234707Ha. Tiny final
energy change2.8e-13Ha is not stationarity. All three runs exhaust declared
budgets; rejected trials and original-metric endpoint repeats are preserved.

This is public response capability and path diagnosis, not production optimizer
integration, KKT/global impossibility or spectral/theory/experimental accuracy.
Next local feasible curvature/step modeling must retain the original hard gates.
Previous native Breit equality holds; dedicated balanced-continuum/complex-pole
reproduction, full QED and experimental forward comparison remain open.
Evidence: shared subspace response (local evidence: `xfel_subspace_response_2026_10_04/README.md`).

## 31. Feasible curvature and explicit isolated-root numerical repair (2026-10-04)

Original frozen target/reference/positive projector/seed metric and actual
majority>.5, residual1e-7 and energy-error1e-10Ha gates remain fixed. The
initial polar identity-tangent path bends inward at second order (P''+172.221)
but costs energy (E''3.62576Ha). Its equal-identity curvature is .874775Ha.
A scalar Newton curve is a diagnostic path restriction, not a replacement
physical objective or KKT certificate. CI second derivatives remain finite
differences; the new public real polar velocity solves an orbital-column
Sylvester equation and does not change radial/no-pair geometry.

Six accepted steps terminate on an original-matrix eigensystem residual
failure; a bounded sampling retry adds four steps before no_valid_curvature_pair.
Two captured304-order N3 J1/2 even blocks fail at unselected near-zero root79.
Extended verification alone and arbitrary LAPACK driver switching do not fix
both. Explicit residual_iterations=1/2 now permits bounded isolated-root
spectral corrections against the SAME H; complete clusters cannot be split.
Default0 preserves bitwise old behavior, including failures. Original double
matrix residual1e-10, Gram1e-10 and order gates remain unchanged.

Diagnostic opt-in adds30 steps and ends maximum_iterations. Total40-step
jointE decrease3.076017meV is not a q/r shift. Endpoint original residual
1.075715982/error2.201434991Ha fails stationarity; identity.500076006500 and
smaller equal-identity tangent norm.00229717 do not certify the original problem.
20 numeric repair calls have max correction6.36e-13/max energy change5.23e-14Ha;
post-repair original relative residual<=2.94e-11. Independent default/explicit
endpoint repeats agree;72 tests including20 new geometry/residual cases pass.
Production energy optimizer unchanged. All author-theory/pole/continuum/QED/
width/yield/experiment/SOTA gates remain open. Next assess broader feasible
curvature and multiple constraints without replacing the original reference.
Evidence: feasible curvature diagnosis (local evidence: `xfel_feasible_curvature_2026_10_04/README.md`).

## 32. Paper-method comparison and finite-channel truncation (2026-10-04)

2017 balanced frozen-core DF/DKB real CI and 2024 complex-scaled extended CI
are distinct valid energy routes; absence of complex scaling alone is not a
precision verdict. Published 2024 Table I assigns908.194/908.069eV to2017
including erratum and908.200/908.071eV to2024 theory. Absolute2017 Breit/QED
corrections are approximately−.29/−.125eV; small splitting corrections cannot
justify the earlier assertion that these do not affect common centre offsets.
No borrowed corrections are accepted into atomx.

Representation-only alignment to the frozen1s preserves the original finite
space/spectrum. Current q160-step box eigenvector has42.0959% closed-core
spectator Q probability, versus.4555% at original reference; r~.0666%. Exact
finite Schur reconstruction confirms retained-Q coupling is already included:
adding its same self-energy again double-counts. Q here contains bound as well
as pseudocontinuum spectator states, not an outgoing autoionization projector.

Freeze all old orbitals/P/projector; add25 of94 complementary p closed-core
channels below60Ha relative core. Old H remains unchanged within2.85e-14Ha.
Small2/4/8 additions move q only2.26/2.46/2.90meV; full25 fragments the old
seed, max strength.499607. Original frozen reference remains uniquely paired
at.995829(q)/.999395(r). Thus old box mixing is basis-sensitive, while the
physical reference has not disappeared. Fixed-core signed CI includes exchange
only in this channel construction; general scattering capability is unchanged.

Same-reference cutoff40→60Ha still moves q−108.358meV; r−.587meV locally,
but20Ha control moves r+499.059meV. No full correlation tail, DF/DKB box/grid
balance, complex pole or matched enlarged ground has converged. Original
stationarity gates remain failed. Diagnostic box centres908.545238/908.359964
cannot be promoted to precision values. No exclusive attribution of original
n3~.22eV error, author-code reproduction or experimental/SOTA certification.
Next separately declare a frozen-core DF/DKB spectroscopy benchmark, retaining
the original shared-orbital task and gates; do not substitute a Q-only enrichment
or same-Q correction for the original physical problem.
Evidence: paper and channel diagnosis (local evidence: `xfel_paper_method_diagnosis_2026_10_04/README.md`).

## 33. Complete fixed-projector spectator tail and exact elimination (2026-10-04)

Freeze old P, all old orbitals and nuclear positive projector; complete the
spectator p complement with94(q)/93(r) channels, matrices534/391. Seven declared
cutoffs60/100/200/500/1000/5000Ha/full retain original frozen-reference matching.
Full versus60Ha shifts−52.917825/−1.683977meV; full versus5000Ha increments
−.0000413654/−.000000303170meV. This closes a conditional finite-projector
spectator tail, not radial-box/continuum-density or full P correlation accuracy.
Original reference majority.991224/.999396 persists; old q160-step seed support
returns to.523422, closed-core probability falls to.141259%. The60Ha loss of
old-seed majority is itself cutoff dependent. Old H unchanged, spectral moments
close, original target eigen-residual ratio<1e-10 without relaxing the gate.

Exact same-H Schur elimination retains every tail channel with energy-dependent
self-energy and analytic derivative. Two effective solves per mode close
|lambda(E)-E|<1e-11Ha; retained60/5000Ha both reconstruct full vectors/energies.
Fixed-vector derivative independent FD relative errors<=3.17e-10; full-H residual
gate passes. No same-Q self-energy added twice, no speedup benchmark claimed and
no production API changed. Complex-pole left/right response is separate.

Diagnostic full box splitting134.039863meV versus2017 displayed DC120.818555
differs13.221308meV. Old frozen-ground centres908.492320/908.358280eV are not
precision predictions. q full-Q below/above pole distances7.934591/.483607Ha
remain unbalanced by~16x. Increasing a cutoff within a fixed finite space cannot
supply the missing continuum-density/grid/DF-DKB/full-correlation controls.
Original orbital stationarity, author reproduction, QED/recoil, physical poles,
widths/experimental forward/SOTA remain open. Preserve original joint objective;
separately declare a frozen-core DF/DKB density benchmark rather than substituting
the conditional spectator result for the shared-orbital problem.
Evidence: complete channel tail (local evidence: `xfel_channel_tail_2026_10_04/README.md`).

## 34. Frozen-core continuum-density controls (2026-10-04)

Eight declared grids and sixteen κ spectra preserve the exact frozen1s function,
Fermi nuclear potential and R40. Original ns100/ks9 uses equal P/Q order, not
DKB. Exact knot insertion refines26 outer intervals; each new nuclear positive
projector is explicitly a new finite model, not the original joint task.
Closed-core angular reduction gives hD+2J−X(K1)/3, independently verified with
Sympy3j and the entire original signedQ operators (97/96 dimensional, relative
Frobenius4.54e−15/3.62e−15). GeneralN production outgoingDW remains unchanged.

Bracketing Q spacing falls q/r8.418/9.532 to .496/.505Ha at ns490, while
asymmetry is .416/.883. Changing knot location at identical ns126 also changes
asymmetry. Gauss9→13 on ns282 changes near levels at most7.32e−5meV. Therefore
fixed-projector spectator-tail stability does not certify continuum density or
balance. No full newN3CI/resonance energy was calculated; these are distances
around fixed oldbox energies, not q/r shifts or fitted experimental values.

Full-spectrum bounded numerical repair fails energy-change/Gram protections;
all failures retained. Only the two bracketing roots are locally certified via
the existing sameH bordered Newton kernel, same-root overlap>1−1e−9, numerical
energy change<1e−8Ha, pairGram<1e−10 and relative matrix residual<1e−10
(actual~1e−16). Complete-spectrum numerical precision remains uncertified.
Original orbital stationarity, authorDF/DKB/complex-scaling reproduction, full
correlation, QED/recoil, widths and experimental/SOTA gates remain open.
Eleven old seals/21basis inputs/cleanGRASP44binaries/332productionPython sources
unchanged. Evidence: density diagnosis (local evidence: `xfel_continuum_density_2026_10_04/README.md`).

## 35. Independent frozen DF/DKB and correlation diagnosis (2026-10-04)

Ne8+ closed1s² DC/Fermi core now independently converges on three public RKB
grids (10iterations, constrained equation residual2.9–5.1e−14). The two finer
core energies agree within1e−10Ha. A complete real point-DKB representation
receives actual finite-Fermi correction and nonlocal frozen exchange before
Fock positive-band selection. This is a declared independent spectroscopy
model; the original joint N3/N2/N1 orbital target remains nonstationary.

CI uses nuclear hD, not Fock eigenvalues plus duplicated mean field. Signed
closed-core QQ equals Ecore+εκ within1e−12Ha; Sympy3j verifies s/p/d/f exchange,
and independent split-triangle Slater differs from YK by at most1.29e−12Ha.
These controls establish operator conventions, not complete physical precision.

Fixed DF/DKB bound P n2→n3→n4→n5→n6spd moves q/r approximately−541/−539,
−149/−149,−64.6/−64.2,−34.3/−34.0meV. The n6 diagnostic pureDC transitions
909.057803/908.917263eV and split140.539795meV remain unconverged. A single
n4 f shell contributes−.232/−.216meV; that does not bound all higher-L tails.
Closed-core p spectator cutoff100→200Ha contributes only~−.023/−.014meV,
excluding general high-energy virtual-orbital correlation. The actual n6
continuum bracketing distances~1.157/4.900Ha give asymmetry~.618; balancing
and full positive-orbital energy-cutoff CI are still missing.

All roots retain the original n2 embedded majority (minimum~.9953); no
experimental root selection or shifts. Thirty DF/DKB spectra,132 CI cutoffs,
ten actual brackets,12 old seals/21basis inputs/cleanGRASP44binaries/332Python
sources are verified. The nonzero-theta finite-Fermi pole provider, general-N
outgoing exchange/coupled scattering, Breit/QED/recoil, experimental forward
model and SOTA acceptance remain open. Evidence:
DF/DKB diagnosis (local evidence: `xfel_frozen_df_dkb_2026_10_04/README.md`).

## 36. General virtual correlation and actual-energy balancing (2026-10-04)

A declared frozen DF/DKB space now includes every n2 pair plus at most one
external s/p/d electron, with common cutoffs0/10/20/30/60/100/200Ha. Magnetic
combinatorial completeness and unique CSFs are verified for that restriction.
Two-/three-external configurations remain missing; this is not the author's
complete energy-cutoff CI and is not nested with the prior full bound n6 space.
Ground and q/r use a common Hamiltonian at each cutoff; the nuclear one-body
operator and complete previous closed-spectator submatrices replay unchanged.

General correlation100→200Ha moves q/r−42.819/−41.751meV at gamma1, versus the
previous closed-p spectator tail~−.023/−.014meV. The apparent30Ha agreement
with final theory is accidental: this is DC only, and30→200Ha moves another
−333.406/−325.096meV. Current below-DC centres cannot allocate missing ground,
excited correlation or projector effects uniquely or certify precision.

Real physical cross-grid identity uses merged-knot Gauss27 integrals and signed
three-electron overlap determinants, always retaining the original gamma1 n2
reference and unique probability>.5. An omitted whitening transform in an
auxiliary primitive-DKB sampler was rejected by public sampling replay; the
full congruence transform was restored. Rejected source/logs and initial files
remain, while old public operators/reports are unaffected. A200/30Ha control
matrix mismatch was likewise rejected and fixed by matching the actual cutoff.

The initial nine-gamma scan changes bracket indices and is explicitly rejected.
A separately declared fine scan balances conditional n2 closed-spectator CI.
Transferred to general CI, the midpoint mismatch is~.08Ha. Two bounded secant
probes per target then balance actual200Ha general-CI energies within1e−6Ha:
q gamma.8936892804965899, r gamma.8937014139547796, actual mismatches
2.038/1.931e−7Ha. Different target gammas do not define one finite-H interval.
The actual q-balanced common grid gives907.851396/907.730748eV and same-grid
interval120.648442meV, but100→200Ha still moves−40.075/−39.055meV. This is
conditional continuum balance, not a spectroscopic uncertainty certificate.

Twenty-five grids,75s/p DF spectra,50cross-grid control CI and126general-CI
cutoffs replay;13old seals/21identity inputs/cleanGRASP44binaries/332production
sources remain unchanged. Missing multi-external correlation, joint density/
Lmax/cutoff controls, original n4 joint-orbital stationarity, Breit/QED/recoil,
finite-nucleus complex poles, widths and experimental/SOTA acceptance remain
open. Evidence: general virtual diagnosis (local evidence: `xfel_general_virtual_2026_10_04/README.md`).

## 37. Matched external-rank discrimination (2026-10-05)

Reusing the sealed bound n3/n4 Hamiltonians permits rank0/1/2/3 comparisons
without radial rebuilding. Rank2 shifts q/r transitions approximately−1.44meV
at n3 and−2.52meV at n4; rank3 adds only−.004 to−.010meV in the tested spaces.
This does not bound general high-energy double/triple correlation.

Three additional matched spaces keep n3 bound orbitals and select one positive
orbital per s/p/d kappa nearest5/30/100Ha. They are sparse band probes, not
complete cutoffs. Whole n3 and rank1 subblocks agree with independent sealed
matrices within3.55e−15Ha. Rank2 shifts change sign at the two higher bands:
common transition increments+3.64/+5.23meV because ground-energy reductions
9.443/9.835meV exceed upper reductions~5.8/~4.6meV. Ground and upper correlation
must be treated together; bands and occupation-switch increments are not additive.

At~100Ha, adding rank2 configurations whose remaining n2 electron is valence
alone shifts q+6.475meV; remaining1s alone shifts−1.217meV; both give+5.219meV.
The next priority is full high-energy rank2 differential correlation, including
both subspaces, before more precise gamma balancing of rank1-only models.

The first new subblock mapper used nonunique CSF debug repr, omitting coupling
trees;64.41Ha mismatch rejected it. Full signature.key matching restores exact
subblocks; old matrices and production kernels were unaffected. Sixty main rank
and30independent occupation-switch replays pass, all original-reference majority
probabilities>.9959 and original relative residual gate<1e−10. Consumed immutable
inputs and332production sources remain unchanged. Recorded internal times.424s
and19.33s are not a matched end-to-end10x benchmark. Full-cutoff/Lmax/density,
original n4 orbital stationarity, author reproduction, corrections, poles and
experimental precision remain open. Evidence:
external rank diagnosis (local evidence: `xfel_external_rank_2026_10_05/README.md`).

## 38. Complete declared double-external windows (2026-10-05)

The full200Ha single-external P is unchanged. Added rank2 pools contain all
n3 bound spd plus every positive s/p/d orbital in (100,200] or (60,200]Ha,
including all allowed cross-energy/cross-kappa pairs and either remaining n2
electron sector. Largest target dimension3536. These are complete declared
pools, not complete200Ha rank2 CI; outside-pool pairs and rank3 are missing.

Ground reductions54.265/150.065meV exceed upper reductions~9.9/~23meV;
transition common shifts are+44.374/+127.121meV. Expanding100→60 changes
q/r+82.946/+82.548meV, rejecting1meV closure. A bound-only rank2 subset
retaining the identical full P shifts q/r−1.998/−1.918meV. Its difference from
the windows isolates collective additions without borrowing small-P results.
Small rank2 probability and sub-meV interval changes do not bound energy tails.

The first control scope incorrectly assumed the prior sparse~100Ha band was
fully inside strict(100,200]: two orbital energies are below100. Missing full
structural keys rejected it. Actual common rank<=2 subblocks are compared;
all rank1 must exist and the60window contains the whole sparse rank<=2 control.
Failed source/log retained, physical pool and gates unchanged.

Six window states, three bound controls, three full nested submatrices and
three direct Slater controls replay. Original-reference majority>=.9875,
relative residual<=5.70e−15, consumed SHA/332 production sources unchanged.
Internal292s assembly/solve and.534s bound-control times are not matched10x
end-to-end benchmarks. Current60window pureDC907.978821/907.858607eV
remains~.651eV below the displayed2017DC values; finalQED theory is a different
Hamiltonian. No attribution of the original joint-orbital n3 discrepancy follows.
Next: low-positive rank2 and cross-window pairs, cutoff/density/Lmax controls;
original n4 stationarity, author reproduction, corrections, poles and experimental
precision remain open. Evidence:
double-window diagnosis (local evidence: `xfel_double_window_2026_10_05/README.md`).

## 39. Lower-energy rank2 and cross-window pairs (2026-10-05)

The identical full200Ha single-external P is retained while the rank2 orbital
pool lower edge moves to40/30Ha. All cross-pairs between new and existing
orbitals enter. The30window is assembled;40 is its complete exact subset.
Largest5870CSFs/276MB; predeclared6000target/1.1million pool/20000raw-block
resource limits change the scope, not the original majority/residual gates.

At30Ha ground lowers314.991meV, q/r uppers51.089/52.226meV; transition
shifts+263.902/+262.765meV. The40→30 increments remain+62.420/+62.082meV,
rejecting1meV closure. Absolute pureDC centres908.115350/907.994503eV remain
~515meV below displayed2017DC; interval120.846561meV differs only~.028meV.
Agreement of an interval without complete correlation/density/Lmax/balance
control is not precision reproduction or an absolute-energy certificate.

Same-P independent rank2 switches at30Ha give common shifts+17.954meV
(remaining1s) and+245.366meV (remainingvalence). The latter lowers ground
291.003meV, dominating this tested differential correction. Both sectors stay;
switches are not forced into additive error budgets. The remaining1s sign
differs from the earlier sparse~100Ha probe because the spaces differ.

Six new states/12switches/6old60submatrices/3nested submatrices/3direct Slater
controls replay, original full-window minimum majority~.987789 and residual<=3.75e−15;
332production sources and consumed immutable inputs unchanged. Internal
560.508s contains531.812s assembly (94.9%);40subset5.827s/switches28.163s.
Assembly/angle reuse is the measured priority; no matched10x end-to-end claim.
Missing low-positive and n>3bound rank2, rank3, cutoff/density/Lmax, original
n4stationarity, author reproduction, corrections/poles/experiment remain open.
Evidence: low-window diagnosis (local evidence: `xfel_low_double_window_2026_10_05/README.md`).

Erratum: the previous sealed prose gave82.830454meV for r100→60; actual
saved energies give82.548454meV. Section38 and trace corrected, old H/results
were correct and the old seal remains immutable; new report records the erratum.

## 40. Exact sparse CSF assembly and separate performance acceptance (2026-10-05)

The common exact CI kernel now constructs configuration-block CSR C, retains
all coefficients and performs the same global orthogonality, J-span/leakage
and Hamiltonian projection checks. Default dense API and complex transpose
conventions remain. Integer connection deduplication and direct triangular
pair enumeration preserve the complete Slater–Condon graph; decoded coordinate
integers are shared to avoid the intermediate peak-memory regression.

Fresh-process q5870 assembly point measurement is181.560→110.120s (~1.65x),
projection/entry remainder58.154→.664s; qRSS9.901→9.543GiB. The predeclared
load1/affinity guard FAILS:~.321>.25. Keep this failed performance acceptance;
only an observed timing point is reported, no controlled speedup or full-task
10x certificate. Scientific equivalence is separately checked on all three
real sealed g/q/r matrices and original-reference identities. Angle coefficient
generation remains dominant. Low-positive rank2, n>3bound, rank3, density/Lmax,
original n4stationarity and2017/2024 author reproduction remain open.

One production source changes and331remain unchanged. Historical332-source
SHA guards must not be bypassed; subsequent experiments must explicitly consume
this validated old-to-new kernel revision and its seal. Failed optional driver,
intermediate source/memory results and initial diagnostic g are retained.
Evidence: sparse assembly report (local evidence: `xfel_ci_sparse_speed_2026_10_05/README.md`).

## 41. Double-excitation allocation and duplicate collection (2026-10-05)

Actual q14624determinants have18826400candidate connections; a deterministic
100141sample has97.7%double excitations. Direct low-bit extraction and exact
occupation parity replace temporary generators/sorts/operator lists. The
static/no-common-core path collects canonical keys once. Defaults and general
core contractions remain, with all signed exchange terms and arbitrary-width
integers. Complete matrix equivalence, not metadata bit identity, is required.

Fresh single-thread q5870 assembly109.939→84.003s
(observed1.31x). Angular structure
93.952→67.948s;
RSS9.521→9.416GiB.
Load guard FAILS:0.319>.25; no controlled factor/end-to-end10x
certificate. Three real fullH maxdifference1.705e-13Ha,
maxresidual3.257e-15, minimummajority0.987893;107passed/1existingxfail,
Ruffpassed. Eight independent occupation-list sign gates cover2–5electrons
and high bits through511. One source changes,331/inputSHAunchanged, explicitly
chained to section40. Physical precision/correlation/author reproduction remains
open. Evidence: pair speed report (local evidence: `xfel_ci_pair_speed_2026_10_05/README.md`).

## 42. Stop unproductive key sorting; reuse existing angular tables (2026-10-05)

The fixed comparison candidate matches75347actual unique keys and signs, with
26checks/Ruff passing and a2.33x helper observation. Fresh fullq83.382→83.738s
fails the declared3%benefit threshold; source restored byte-exact, candidate
archived. No production change remains.

Existing angular LRU reuse in one process is verified: coldq83.208s, warm15.073s
(observed5.52x assembly). Each fresh radial provider recomputes1291062callbacks,
1286067integrals/28060potentials. Warm angular regeneration is forbidden.
Full same-H error5.684e-14Ha/residual3.185e-15; original majority.987893 and
selected-vector squared overlap>1−1e−9 remain. Formal one-body+.01Ha control
requiresH+.03I forN=3, passes2.842e-13Ha matrix error; not a physical correction.
Only same-wave/fresh-provider and identity control are tested, not new-orbital
SCF or new spaces/operators. First assembly remains83s. Load.314>.25 fails;
no controlled, first-run or end-to-end10x certificate.

Provenance correction: stage41 after-map contained a single-source delta,
which this stage initially inherited as a full inventory. Preserve executed
protocol; reconstruct complete332sources from stage40 before-map plus stage41
kernel override, verify current path/hash equality and actual radial before
snapshot. Stage41's331unchanged-sources wording lacked a complete live guard;
closeout equality now verified, without retroactively claiming that guard ran.
Older seals unchanged. Corrected replay uses the full map. Physical correlation,
n4stationarity,2017/2024 reproduction, corrections/poles/experiment stay open.
Evidence: key/cache report (local evidence: `xfel_radial_key_speed_2026_10_05/README.md`).

## 43. Closed-channel correlation distinguished from the open Auger family (2026-10-05)

Same frozenDF/DKB hD+Coulomb rank1-to200Ha and complete rank2 cross pairs;
lower edge30→20Ha, nested25/30 from the same full H. g/q/r5363/7722/5364,
resource caps8000/24000det predeclared; physical gates unchanged. Total shifts
-91.437951/-21.160647/-21.409215meV; transitions
+70.277304/+70.028736meV. Last25→20 still
+42.023254/+41.837144meV:1meV closure fails.

NewQ has at most one1s and excludes1s²epsilon-p open Auger family, already inP.
Nearest newQ denominators69.574/36.298/36.302Ha; second-order
versus full-CI differences <= 0.009244meV, exact finiteSchur rebuilt. Supports
ordinary closed-channel correlation for this increment; does not bound oldP
continuum/pole error. RetainedQ self-energy cannot be added a second time.
2017 balancing/2024 complex scaling each also require correlation convergence;
Breit/QED absolute shifts differ from their small fine-structure differences.
FrozenDF He core is not the correlatedHe-like energy zero in2017TableIII.

15saved-state residual/identity checks,3old full submatrices,3Schur and3direct
radial controls pass; maxresidual4.927e-15, minimum
original majority0.987880456.332source/inputSHA
unchanged at start/end/replay; no production change or speed certificate.
Low-positive/n>3bound rank2, rank3/higherL/density/cutoff, n4stationarity,
corrections/poles/author reproduction remain open.
Evidence: threshold physics report (local evidence: `xfel_rank2_threshold_2026_10_05/README.md`).

## 44. Same-space Breit/QED/recoil and five correlation windows (2026-10-05)

Frozen DF/DKB/Fermi nucleus and unchanged CSF conventions. Complete static
Gaunt+retardation, full local/nonlocal QEDMOD SE+Uehling+WK, one- and two-body
Breit recoil on identical wavefunctions; all same-kappa offdiagonal QED kept.
20Ne nuclear mass36433.989009891855me. 24 exact numeric Breit row-shard processes avoid the original full angular-table materialization;3 complete s/p/d blocks agree to1.735e-18Ha. Three sector eigensolver processes assemble15Ha
once and extract complete30/25/20/17.5 nested subsets. g/q/r6295/9068/6309.
Original identity/residual/1meV physical gates unchanged.

15Ha corrected q/r=907.778626721/907.652725366eV,
split=125.901355meV; experimental residuals
-423.273279/-426.874634meV.
Last17.5→15 changes+13.317728/
+13.206396meV; last-step1meV gate
FAIL. Full correlation, paper reproduction
and resonance pole remain uncertified; numerical residual is not physics error.

Weighted full-rank DKB reconstruction preserves archived waves to4.24e-13.
9769 independent Fortran QED matrix elements agree to3.617e-06meV;45 original
log-origin NaNs retained and explicit Simpson plus short-range omission checked.
QEDMOD q=-F versus atomx Q=F requires explicit unitary spinor conversion;
initial wrong-convention agreement rejected by n2 physical control, original
hydrogenic SE table checked to2.300e-13Ha.
Signed VINTI legs cannot use unsigned Coulomb cache normalization: initial scan
driver wiring rejected, correct signed SMS independently checked on3 blocks
to5.42e-20Ha, valid DC/scalar checkpoints reused, Breit assembled with independently validated exact row shards. Existing production SMS did
not have this driver wiring error. Direct scalar QED/NMS lifts separately checked.
75 states replayed; maxresidual5.225e-15,
minimum original probability0.986919195, minimum
DC/corrected squared overlap0.999955801.332sources unchanged;
88 existing reference tests pass. Missing low-positive/n>3bound rank2, rank3,
higher L/radial density/uppercutoff, frequency/radiative/finite-size recoil,
original n4 stationarity and physical poles remain open.
2017+erratum908.194(4)/908.069(4) and2024new908.200(4)/908.071(4) are distinct.
Evidence: corrected window report (local evidence: `xfel_corrected_windows_2026_10_05/README.md`).

## 45. 多电荷复尺度共享径向、二维身份与 rank 闭合（2026-10-05）

- 隔离框架实算Ne N2/N3/N4（8+/7+/6+）六个J/宇称块；每θ只生成
  一次共同核Dirac正能DKB径向快照，复用原完整Coulomb/Gaunt/retardation
  与jj CSF。共同轨道不意味着多电荷联合优化。点核复DCB与§44有限Fermi
  冻结DF实数修正是不同明确模型，不能拼实态QED补丁或重复Auger自能。
- 完整n≤2 +≤1 external所有模式/树，R4/count16→20、θ.2→.4五角；
  R6/count24另作联合几何/截断控制。ns125/ks9/rmin1e−6/γ.4/Gauss25。
  不按伪态n声称连续覆盖；保存最高能量、正能支大小和共同残余阈值。
- 同θ核锚与相邻θ归一化coordinate-L2 determinant minors共同匹配；
  复Hamiltonian仍c-product，不把L2或系数平方当物理占率。原固定θ0
  远角身份失败保留；dilation变化本身影响L2，不能独断态丢失。
  跨γ union-cell解析DKB重采样保留原波函数相位，最大重建差
  5.607e-15；同/邻θ与跨γ身份均通过。
- γ.35的Gauss25被原DKB对称门拒绝1.15723e−10>1e−10。失败保留；
  Gauss33重算γ.35/.4/.45并核对中心25/33，不放宽门。已算局部θ/γ
  导数，尚未找联合驻点；不能挑能量最贴论文/实验的角度。
- count20 N3核seed2两块θ复跨度30.366/29.656meV，ImE仍为正；
  N2/N3/N4束缚虚部最高约.00554/.00609/.00641Ha，均超1e−7门。
  全rank1基态/孔态θ跨度未过1meV。部分其他孔态ImE<0亦不能因此
  认证；形式−2ImE不自动转Auger率。LS/E1 q/r归属尚未认证。
- 通用external-rank枚举完整交叉对，原资源cap12000先检查；count4
  六块rank2实算，rank1完整子矩阵同模型相符，N2 rank2完整Fock与
  公共全谱相符。N4 count16 rank2需约3.3–3.8万det，原资源门拒绝；
  下一步分块表示/明确资源后闭合，不删cross-pairs凑cap。
- 84个原始H、31834对根重放，最大pair残差
  1.820e-14，Hᵀ对称差0；12个DC/DCB公共N2/3/4小块、
  独立氢样/闭壳det minor/复相位身份锚通过。332生产源路径/SHA起止
  不变，本轮生产改动0。驱动已可运行，尚未新增公共multi-charge API。
  证据docs/xfel_multicharge_complex_2026_10_05，极点/完整相关/作者复现为false。

## 46. 相关类氦零点与基态低能双激发干预（2026-10-05）

- 同冻结DF/DKB、有限Fermi核、核hD+Coulomb，221轨道全配对N2
  J0偶宇称4995CSF/17513det；24进程精确行分块，原606阶完整
  对照差1.137e-13Ha，嵌入原15Ha子块同值。空间内N2完整不等于
  径向/截止/高L收敛，不将He零点加到旧q/r作为修正。
- He15→全部轨道下降289.268meV。全部轨道He能量
  -94.017691501321Ha；以此有限零点对2017印刷Coul列，原N3
  基态相对He偏高348.916meV，上q/r相对He偏低
  43.445/44.008meV；相减保持跃迁差-392.361/-392.924meV。
  同15窗口He零点则为+59.649/-332.712/-333.275meV，说明逐态
  归因依赖零点物理误差；不把349当绝对基态误差证书。
- N2完整s→sp→spd能移-564.688/-46.078meV，末分波1meV门
  未过。径向密度/200Ha截止/更高L未验；He零点未认证。
- N3g完整6295阶15Ha空间保留，只扩剩余2s的全部rank2
  双外部池至5Ha，含新旧/跨κ配对；7452CSF/30559det。
  三个完整组态小块与原全局角表最大差5.329e-15Ha；只重算至少
  一端为新det的连接，原6295阶完整子块差0，不删任何物理项。
- g下降148.107942meV，原15Ha同态平方重叠.999959357，
  原n2参考概率.999758533，残差2.300e-15。新增Q1157维最邻近
  谱分母90.615459Ha；二阶-148.076271meV与完整CI差
  -.031671meV，精确Schur Q向量差1.540e-17。这是远离基态的
  普通闭通道相关，不能称Auger宽度。
- 初次保存属性名错误在结果落盘前主动停止；小块固定CSF数截断
  被完整J空间门拒绝，改按完整组态保留全部树后重验。两个失败
  驱动和日志保存，原门未改。332生产源起止SHA不变，14态重放通过。
- 支持优先闭合基态低能双外部相关，未证明全部偏差唯一来源。
  本次不是完整5Ha三电子空间；q/r上态与Breit/QED/反冲未扩窗重算，
  不把148meV直接贴到旧全修正谱位。完整相关/作者复现/极点仍false。
  证据docs/xfel_dc_zero_audit_2026_10_05。

## 47. 三态同步低能双激发扩窗与完整修正（2026-10-05）

- 同有限Fermi核/冻结DF-DKB221轨道，rank1到200Ha不变，rank2
  同步15→10→5Ha；保留声明核心子集全部剩余核心/新旧/跨κ配对、
  完整CSF树。5Ha g/q/r为11057/16038/11058CSF、
  46759/41489/48356det；n≤3束缚池，非完整N3相关空间。
- 每窗DC→静态Gaunt+retardation→全QEDMOD→NMS→signed-SMS
  顺序重求态，无论文常数平移。旧15Ha完整算符子块最大差
  0.000e+00Ha；上一轮7452阶基态单通道子块嵌入差
  8.527e-14Ha，完整5Ha DC基态相对
  该单通道另移-15.925625meV。
- 5Ha纯DC q/r=908.346580084/908.224934609eV；
  完整修正=907.890128865/907.763927251eV，分裂
  126.201614meV；相对2024实验q/r为
  -311.771135/-315.672749meV，分裂差
  +4.001614meV。不混同旧20Ha纯DC近实验与
  相同物理修正口径，不把旧基态148meV直接贴入谱位。
- 全修正15→10谱线移动+49.846751/+49.728537meV；
  10→5移动+61.655393/+61.473348meV。末步绝对g/q/r能移
  -94.734731/-33.079338/-33.261383meV，
  谱移严格为上态减基态。末窗各1meV门={'q': False, 'r': False, 'split': True}。
- 5Ha q/r修正meV：Breit-308.138706/-312.173570，
  QED-126.086938/-126.436223，
  NMS-24.901532/-24.894749，
  SMS+2.675957/+2.497185。
- 六个完整小块/粒子数/旧矩阵扰动态求解控制通过；45保存态
  完整矩阵重放，最大绝对残差6.591e-11Ha，
  最小原参考概率0.987080007，最小前窗同态平方
  重叠0.999972053。原多数选态、不按实验
  挑根；未枚举新全谱root顺序。数值残差不当模型不确定度。
- 24进程535精确行分块，资源cap明示提高17000CSF/50000det；
  物理门未放宽。两次预检控制接口/宇称集错误保留失败证据，
  均在科学主算前拒绝。332生产源路径/SHA起止不变，本轮生产改动0。
- n>3束缚rank2、rank3、高L、径向/上截止/连续态平衡仍缺；
  2017平衡连续伪态和2024复尺度的物理验收不可用扩窗替代。
  完整相关/作者复现/极点仍false。
  证据docs/xfel_low_closed_corrected_2026_10_05。

## 48. 复空间束缚锚的自适应保留池控制（2026-10-05）

- 用户要求使用复空间思考自适应收敛。同Z10点核、共同正能Dirac
  投影、完整静态DCB，R4/ns125/ks9/Gauss33/θ.3/γ.4，κs/p三支。
  固定θ和γ，独立比较每κ保留数与external rank，完整声明组态树/
  新旧/跨κ对；不借用实数有限核QED/反冲位移，不生成q/r新预测。
- count8 rank1→完整rank2，N2 ImE .012469→.014273Ha，
  N3 .026221→.028379Ha；rank2不自动消除小轨道池的虚部。
  原rank1完整子矩阵一致。count20两块完整H与旧Gauss33封存中心
  误差0。rank1 count20→28，N2 ImE .00443388→.00083420Ha，
  N3 .00497391→.00090807Ha，支持优先增加保留轨道覆盖。
- 实际自适应分支仅在两种电荷的泄漏均下降后推进下一池，
  自动28→44→64；若恶化/身份失效/资源门失败则停当前方向，
  不靠改θ掩盖虚部。44时N2/N3约5.0864e-7/5.1880e-7Ha。
- 最后count64 N2/N3 ImE=7.115202333e-10/
  -1.879628907e-08Ha；原1e-7束缚虚部门单点结果
  =True。这不是角度/径向/相关闭合或物理极点证书。
- 12复矩阵/4537本征对重放，最大pair残差
  1.829e-13，c归一化/同κ c-Gram/完整CSF检查通过。
  核系数多数仅作完整谱选根特征，再对所选根检查coordinate-L2，
  非占率、非所有根L2唯一性认证；扩池另检查前池所选态重叠。
- 下一控制先核对束缚锚θ/γ邻域与完整rank2闭合；再按2019式11
  对每个上态最小化复dE/dθ、dE/dγ合成范数。五点局部stencil、
  步长减半/信赖域，按完整组态Schur指标择下一相关层（指标非误差界），
  重找驻点并验连续两次增量、ReE及宽度/阈值隔离、身份、径向/L/
  箱/截止。当前仅实现固定参数扩池分支，未执行完整二维优化。
- 初次旧manifest键名读取失败在科学结果前拒绝，失败代码/日志/
  来源SHA保存，按真实旧schema修复。332生产源路径/SHA起止不变，
  生产修改0；完整相关/作者复现/极点仍false。
  证据docs/xfel_complex_adaptive_2026_10_05。

## 49. 复空间束缚邻域与混合相关池验收（2026-10-05）

- 同§48点核正能Dirac投影/完整静态DCB，每κ64轨道rank1保持；
  θ/γ四个轴邻点h=.025，数值/身份通过后再h=.0125。未执行驻点优化，
  未用实验或旧实数QED/反冲修正。每参数径向共享N2/N3，仅装配所需块。
- N2_g邻域max|ImE|=2.428931445e-09Ha，
  两两最大复跨度0.000156meV；虚部/1meV门
  =True/True。保留原1e-7Ha门，未扣共同虚部。
- N3_g邻域max|ImE|=2.442578051e-08Ha，
  两两最大复跨度0.001040meV；虚部/1meV门
  =True/True。保留原1e-7Ha门，未扣共同虚部。
- 64池rank1与4/8小池全部rank2取并集，完整新旧/跨κ/核心模式/CSF树；
  不等于64池完整rank2。复核旧完整rank1子矩阵，8池另核旧完整rank2。
- N2_g小池4：ImE=+1.295091431e-04Ha，
  上一空间复移=-7.556445+3.524104i meV；虚部/增量门
  =False/False。束缚Im不可当宽度。
- N2_g小池8：ImE=+1.805708829e-03Ha，
  上一空间复移=-40.354886+45.611717i meV；虚部/增量门
  =False/False。束缚Im不可当宽度。
- N3_g小池4：ImE=+1.521258126e-04Ha，
  上一空间复移=-12.004939+4.140066i meV；虚部/增量门
  =False/False。束缚Im不可当宽度。
- N3_g小池8：ImE=+2.170511527e-03Ha，
  上一空间复移=-53.197454+54.923073i meV；虚部/增量门
  =False/False。束缚Im不可当宽度。
- 20新复矩阵/15338本征对重放，最大pair残差
  2.235e-13；c归一化/对称/径向c-Gram/完整声明空间/核心
  多数选根和中心所选态L2复查。跨γ解析DKB重采样不重求/重相位。
  L2与核心系数是身份诊断而非物理占率，不宣称所有根L2唯一认证。
- 完整新增块Schur反馈重放精确Q向量与有效P方程，c-product不取共轭；
  二阶与完整CI实测比较、谱分母和最小奇异值记录，仅作优先级而非误差界。
- 8池N3初次在完整态自重叠2172024对超过200万资源门时拒绝；保留失败，
  明示配对预算300万，原500000值分块与全部minors保持。重算仅该N3块，
  已完成N2 SHA复用；原矩阵/谱在身份检查前另存检查点，物理门未变。
- 按失败方向先验截止/径向/完整rank2平衡，再找上态θ/γ复驻点；
  扩基组后重新找驻点，连续两次增量与stencil减半稳定，箱/高L独立验。
  四进程BLAS1，未认证加速倍数。332生产源路径/SHA起止不变，生产修改0；
  完整相关/作者复现/极点仍false。证据docs/xfel_complex_neighborhood_2026_10_05。
