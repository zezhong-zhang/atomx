AtomX vs FAC — remaining capability gaps
After the recent 5 fixes (REPORT_FAC_PARITY.md), most module-level holes are filled. Here's what's still missing from AtomX vs FAC's complete surface:

🔴 Major gaps (affect physics correctness or coverage)
FAC capability	AtomX status	Impact
R-matrix close-coupling (RMatrixBasis/Boundary/CE/Convert/Expansion/FMode/NBatch/NMultipoles/Surface/Targets)	scattering/r_matrix/__init__.py is a 55-line stub — compute() body is empty	Required for low-energy near-resonance excitation/scattering, dielectronic-recombination interference
Two-photon decay (TwoPhoton)	Not implemented	Dominates 2s → 1s decay in H/He-like ions; key for X-ray plasmas
External-field structure (StructureEB, CETableEB, TRTableEB)	Zeeman g-factor exists but no joint Stark+Zeeman Hamiltonian or M-resolved transitions	EBIT spectra, magnetic-fusion plasmas, Zeeman-resolved astrophysics
MBPT for transitions / structure (StructureMBPT, TransitionMBPT)	methods/mbpt.py handles MP2 energies only, not transition matrix elements	High-precision atomic clocks, parity-violation work
Magnetic-substate-resolved cross sections (CETableMSub, CITableMSub, AITableMSub)	All AtomX scattering returns σ_total (summed over m)	Polarization spectroscopy, anisotropic emission
AverageAtom (AverageAtom)	plasma/screening.py has Debye/ion-sphere static screening, but no full self-consistent average-atom solver	Hot dense plasma equation-of-state, NLTE opacity
🟡 Coverage gaps (reduced flexibility, no missing physics)
FAC capability	AtomX status
Coulomb-Born CI (SetCILCB, 'cb' mode)	Only DW + BEB; no analytic Coulomb-Born for cross-checks
Born-approx CE high-E limit (SetCEBorn)	Only DW; no fast Born scoping
Inner-shell / K-shell Auger (SetInnerAuger)	New auger_rate only handles two-source-electron topology; not inner-shell vacancy cascades
X-ray cascade (SetCascade)	Not implemented
Asymmetry parameter β (radiation-pattern coefficient)	scattering/polarisation has alignment + Stokes but no β
Hydrogenic analytic cross sections (PICrossH, RRCrossH, TRRateH)	methods/hydrogenic.py has wavefunctions but not closed-form σ_PI / σ_RR / A-rates
CX Linear-Diabatic model (LandauZenerLD)	Only standard Landau-Zener (landau_zener_cross_section)
CXTable (full CX charge-state matrix)	Per-channel only
MPI parallelism (InitializeMPI/FinalizeMPI)	AtomX is serial; FAC has MPI scaling
🟢 Module-level coverage that's already there (just hidden)
These exist in AtomX but were missing in the audit because the wrappers were incomplete or misleading:

CRM (plasma/crm.py) — CollisionalRadiativeModel class, 332 lines; FAC equivalent (DRBranch / MAIRates / MCERates / SpecTable). Likely sufficient for level populations but not benchmarked vs FAC's crm.c.
MaxwellRate — plasma/rates.py:maxwellian_rate is wired.
Higher multipoles (M1, E2, M2…) — einstein_A_eL/mL formula handles arbitrary L; only E1 has been benchmarked.
Verdict
After the 5 fixes:

Physics correctness: PI/RR ✅, EII ✅, CE structurally ✅, AI/DR wired ✅, MCDHFResult parity ✅
Module-level holes: R-matrix is the only completely empty box.
Feature gaps for niche users: two-photon decay, external-field Hamiltonians, M-substate resolved cross sections, MBPT-for-transitions, average-atom — all are real FAC features AtomX doesn't ship yet.
Production-grade caveats: the new auger_rate uses a simplified angular factor (need full CSF coupling for quantitative AI/DR); high-Z K-shell PI/CE/CI need finer inner B-spline grid or direct radial integration.
If you'd like, I can prioritize and implement any of these (R-matrix and two-photon decay are the cleanest one-week features; MBPT-for-transitions is the highest-leverage for precision spectroscopy).