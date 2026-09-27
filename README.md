# atomx

High-performance atomic structure calculations, from textbook Schrodinger & Dirac to frontier QED.

## Overview

atomx is a Python library for computing atomic energy levels, wavefunctions, transition rates, hyperfine structure, and many other atomic properties. It implements a complete theory ladder — from simple hydrogenic models through multi-configuration Dirac-Hartree-Fock (MCDHF) with Breit and QED corrections — using B-spline basis sets and JIT-compiled numerical kernels.

## Features

- **Theory ladder** — Hydrogenic, Hartree-Fock, Breit-Pauli, Dirac-Fock, Dirac-Fock+Breit, Dirac-Fock+QED, MCDHF, MBPT(2), all-order pp-ladder pair correlation
- **Decoupled engine × model** — choose the radial numerics (`bspline` B-spline Galerkin or `shooting` Salvat ODE) independently of the exchange model (`hf` non-local, `slater` with any α, `gu` FAC-style empirical α_xc)
- **Atomic properties** — energy levels, orbital wavefunctions, transition rates (E1/M1/E2/M2), hyperfine structure (A, B), isotope shifts, Zeeman effect, g-factors, polarizability, EDM, PNC
- **Scattering** (FAC-verified distorted wave) — collisional excitation with exact-GOS Born completion, electron-impact ionization, excitation-autoionization, autoionization (Auger), ab-initio dielectronic recombination with density suppression, resonant excitation, photoionization (Fano), radiative recombination, charge exchange, R-matrix
- **Plasma modeling** — Debye/ion-sphere screening, collisional-radiative model (CRM) consuming the verified DW collision strengths, average-atom model, cascades
- **Spectrum synthesis** — CRM lines + free-free (bremsstrahlung) + free-bound (RR) continuum channels, UTA (unresolved transition arrays)
- **QED corrections** — vacuum polarization, self-energy (Mohr, Welton, Indelicato, Shabaev-local, Flambaum-Ginges backends), pluggable `CorrectionStack`
- **Visualization** — matplotlib, Plotly, and Manim backends for energy levels, radial orbitals, Grotrian diagrams, spectra, cross sections with reference-ratio panels, DR resonance spectra

The [NIST spectrum atlas](docs/NIST_SPECTRUM_ATLAS.md) builds an auditable line and level catalog, interactive periodic table and ion viewer. Its [theory comparison](docs/THEORY_NIST_COMPARISON.md) shows calculated versus NIST wavelengths, ionization energies, and complete evaluated configuration centroids on separate coverage-aware periodic maps.

## Quick start

```python
from atomx import Atom

# Neutral neon — Dirac-HF on the shooting engine with Breit + QED
# corrections, all on by default
atom = Atom(Z=10)
atom.solve()
print(atom.total_energy)       # -128.61 Hartree
print(atom.orbital_energies)   # {'1s1/2': -32.82, '2s1/2': -1.94, '2p1/2': -0.85, ...}

# Correction-free (pure Dirac-Coulomb) reference run
atom = Atom(Z=10, corrections="none")
atom.solve()

# Multi-configuration CI, including two-electron replacements.  1s2 and
# 1s1 2s1 are both 1S0 (J=0, even) and couple through the Hamiltonian.
# A *list* of configurations selects the MCDHF/CI path.
atom = Atom(Z=2, config=["1s2", "1s1 2s1"])
atom.solve()
print(atom.total_energy)                        # -2.8629 Hartree (HF: -2.8617)
print(atom.result.mixing_coefficients[:, 0])    # [-0.999, 0.042, 0.0]

# Same-symmetry *double* replacements couple too (the 2s²/2p²
# near-degeneracy pair): the ground state mixes 2s² with both 2p² jj pairs
# and drops 0.041 Ha below the single-configuration HF energy.
atom = Atom(Z=4, config=["1s2 2s2", "1s2 2p2"])
atom.solve()
print(atom.total_energy)                        # -14.6139 Hartree
print(atom.result.mixing_coefficients[:, 0])    # [0.944, 0.282, 0.0, 0.0, 0.0, 0.171]

# The radial ENGINE and the exchange MODEL are independent choices.  A
# knob that only exists for one model (alpha_x, slater_method) travels
# with that model on the Exchange object.
from atomx import Exchange

atom = Atom(Z=26, charge=16, config="1s2 2s2 2p6",
            engine="shooting", exchange="gu")     # FAC-style local DFS
atom = Atom(Z=26, charge=16, config="1s2 2s2 2p6",
            engine="bspline",
            exchange=Exchange(model="slater", alpha_x=2/3))
atom.solve()
print(atom.provenance)   # {'engine': 'bspline', 'exchange': 'slater', 'averaging': 'statistical'}
```

### The ten constructor arguments

`Atom` takes ten arguments, one per physical question.  Five of them accept
either a plain value (the common case) or a frozen dataclass from
`atomx.atom_inputs` when there is more to say:

| Argument | Plain form | Object form | What it decides |
|----------|-----------|-------------|-----------------|
| `Z` | `10` | — | nuclear charge (required, positional) |
| `config` | `"1s2 2s2 2p6"` | `["1s2 2s2", "1s2 2p2"]` for CI | which configurations enter |
| `charge` | `0` | — | net ion charge, so `N = Z - charge` |
| `nuclear` | `"fermi"` | — | nuclear charge distribution |
| `exchange` | `"hf"` / `"slater"` / `"gu"` | `Exchange(model=..., alpha_x=..., slater_method=...)` | exchange model + the knobs only it reads |
| `corrections` | `"none"` / `"breit"` / `"qed"` / `"breit+qed"` | `Corrections(breit=..., qed=..., qed_backend=..., stack=...)` | Breit + QED, and the self-energy backend |
| `correlation` | `"s"` / `"sd"` | `Correlation(expansion=..., averaging=..., radial_optimization=..., j_resolved_csfs=..., per_block_orbital_sets=...)` | correlation expansion + CI/SCF protocol |
| `engine` | `"bspline"` / `"shooting"` | — | radial engine (was `numerics=`) |
| `basis` | — | `Basis(ns=..., ks=..., r_max=..., grid_hi=..., grid_he=..., grid_hmax=..., starting_potential=..., sampling=Sampling.high)` | radial basis, box, knot grid, accuracy tier |
| `environment` | — | `Environment(plasma_ne=..., plasma_Te=..., screening="debye"\|"stewart_pyatt")` | the surrounding plasma, screened inside the SCF |

There is exactly one spelling per question — the old flat keywords are gone,
and each one raises a message naming its replacement (see
[`docs/MIGRATION.md`](docs/MIGRATION.md)).

```python
from atomx import Atom, Basis, Corrections, Correlation, Environment

# Ne^2+ in a dense plasma (every argument spelled out)
atom = Atom(
    10,                                       # Z
    config="1s2 2s2",                         # reference configuration
    charge=2,                                 # net ion charge
    nuclear="fermi",                          # finite-size nucleus
    exchange="hf",                            # non-local HF exchange
    corrections=Corrections(breit=True, qed=True, qed_backend="auto"),
    correlation=Correlation(averaging="ci"),  # Guerra CI averaging
    engine="bspline",                         # Galerkin B-spline SCF
    basis=Basis(ns=200, ks=9, r_max=90.0),    # 200 B-splines, 90 Bohr box
    environment=Environment(plasma_ne=1e22, plasma_Te=100.0,
                            screening="debye"),   # free-electron sea
)
atom.solve()
print(atom.resolved_inputs)
# Z=10 charge=2 fermi | 1s2 2s2 | engine=bspline exchange=hf averaging=ci
# | breit=on qed=on(auto) | radial=bspline(ns=200, ks=9, grid=fixed)
#   r_max=90.0 | plasma(ne=1e+22, Te=100.0, debye)
```

The plasma is not decoration: `plasma_ne`/`plasma_Te` define the Debye length
`lambda_D = sqrt(kT / 4 pi n_e)` and the screened potential
`-Z exp(-r/lambda_D)/r` is added to the nuclear potential *inside* the SCF,
so every orbital moves up by ~`Z/lambda_D` (the ionisation potential
depression).  A neutral atom has no net charge to screen, so at these
densities the model predicts complete pressure ionisation and the solver
says so instead of returning an unscreened atom.

`atom.resolved_inputs` is a one-line summary of what was actually chosen:
which engine `engine=None` resolved to, whether Breit and QED are on, which
grid the run used, and whether the atom sits in a plasma.  Physically
impossible combinations raise with the argument named and the alternative
spelled out: `alpha_x` outside `Exchange(model="slater")`, `grid_*` with the
shooting engine (or with an explicit `ns`), `plasma_Te` without
`plasma_ne`, `j_resolved_csfs` without a multi-configuration input, or a
`Corrections(stack=...)` that contradicts an explicit `breit`/`qed`.  See
`docs/gallery/atom_inputs_example.py` for a runnable tour.

### Theory ladder

Climb the physics ladder rung-by-rung, seeing how each effect changes the total energy:

```python
from atomx import Atom, Ladder, Sampling
from atomx.methods import DiracFock

atom = Atom(Z=10)
run = atom.climb(through=DiracFock(breit=True, qed=True), sampling=Sampling.quick)
run.show(unit="eV")
```

### Compare methods

```python
from atomx import Atom, compare
from atomx.methods import HartreeFock, BreitPauli, DiracFock

results = compare(
    Atom(Z=10),
    [HartreeFock(), BreitPauli(), DiracFock(), DiracFock(breit=True, qed=True)],
)
```

## Installation

```bash
pip install -e .
```

Requires Python >=3.10. For development dependencies:

```bash
pip install -e ".[dev]"
```

### Optional extras

| Extra | Description |
|-------|-------------|
| `pyscf` | PySCF back-end for the electronic-structure providers (molecules, embedded clusters) |
| `units` | Pint-based unit handling |
| `viz` | matplotlib + Plotly visualization |
| `viz-manim` | Manim animation backend |
| `viz-all` | All visualization backends |
| `all` | Everything |

### External electronic-structure providers

Atomx can drive its Auger/CSF kernels from external electronic-structure
back-ends through the unified provider contract (`src/atomx/providers/`,
ADR-0001). PySCF support is optional:

```bash
pip install "atomx[pyscf]"
```

Entry points:

- `atomx.providers.pyscf_adapter.PySCFProvider` / `from_scf(mf)` — translate a
  converged PySCF mean field into a validated `ElectronicStructurePayload`.
- `atomx.providers.pyscf_adapter.build_core_hole_delta_scf_payload` —
  MOM-tracked ΔSCF core-hole states for molecules.
- `atomx.providers.pyscf_embedding` — cluster/point-charge embedding
  (`rocksalt_embedding`, `build_embedded_scf`), the local-orbital bridge
  (`core_orbital_identity`, `bridge_core_orbitals`) that keeps a central
  atom's core-orbital identity traceable across isolated atom / molecule /
  embedded cluster, embedded ΔSCF binding energies, and
  `study_cluster_convergence` for environment convergence studies.

Run the condensed-matter convergence benchmark (rock-salt MgO, Mg 1s):

```bash
python benchmarks/bench_pyscf_embedding_convergence.py
```

## Architecture

```
src/atomx/
  atom.py            # High-level Atom API
  types.py           # Enums (Exchange, TheoryLevel, Gauge, ...)
  results.py         # Data classes (AtomicResult, HyperfineResult, ...)
  constants.py       # CODATA 2022 physical constants
  methods/           # Theory-level Method objects (DiracFock, MCDHF, ...)
  numerics/          # B-spline basis, ODE shooting, convolutions (numba JIT)
  physics/           # Operator definitions (Dirac, Schrodinger, Breit-Pauli)
  angular/           # Wigner symbols, Clebsch-Gordan, recoupling algebra
  config/            # Electron configuration handling
  corrections/       # Breit + QED correction stack
  potentials/        # Nuclear charge distributions, statistical potentials
  properties/        # Observables (transitions, hyperfine, isotope shift, ...)
  scattering/        # Continuum processes (photoionization, DR, R-matrix, ...)
  plasma/            # CRM, screening, average-atom, cascades
  spectrum/          # Spectrum synthesis
  viz/               # matplotlib, Plotly, Manim backends
```

The layered design means you can use atomx at whatever level suits your needs:
- **High-level**: `Atom(Z=10).solve()` for quick calculations
- **Method-level**: Compose specific `Method` objects for fine-grained control
- **Numerics-level**: Use B-spline bases, shooting solvers, and angular algebra directly

## Verification against FAC

The scattering and spectroscopy core is developed against a Flexible Atomic
Code (FAC) build under a clean-room protocol (matched physics flags,
exact-mode integrals, pinned energy grids). **What is gated in CI, and what is
only a recorded result, is listed per row in [`docs/VERIFICATION.md`](docs/VERIFICATION.md)**
— read it before quoting any number below. In short:

| Quantity | Parity vs FAC | Gated by |
|----------|---------------|----------|
| Continuum waves (100 eV – 200 keV, l = 0–5, Fe/Xe) | amplitude √(2/π)·(1 ± 0.01), shape 0.001 % in matched potentials | `tests/test_continuum_vs_fac.py` — recomputes atomx against stored FAC dumps |
| Collisional excitation Ω (He-like Z = 10/26/54, 1.05–5× threshold) | **live-measured** rms 3.3 %, every point ≤ 5.2 % (2026-09-18) | `tests/test_fac_dw_benchmarks.py` — recomputes atomx for the grid and asserts both the recorded FAC column and the recorded atomx column; regenerating the FAC reference needs `sfac` |
| Electron-impact ionization (K-shell, vs exact-mode FAC) | **live-measured** rms 2.0 %, every point ≤ 3.1 % | same file |
| DR radiative widths and α(T) (Fe XVII dr33) | widths 1.09–1.27; α(T) ratios 0.98 / 1.08 / 1.16 at 5 / 10 / 50 MK (worst FAC deviation 16.4 %, incl. the documented E_res offset) | `tests/test_dr_manifold_e2e.py` (`slow`), `tests/test_dr_vs_fac.py` — recompute against the clean-room fixture |
| Auger widths (Li-like KLL) | 1.005–1.023 | **no executing gate for these ratios.** `tests/test_auger_fac_panel.py` computes atomx live for the five strongest Fe XVII KLL resonances and measures live/FAC = 1.13–14.8 (2026-09-17); the strict 10 % gate is `xfail`. The `1.005–1.023` figures in `docs/scattering_fac_verification_2026-07.md` are a development note, not a reproducible claim |

Two further caveats that the numbers above do not convey:

* **Structure vs FAC** (`benchmarks/bench_fac_structure.py`) is reported
  relative to *total* energy, which hides absolute disagreements.  Re-measured
  2026-09-19 with the flags matched to FAC's default (`breit=False, qed=True`,
  which is now the driver's default): the whole Z = 10…82 panel agrees to
  **≤ 9.6e-4 (0.096 %)**, the worst absolute row being **426 eV at Z = 82**
  (Ar-like Pb).  The previously documented "7 keV at Z=82 (1.6 %)" came from an
  atomx column that had gone stale for five months *and* an unmatched
  `qed=False`; see `docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md` finding H1.
  `tests/test_fac_structure_snapshot.py` now gates the recorded column.
* **The CRM/plasma benchmarks** (`bench_fac_crm_*.py`: Au/Fe/W cooling,
  charge-state balance, NLTE-9 ⟨Z⟩) are **synthetic placeholders**, disclosed
  in each file; the tests that asserted agreement with them are skipped with
  the reason spelled out. They are not validation.

`benchmarks/` regenerates the FAC references (`bench_fac_dw_excitation.py`,
`bench_fac_dw_eii.py`, `bench_fac_dr.py`) when a live `sfac` is configured
(`FAC_SFAC_PATH`, see `docs/VERIFICATION.md`). See `benchmarks/REPORT.md` and
`docs/scattering_fac_verification_2026-07.md` for the methodology.

## Performance

Numba-JIT kernels, bilinear R^k Slater tables, threaded wave prefill and
value-invariant caching keep production workloads interactive (32-core
desktop, warm):

- ground-state DHF: Ne 0.86 s, Fe XVII 0.58 s (B-spline HF); 50 ms via the
  shooting engine with Gu exchange
- DW excitation: ~20 ms per energy point (He-like Fe, κ ≤ 10)
- DW ionization: 0.23 s per energy point at production settings
- MBPT(2)/all-order pair ladders: R^k tables via GEMM (28× over scalar)

## Documentation

- `docs/MIGRATION.md` — Migration notes between versions
- `docs/gallery/` — Runnable example scripts
- `benchmarks/REPORT.md` — FAC parity benchmark report

## Development

```bash
# Run tests (fast: excludes the slow integration tests)
pytest

# Everything, including the slow suite and the book claims
pytest -m ""

# Detectors, by what they assert
pytest -m external_fac   # atomx vs stored external FAC reference data
pytest -m snapshot       # atomx vs its own recorded output (drift detector)
pytest -m book           # executable claims from docs/book

# Numerics gate: N fresh processes must agree bit-for-bit, and no numba
# kernel may read out of bounds (see docs/VERIFICATION.md)
python scripts/check_determinism.py --repeat 8 --panel full
python scripts/check_determinism.py --repeat 1 --panel full --boundscheck
```

## Contributing checks

The CI runs `ruff check --no-cache src tests scripts benchmarks` with the rule
set in `pyproject.toml`, `pytest -m "not slow"` on Python 3.10 and 3.13, and
the determinism gate. `pre-commit install` wires the same ruff selection into
your commits.

## License

No license has been chosen for this repository yet — `pyproject.toml` carries
no `license` field and there is no `LICENSE` file, so the default "all rights
reserved" applies. Pick one before distributing the package.
