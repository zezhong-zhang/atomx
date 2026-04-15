# Project Index: atomx

Generated: 2026-04-15

> High-performance relativistic atomic structure calculations using B-spline Dirac-Fock.
> Python 3.10+ · numpy/scipy/numba · pint units · optional matplotlib/plotly/manim viz.

## 📁 Project Structure

```
atomx/
├── src/atomx/              # Library source (hatchling-packaged)
│   ├── __init__.py         # Public API re-exports (Atom, Methods, units, …)
│   ├── atom.py             # High-level Atom facade
│   ├── types.py            # Enums: NuclearModel, Exchange, Correlation, …
│   ├── results.py          # AtomicResult, HyperfineResult, Wavefunctions
│   ├── constants.py        # CODATA 2022 physical constants
│   ├── units.py            # pint-based unit handling (optional)
│   ├── compare.py          # theory_ladder, what_changed, compare
│   ├── log.py              # Logging helpers
│   ├── methods/            # Theory-level Method objects (Hydrogenic→MCDHF)
│   ├── scf/                # Self-consistent field drivers (DFSolver, Slater)
│   ├── dirac/              # Dirac equation: hydrogenic, matrix, continuum, solver
│   ├── mcdhf/              # Multi-configuration Dirac–Hartree–Fock
│   ├── shooting/           # ODE shooting solvers (Schrödinger + Dirac)
│   ├── bspline/            # B-spline basis, knots, integrals, convolutions
│   ├── angular/            # Wigner/Clebsch, LS/jj recoupling, coefficients
│   ├── config/             # Electron configurations, jj-coupling
│   ├── potentials/         # Nuclear, Thomas–Fermi, parametric (GSZ)
│   ├── corrections/        # Breit, QED (qedmod/Welton)
│   ├── transitions/        # Multipole rates (E1/M1/E2…), angular factors
│   ├── hyperfine/          # Dipole, quadrupole, nuclear moments, QED
│   ├── autoionization/     # Auger, dielectronic recombination
│   ├── photoionization/    # Fano profiles, photoionization
│   ├── isotope/            # Field shift, SMS
│   ├── zeeman/             # Landé g-factors, shifts
│   ├── plasma/             # Screening, collisional-radiative model (CRM)
│   ├── collisions/         # Distorted-wave, rate coefficients
│   ├── lineshapes/         # Voigt, Stark profiles
│   ├── spectrum/           # Spectrum synthesis, CRM synthesis
│   ├── uta/                # Unresolved-transition-array moments
│   ├── fitting/            # Slater-type orbital fitting
│   └── viz/                # matplotlib / plotly / manim back-ends
├── tests/                  # pytest suite (24 modules)
├── scripts/                # Benchmarks + density_plots
├── docs/gallery/           # Example scripts (Grotrian, radial, spectrum)
├── papers/                 # Reference PDFs (Shabaev, Tupitsyn, …)
├── media/                  # Generated figures / animations
├── pyproject.toml          # Hatchling build, deps, pytest config
└── uv.lock                 # uv-managed lockfile
```

## 🚀 Entry Points

- **Public API** — `src/atomx/__init__.py` — `Atom`, `DiracFock`, `MCDHF`, `HartreeFock`, `HydrogenicDirac`, `HydrogenicSchrodinger`, `BreitPauli`, `compare`, `theory_ladder`, `ureg`.
- **High-level facade** — `src/atomx/atom.py:33` — `Atom(Z, config=..., breit=..., qed=..., exchange=..., correlation=..., …).solve()`.
- **Methods** — `src/atomx/methods/__init__.py` — pedagogical theory ladder (Hydrogenic → BreitPauli → DHF → MCDHF).
- **Benchmarks** — `scripts/benchmark_energies.py`, `scripts/compare_energies.py`.
- **Examples** — `docs/gallery/{grotrian,radial_orbital,spectrum}_example.py`.
- **Tests** — `tests/test_*.py` (pytest; configured in `pyproject.toml`).

## 📦 Core Modules

### `atomx.atom` — `src/atomx/atom.py` (481 lines)
High-level `Atom` class. Knobs: `Z`, `config/configs`, `charge`, `ns/ks` (B-spline count/order), `nuclear` ∈ {fermi, point, uniform}, `breit`, `qed`, `exchange` ∈ {hf, slater, shooting, tfda, gsz}, `correlation` ∈ {None, s, sd}, `plasma_ne/Te`. Drives `DFSolver`, returns `AtomicResult`.

### `atomx.methods` — `src/atomx/methods/`
Method objects encode theory level:
- `base.Method` (abstract)
- `hydrogenic.HydrogenicSchrodinger` / `HydrogenicDirac` (exact)
- `hartree_fock.HartreeFock` (non-rel SCF, LS-coupling)
- `relativistic_hf.BreitPauli` (HF + perturbative rel.)
- `dirac_fock.DiracFock` (four-component, jj-coupling)
- `mcdhf.MCDHF` (multi-configuration with correlation)

### `atomx.scf` — `src/atomx/scf/`
- `solver.DFSolver` (2307 lines) — primary B-spline Dirac–Fock engine.
- `slater` — local Slater-exchange variant.

### `atomx.dirac` — `src/atomx/dirac/`
Dirac equation kernels: `hydrogenic` (Sommerfeld + wavefunctions), `matrix` (B-spline Hamiltonian), `solver`, `continuum`.

### `atomx.mcdhf` — `src/atomx/mcdhf/`
CSF-based multi-configuration DHF: `config`, `excitations`, `matrix`, `solver`.

### `atomx.shooting` — `src/atomx/shooting/`
Reference ODE integrators (Schrödinger + Dirac bound/integrator/power-series) with grid, spline, quadrature utilities. Used for cross-validation against B-spline results.

### `atomx.bspline` — `src/atomx/bspline/`
`knots`, `basis`, `evaluate`, `integrals`, `convol` — semi-exponential DBSR-HF-style grid.

### `atomx.angular` — `src/atomx/angular/`
`wigner` (3j/6j/9j), `clebsch`, `coefficients`, `ls_coupling`, `recoupling`.

### `atomx.corrections` — `src/atomx/corrections/`
`breit.py` (Gaunt + full Breit), `qed.py`, `qedmod/{base,welton}` (Shabaev-style local QED potential).

### Physics modules
- `transitions/` — multipole (E1/M1/E2…), angular factors, rates calculator.
- `hyperfine/` — dipole A, quadrupole B, nuclear moments, QED corrections.
- `autoionization/` — Auger + dielectronic recombination.
- `photoionization/fano.py` — Fano resonance profiles.
- `isotope/field_shift.py` — normal/specific mass shift + field shift (ab-initio closed-shell SMS from second-quantized operator).
- `zeeman/` — Landé g-factors + shifts.
- `plasma/` — screening (ion-sphere/Debye), collisional-radiative model.
- `collisions/` — distorted-wave cross sections, thermal rates.
- `lineshapes/` — Voigt, Stark profiles.
- `spectrum/` — synthesis from transition lists + CRM.
- `uta/moments.py` — unresolved-transition-array moments.
- `fitting/slater.py` — STO fits.
- `potentials/` — nuclear (Fermi/uniform/point), Thomas–Fermi, GSZ.
- `viz/` — matplotlib, plotly, manim back-ends.

## 🔧 Configuration

- `pyproject.toml` — project metadata, dependencies, pytest `testpaths = ["tests"]`, wheel packages `src/atomx`.
- `uv.lock` — uv-resolved lockfile.
- `.gitignore` — standard Python ignores.

## 📚 Documentation

- `docs/gallery/` — runnable examples:
  - `grotrian_example.py` — level diagram.
  - `radial_orbital_example.py` — orbital plots.
  - `spectrum_example.py` — synthesis pipeline.
- `papers/` — reference PDFs used by the code (Shabaev QED, Tupitsyn, arXiv 0507067/0610152, 2301.04132).

## 🧪 Test Coverage

24 pytest modules under `tests/`:
angular · atom_api · autoionization · bspline · constants · continuum · convol · dirac_hydrogenic · fitting · hyperfine · isotope · lineshapes · mcdhf · photoionization · plasma · potentials · qed · qedmod · scf · spectrum_and_uta · transitions · units · viz · zeeman.

Run: `pytest` (test discovery from `pyproject.toml`).

## 🔗 Key Dependencies

Required:
- `numpy>=1.24` — arrays / linalg.
- `scipy>=1.10` — sparse eigen, special functions.
- `numba>=0.58` — JIT for hot loops (B-spline integrals, etc.).
- `h5py>=3.0` — I/O.
- `plotly>=6.7.0`, `manim>=0.19.1` — baseline viz (also split out as extras).

Optional extras: `dev` (pytest, pytest-benchmark), `units` (pint>=0.21), `viz`/`viz-manim`/`viz-all`, `all`.

## 📝 Quick Start

```bash
# Install (editable, dev + units extras)
pip install -e ".[dev,units]"
# or: uv sync

# Minimal example
python -c "
from atomx import Atom
ne = Atom(Z=10)
ne.solve()
ne.summary()
"

# Theory ladder on the same system
python -c "
from atomx import Atom, HydrogenicDirac, HartreeFock, DiracFock, theory_ladder
theory_ladder(Atom(Z=10), [HydrogenicDirac(), HartreeFock(), DiracFock(breit=True)])
"

# Tests
pytest
```

---
*Auto-generated index — refresh with `/sc:index-repo`.*
