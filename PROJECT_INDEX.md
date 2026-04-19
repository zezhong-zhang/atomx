# Project Index: atomx

Generated: 2026-04-19

> High-performance relativistic atomic structure calculations using B-spline Dirac-Fock.
> Python 3.10+ · numpy/scipy/numba · pint units (optional) · matplotlib/plotly/manim viz.

## 📁 Project Structure

```
atomx/
├── src/atomx/
│   ├── __init__.py            # Public API re-exports (Atom, Methods, units, …)
│   ├── atom.py                # High-level Atom facade (802 lines)
│   ├── types.py               # Enums: NuclearModel, Exchange, Correlation, Multipole, Gauge, …
│   ├── results.py             # AtomicResult, HyperfineResult, OrbitalWavefunction, Waves
│   ├── constants.py           # CODATA 2022 physical constants
│   ├── units.py               # pint-based unit handling (pint-optional)
│   ├── compare.py             # compare, theory_ladder, what_changed
│   ├── log.py                 # Logging helpers
│   ├── methods/               # Theory-level Method objects (Hydrogenic → MCDHF, CC, RPA, MBPT, CI, KS)
│   │   ├── base.py            #   abstract Method
│   │   ├── hydrogenic.py      #   exact: HydrogenicSchrodinger / HydrogenicDirac
│   │   ├── hartree_fock.py    #   non-rel SCF (LS-coupling)
│   │   ├── relativistic_hf.py #   BreitPauli (HF + perturbative rel.)
│   │   ├── dirac_fock.py      #   DiracFock — primary B-spline DHF engine (2328 lines)
│   │   ├── scf.py             #   shared SCF glue
│   │   ├── mcdhf/             #   multi-configuration DHF: config, excitations, matrix, solver
│   │   ├── ci.py, coupled_cluster.py, mbpt.py, rpa.py, kohn_sham.py
│   ├── numerics/              # Low-level numerical kernels
│   │   ├── bspline/           #   knots, basis, evaluate, integrals, convol, dirac_assembly, continuum
│   │   ├── shooting/          #   ODE bound-state shooting (Dirac + Schrödinger)
│   │   └── analytic/          #   closed-form hydrogenic_dirac
│   ├── physics/               # Physics operators: dirac, schrodinger, breit_pauli, two_body, base
│   ├── angular/               # wigner, clebsch, coefficients, ls_coupling, recoupling, spinor_harmonics
│   ├── config/                # Electron configurations, jj_coupling
│   ├── corrections/           # Breit, QED stack (adapters, protocol, stack, qedmod/{shabaev,welton,mohr,flambaum_ginges})
│   ├── potentials/            # nuclear (Fermi/uniform/point), thomas_fermi, parametric (GSZ)
│   ├── prop/                  # Derived properties / observables
│   │   ├── transitions/       #   multipole (E1/M1/E2…), angular, rates, calculator
│   │   ├── hyperfine/         #   dipole A, quadrupole B, nuclear moments, QED corrections
│   │   ├── isotope_shift/     #   field_shift, mass_shift, sms_abinitio
│   │   ├── zeeman/            #   Landé g-factors, shifts
│   │   ├── g_factor/, polarizability/, edm/, pnc/   # (stubs / in progress)
│   ├── scattering/            # Continuum processes
│   │   ├── photoionization/   #   fano
│   │   ├── autoionization/    #   auger
│   │   ├── dielectronic_recombination/solver.py
│   │   ├── distorted_wave/solver.py
│   │   └── r_matrix/          # (stub)
│   ├── plasma/                # Screening (ion-sphere/Debye), collisional-radiative model
│   ├── lineshapes/            # Voigt, Stark profiles
│   ├── spectrum/              # Spectrum synthesis (transition lists + CRM)
│   ├── uta/                   # Unresolved-transition-array moments
│   ├── fitting/               # Slater-type orbital fits
│   ├── viz/                   # mpl, plotly, plotly3d, manim back-ends
│   └── data/                  # hyperfine_presets, mohr_se, flambaum_ginges, nuclear_structure
├── tests/                     # pytest suite (30 modules)
├── scripts/                   # benchmark_energies, compare_energies, generate_density_gs, density_plots/
├── docs/gallery/              # runnable examples (grotrian, radial_orbital, spectrum)
├── papers/                    # Reference PDFs
├── media/                     # Generated figures / animations
├── pyproject.toml             # Hatchling build, deps, pytest config
└── uv.lock                    # uv-managed lockfile
```

## 🚀 Entry Points

- **Public API** — `src/atomx/__init__.py` — `Atom`, `HydrogenicSchrodinger`, `HydrogenicDirac`, `HartreeFock`, `BreitPauli`, `DiracFock`, `MCDHF`, `compare`, `theory_ladder`, `what_changed`, `ureg`.
- **High-level facade** — `src/atomx/atom.py` — `Atom(Z, config=..., breit=..., qed=..., exchange=..., correlation=..., nuclear=..., plasma_ne=..., Te=...).solve(Method?)`.
- **Primary SCF engine** — `src/atomx/methods/dirac_fock.py` (2328 lines) — B-spline Dirac-Fock with Breit + QED.
- **Benchmarks** — `scripts/benchmark_energies.py`, `scripts/compare_energies.py`, `scripts/generate_density_gs.py`.
- **Examples** — `docs/gallery/{grotrian,radial_orbital,spectrum}_example.py`.
- **Tests** — `pytest` (discovery from `pyproject.toml`).

## 📦 Core Modules

### `atomx.atom` — `src/atomx/atom.py`
High-level `Atom` class. Knobs: `Z`, `config`/`configs`, `charge`, `ns`/`ks` (B-spline count/order), `nuclear` ∈ {fermi, point, uniform}, `breit`, `qed`, `exchange` ∈ {hf, slater, shooting, tfda, gsz}, `correlation` ∈ {None, s, sd}, `plasma_ne`/`Te`. Drives a `Method` (default `DiracFock`) and returns `AtomicResult`.

### `atomx.methods`
Method objects encode theory level and are mutually comparable via `theory_ladder`:
- `base.Method` (abstract)
- `hydrogenic.HydrogenicSchrodinger` / `HydrogenicDirac` (exact)
- `hartree_fock.HartreeFock` (non-rel SCF, LS-coupling)
- `relativistic_hf.BreitPauli` (HF + perturbative rel.)
- `dirac_fock.DiracFock` (four-component B-spline, jj-coupling) — main engine
- `mcdhf/` (CSF-based multi-configuration DHF; config/excitations/matrix/solver)
- `ci`, `coupled_cluster`, `mbpt`, `rpa`, `kohn_sham` (scaffolds / specialized variants)

### `atomx.numerics`
- `bspline/` — semi-exponential DBSR-HF-style grid, Dirac Hamiltonian assembly, continuum.
- `shooting/` — reference ODE integrators (Dirac + Schrödinger bound/integrator/power-series) used for cross-validation.
- `analytic/hydrogenic_dirac.py` — Sommerfeld closed-form wavefunctions.

### `atomx.physics`
Standalone operator definitions: `dirac`, `schrodinger`, `breit_pauli`, `two_body`, `base`.

### `atomx.corrections`
`breit.py` (Gaunt + full Breit), `qed.py`, `qedmod/` (Shabaev / Welton / Mohr / Flambaum-Ginges local QED potentials) plus a `stack`/`protocol`/`adapters` layer.

### `atomx.prop` — observables
- `transitions/` — multipole operators, angular factors, rates.
- `hyperfine/` — dipole A, quadrupole B, nuclear moments, QED corrections.
- `isotope_shift/` — field shift + NMS/SMS (including ab-initio closed-shell SMS).
- `zeeman/` — Landé g-factors + shifts.
- `g_factor/`, `polarizability/`, `edm/`, `pnc/` — stubs / in progress.

### `atomx.scattering`
- `photoionization/fano.py`, `autoionization/auger.py`, `dielectronic_recombination/solver.py`, `distorted_wave/solver.py`, `r_matrix/` (stub).

### Other
- `plasma/` (screening + CRM), `lineshapes/`, `spectrum/`, `uta/moments`, `fitting/slater`, `potentials/`, `angular/`, `config/jj_coupling`, `viz/`.
- `data/` — lookup tables: `hyperfine_presets`, `mohr_se`, `flambaum_ginges`, `nuclear_structure`.

## 🔧 Configuration

- `pyproject.toml` — project metadata, dependencies, `testpaths = ["tests"]`, wheel packages `src/atomx`.
- `uv.lock` — uv-resolved lockfile.

## 📚 Documentation

- `docs/gallery/` — runnable examples: `grotrian_example.py`, `radial_orbital_example.py`, `spectrum_example.py`.
- `papers/` — reference PDFs (Shabaev QED, Tupitsyn, arXiv 0507067/0610152, 2301.04132).
- `trace.md` — rolling EARS/session trace.

## 🧪 Test Coverage

30 pytest modules under `tests/`:
angular · atom_api · atom_unified_api · autoionization · bspline · ci · constants · continuum · convol · corrections_stack · data_loaders · dirac_hydrogenic · fitting · hyperfine · isotope · lineshapes · mcdhf · photoionization · plasma · potentials · qed · qedmod · scaffolds · scf · spectrum_and_uta · transitions · units · viz · viz_plotly3d · zeeman.

Run: `pytest` (discovery from `pyproject.toml`).

## 🔗 Key Dependencies

Required: `numpy>=1.24`, `scipy>=1.10`, `numba>=0.58`, `h5py>=3.0`, `plotly>=6.7`, `manim>=0.19.1`, `matplotlib>=3.10.8`, `nbformat`, `ipython`, `ipykernel`.

Optional extras: `dev` (pytest, pytest-benchmark), `units` (pint), `viz`/`viz-manim`/`viz-all`, `all`.

## 📝 Quick Start

```bash
# Install (editable, dev + units extras)
pip install -e ".[dev,units]"
# or: uv sync

# Minimal example
python -c "
from atomx import Atom
ne = Atom(Z=10); ne.solve(); ne.summary()
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
