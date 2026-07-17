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

## Quick start

```python
from atomx import Atom

# Neutral neon with default Dirac-Fock SCF
atom = Atom(Z=10)
atom.solve()
print(atom.total_energy)       # -128.04 Hartree
print(atom.orbital_energies)   # {1s: -32.6, 2s: -1.6, ...}

# With Breit + QED corrections
atom = Atom(Z=10, breit=True, qed=True)
atom.solve()

# Multi-configuration (Be with 2s² and 2p² mixing)
atom = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
atom.solve()

# Numerics engine and exchange model are independent choices:
atom = Atom(Z=26, charge=16, config="1s2 2s2 2p6",
          numerics="shooting", exchange="gu")     # FAC-style local DFS, ~50 ms
atom = Atom(Z=26, charge=16, config="1s2 2s2 2p6",
          numerics="bspline", exchange="slater", alpha_x=2/3)
atom.solve()
print(atom.provenance)   # {'numerics': 'bspline', 'exchange': 'slater'}
```

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
| `units` | Pint-based unit handling |
| `viz` | matplotlib + Plotly visualization |
| `viz-manim` | Manim animation backend |
| `viz-all` | All visualization backends |
| `all` | Everything |

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

The scattering and spectroscopy core is verified point-by-point against a
live Flexible Atomic Code (FAC) build under a clean-room protocol
(matched physics flags, exact-mode integrals, pinned energy grids):

| Quantity | Parity vs FAC |
|----------|---------------|
| Continuum waves (100 eV – 200 keV, l = 0–5, Fe/Xe) | amplitude √(2/π)·(1 ± 0.01), shape 0.001 % in matched potentials |
| Collisional excitation Ω (He-like Z = 10/26/54, 1.05–5× threshold) | rms 3.7 %, every point ≤ 6.7 % |
| Electron-impact ionization (K-shell, vs exact-mode FAC) | ≤ 6 % at 5× threshold for all Z; Z ≤ 26 within 15 % everywhere |
| Auger widths (Li-like KLL, per level) | 1.005–1.023 |
| DR radiative widths (Fe XVII dr33, per level) | 1.09–1.27 |
| DR rate coefficient α(T) (complete clean-room dr33 set) | 0.98–1.17 at 5–50 MK (rms 0.27 incl. the documented 1-MK E_res offset) |

`benchmarks/` regenerates these live (`bench_fac_dw_excitation.py`,
`bench_fac_dw_eii.py`, `bench_fac_dr.py`); `pytest -m fac_parity` gates
them. See `benchmarks/REPORT.md` and
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
# Run tests
pytest

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run benchmarks
pytest -m fac_parity
```

## License

See `pyproject.toml` for project metadata.
