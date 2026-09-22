# atomx Documentation

## Reviews and audits

- [Physics & numerics review (2026-09-19)](PHYSICS_NUMERICS_REVIEW_2026-09-19.md)
  and its [remediation report](PHYSICS_NUMERICS_REVIEW_2026-09-19_FIXES.md)
- [**`correlation="s"` / `correlation="sd"` audit (2026-09-20)**](CORRELATION_SD_RESEARCH_2026-09-20.md)
  — root cause of the open N1 finding: `build_ci_matrix` is not a
  representation of the Hamiltonian on the CSF basis, so the feature is not
  variational. Includes the independent m-scheme Slater–Condon CI oracle used
  as arbiter (`benchmarks/mscheme_ci_reference.py`).
- [Critical review (2026-09-17)](CRITICAL_REVIEW_2026-09-17.md)
- [FEYNMAN review (2026-07-15)](FEYNMAN_REVIEW_2026-07-15.md)

## Verification

- [Verification claim → gate table](VERIFICATION.md)
- `benchmarks/EXTERNAL_VERIFICATION.md` — live cross-verification against the
  codes in `~/code/ASAP`

## Guides

- [Migration Notes](MIGRATION.md) — Version migration notes (e.g., QED backend defaults)

## Gallery

Runnable example scripts in `gallery/`:

| Example | Description | Run command |
|---------|-------------|-------------|
| [atom_inputs_example.py](gallery/atom_inputs_example.py) | The ten canonical `Atom` arguments, `resolved_inputs`, plasma screening, and the validation errors | `python docs/gallery/atom_inputs_example.py` |
| [ladder_example.py](gallery/ladder_example.py) | Climb the theory ladder rung-by-rung for Ne | `python docs/gallery/ladder_example.py` |
| [local_exchange_comparison.py](gallery/local_exchange_comparison.py) | Compare HF vs Slater vs FAC/Gu exchange for Ne | `python docs/gallery/local_exchange_comparison.py` |
| [radial_orbital_example.py](gallery/radial_orbital_example.py) | Animate Ne 2p3/2 radial wavefunction | `manim -pqh docs/gallery/radial_orbital_example.py NeRadial2p` |
| [grotrian_example.py](gallery/grotrian_example.py) | Grotrian energy-level diagram with transitions | `manim -pqh docs/gallery/grotrian_example.py HydrogenGrotrian` |
| [spectrum_example.py](gallery/spectrum_example.py) | Synthetic spectrum build-up animation | `manim -pqh docs/gallery/spectrum_example.py NeSpectrum` |

## Plans

Development and implementation plans:

- [FAC Parity Implementation Plan](plans/FAC_PARITY_IMPLEMENTATION_PLAN.md) — Roadmap for FAC-equivalent physics
- [FAC Calculator Plan](plans/2026-04-20-fac-calculator.md) — Detailed FAC calculator design
- [FAC Calibration Findings](plans/2026-04-20-fac-calibration/FINDINGS.md) — Calibration results against FAC reference data
- [Feature Gaps](plans/gaps.md) — Feature gap analysis

## Benchmarks

See `benchmarks/REPORT.md` for the full FAC parity benchmark report.
