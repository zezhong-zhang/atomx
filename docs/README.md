# atomx Documentation

## Guides

- [Migration Notes](MIGRATION.md) — Version migration notes (e.g., QED backend defaults)

## Gallery

Runnable example scripts in `gallery/`:

| Example | Description | Run command |
|---------|-------------|-------------|
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
