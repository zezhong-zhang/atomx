# AtomX vs FAC Benchmark Summary

Run date: 2026-05-04  
Workspace: current `atomx` working tree  
FAC binary: `/Users/zhangzz/code/ASAP/fac/bin/bin/sfac`

Commands run:

```bash
uv run python benchmarks/bench_local_exchange.py
uv run python benchmarks/bench_fac_structure.py --z 10 18 26 36 54 74 82 --iso He-like Ne-like Ar-like --qed
uv run python benchmarks/bench_fac_lithium_like.py --z 10 18 26 36 54 74 82 92
uv run python benchmarks/bench_fac_levels_transitions.py
uv run python benchmarks/bench_fac_pi_rr.py
uv run python benchmarks/bench_fac_ce_ci_ai.py
```

## Summary

| Capability | Rows | Accuracy vs FAC | Speed |
|---|---:|---|---|
| Local-exchange DFS totals, `Atom(exchange="fac")` | 10 | median relative error `9.06e-05`; max `9.44e-04` | AtomX median `6.17x` faster than `sfac` (`5.02x` to `11.3x`) |
| Ground-state structure + QED, He/Ne/Ar-like | 20 | median relative error `2.46e-03`; max `3.16e-02` | AtomX median `2.38x` slower than FAC (`1.42x` to `4.43x`) |
| Li-like ground energy + 2s-2p rates | 8 | median ground-energy delta `53 eV`; max `1113 eV` | AtomX median `3.16x` slower than FAC |
| Fe XVII levels/E1 transitions | 6 AtomX transitions vs 24 FAC transitions | MCDHF basis still too coarse for FAC-level spectra; A ratios range from severe undercount to large overcount | AtomX `12.38 s`; FAC `0.573 s`; AtomX `21.6x` slower |
| PI/RR | 24 | He-like Fe RR median `4.71x`, max `23.2x`; Li-like Fe RR median `62.2x`, max `973x` | Script total `~2.6 s`; FAC subruns `0.21 s` and `0.50 s` |
| CE, He-like Fe | 12 | median relative difference `5.11e+02`; max `7.49e+03` | FAC subrun `1.35 s`; AtomX dominates total runtime |
| CI/EII, Fe XVII | 24 | median relative difference `1.0`; max `3.9` | FAC subrun `0.45 s`; AtomX dominates total runtime |
| AI/Auger, Ne-like Se | 5 | AtomX coarse wrapper returned `nan` for selected FAC top rates; not production-comparable yet | FAC subrun `0.65 s` |
| Higher multipoles, He-like Fe | 2 FAC rows | FAC rows parsed; AtomX comparison not populated in CSV yet | FAC subrun `0.27 s` |

## Structure Accuracy by Z Band

| Band | Median relative error | Max relative error |
|---|---:|---:|
| `Z <= 18` | `1.72e-04` | `4.40e-04` |
| `26 <= Z <= 36` | `1.42e-03` | `3.19e-03` |
| `Z >= 54` | `1.30e-02` | `3.16e-02` |

Interpretation: structure is good through low/mid Z, with high-Z gaps dominated by QED/self-energy model differences and heavy-ion basis/physics choices.

## Raw Outputs

- `benchmarks/results_local_exchange.csv`
- `benchmarks/results_fac_structure.csv`
- `benchmarks/results_fac_lilike.csv`
- `benchmarks/results_fac_levels_levels.csv`
- `benchmarks/results_fac_levels_transitions.csv`
- `benchmarks/results_fac_pi_rr.csv`
- `benchmarks/results_fac_ce.csv`
- `benchmarks/results_fac_ci.csv`
- `benchmarks/results_fac_ai.csv`
- `benchmarks/results_fac_higher_multipoles.csv`

## Main Takeaways

1. AtomX is already strong where the physics/numerics are matched to FAC: local-exchange DFS totals are below `1e-3` relative and faster than `sfac`.
2. Ground-state structure is usable through mid-Z but diverges at high Z unless QED/Breit model choices are made more FAC-like.
3. Scattering/continuum capabilities are now runnable and decoupled, but high-Z PI/RR/CE accuracy is limited by continuum construction. FAC’s direct energy-normalized continuum remains the target.
4. EII/CI auto-mode is stable and order-of-magnitude comparable; median error is about a factor of two.
5. Full CSF coupling remains the blocker for FAC-level Fe XVII spectra and AI/DR magnitudes.
