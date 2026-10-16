# Closed saved-gradient decomposition

The sole offline consumer passed all149 gates on the literal global80 and global208 points. It reconstructed the complete saved raw/tangent gradient and original `T M T` preconditioned gradient, preserved all29 columns/seven complete kappa groups, and retained all cross terms. [completion.json](completion.json) closes its inputs and scope with six stdlib checks; no new Hamiltonian, CI, density contraction, gradient producer, Dirac, eigensolver or optimizer was invoked.

The residual fell0.002641370232→0.000955326596, and its squared norm fell 86.919%. This is a decrease in the saved coordinate residual, not a strict physical energy or spectrum error bound. The original1e−7 residual and1e−8Ha indicator gates still both fail.

| Coordinate group | global80 residual² share/% | global208 residual² share/% |
|---|---:|---:|
| core19 | 29.790 | 20.952 |
| bare_valence5 | 6.646 | 25.668 |
| new_virtual5 | 63.564 | 53.380 |

| Complete kappa group | global80 share/% | global208 share/% |
|---|---:|---:|
| -4 | 0.819 | 0.768 |
| -3 | 20.198 | 20.755 |
| -2 | 64.320 | 58.288 |
| -1 | 5.797 | 9.898 |
| 1 | 7.795 | 7.057 |
| 2 | 0.370 | 2.485 |
| 3 | 0.701 | 0.750 |

At208 the largest columns are7p3/2(26.10%),6p3/2(18.05%),6d5/2(17.28%) and5p3/2(10.62%). The complete kappa−2 block carries58.29%. The deep1s/2s/2p1/2 columns each contribute less than0.000054% at208. The measured result does not support core-dominated residual based on large absolute core energy.

The net residual² reduction is mostly7p3/2(44.73%),5p3/2(19.31%) and6d5/2(10.45%). These are coordinate partitions of the observed change, not a causal decomposition of EOL or gap improvement. Some columns, including6s, grew while the whole norm shrank.

The raw-gradient partitions are strongly coupled. At208 the separate `T M T` driving-force norms are1.338/0.632/1.204, whereas their sum is only0.000955. All signed3×3 cross terms are essential. The actual `g·pg_group` values in core/bare-valence/new order are [-0.0004970470472068222, 4.2584840820435003e-05, 0.0004573810794471561]; the negative core entry means its isolated negative-gradient group direction would be uphill at this point. Masked raw-gradient driving forces therefore cannot be treated as independent small contributions or used to freeze/boost a group by its isolated norm.

The public optimizer starts every resumed call with empty secant history and keeps at most10 transported positive-curvature pairs by default. It fixes the saved screened spectral inverse, begins every search at unit step, and halves it on rejection. Run005 accepted72 unit,54 half and2 quarter steps, with0 fallback searches;58 extra CI trials were ordinary backtracks. Saved records do not expose secant-drop counts or pre-search non-descent resets. These observations do not identify line search or history reset as the cause of remaining nonstationarity.

The minimum next source-only proposal is matched `history_size=10` versus40 at this same literal208 point, with32 accepted steps per branch at most, equal empty histories initially, unchanged objective/P/inverse/stopping gates and the same finite-H checkpoint guard. Only the search memory parameter changes. The first11 accepted points can be identical; up to21 later directions may use pairs beyond10, while40 still cannot be fully populated by32 steps. No difference is an inconclusive capacity result in this window, not evidence against larger memory globally. No numerical comparison is authorized by this closure itself.
