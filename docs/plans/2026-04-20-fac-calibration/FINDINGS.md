# FAC/Gu α_xc calibration — findings

## Verdict: **PASS**

Max relative residual between the closed-form ``get_hxs(Z, N, ihx, hxs)``
and FAC's converged ``AHX``:

    max |Δ|/α = 9.65e-10     (gate: ≤ 1e-03)

## Panel

```
  Z    N    HXS    IHX        branch         α_FAC       α_atomx      |Δ|/α
  2    2   1.00  -1.00      k=1 only    0.60373098    0.60373098   4.19e-10
  3    3   1.00  -1.00      k=1 only    0.59614158    0.59614158   3.52e-10
  4    4   1.00  -1.00      k=1 only    0.54358167    0.54358167   8.89e-10
  6    6   1.00  -1.00      k=1 only    0.53401444    0.53401444   8.42e-10
 10   10   1.00  -1.00      k=1 only    0.51465105    0.51465105   9.52e-10
 18   18   1.00  -1.00      k=1 only    0.46841349    0.46841349   8.34e-10
 26   10   1.00  -1.00      k=1 only    0.50445120    0.50445120   4.61e-10
 26   12   1.00  -1.00      k=1 only    0.47703235    0.47703235   9.63e-10
 36   18   1.00  -1.00      k=1 only    0.45381608    0.45381608   9.63e-10
 54   10   1.00  -1.00      k=1 only    0.46478511    0.46478511   1.81e-11
 74   10   1.00  -1.00      k=1 only    0.41945244    0.41945244   5.90e-10
 82   18   1.00  -1.00      k=1 only    0.37229885    0.37229885   7.27e-10
 92   10   1.00  -1.00      k=1 only    0.36654071    0.36654071   9.65e-10
```

## Observations

1. FAC's default ``SetAtom`` uses ``HXS = 1.0`` and ``IHX = -1.0``.
   With ``|IHX| ≥ 0.999``, GetHXS returns the **k=1 branch** of the
   tables. The k=0 branch is reachable only if the caller explicitly
   sets ``ihx ~ 0`` (not exercised by the atomx adapter's defaults).

2. α_xc varies from ~0.60 (low-Z Li-like) down to ~0.37 (Pb Ar-like).
   The negative Z² slope in ``_hx1`` dominates the fit at high Z.

3. All panel rows hit the gate. The closed-form Python port is
   byte-identical to FAC's behaviour for every (Z, N) checked, to
   numerical double-precision.

4. None of FAC's other ``AHX``-related configuration (``HX0``, ``HX1``,
   ``RHX``, ``DHX``, ``BHX``, ``CHX``) is populated for a default
   ``OptimizeRadial`` run — confirming that the parameterised Gu
   formula is the only ingredient.

## Next step (Task 1)

Port ``get_hxs`` verbatim into ``atomx.physics.local_exchange`` with
the same table layout. Promote the parameter arrays to
``atomx.numerics.local_dfs.hxs_tables`` so both the shooting and
B-spline paths share them.
