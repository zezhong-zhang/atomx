# Migration Notes

## QED backend default

`Atom(..., qed=True)` now defaults to `qed_backend="auto"`.

`auto` keeps the historic Mohr/MSS self-energy backend for `Z <= 36` and
uses the FAC/Indelicato-Mohr K/L-shell backend for `Z >= 37`.  This closes
the documented high-Z residual against FAC.

Users who need bit-stability against the old MSS-only output should pass:

```python
Atom(Z=82, config="1s1", qed=True, qed_backend="mss")
```
