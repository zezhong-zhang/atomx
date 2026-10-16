# Closed numerical failure 002

Whole20 headers and18 decoded geometry/radial fields passed. All7occupied V^N fields were measured:19 old-energy S-dual residuals reach4.76024559×10⁻⁸Ha (2p3/2); historical flags remain unchanged and current stationarity is unqualified.

κ4 returned the full119-root electron branch. FullS-Gram5.6621×10⁻¹⁵ and metric-projector idempotence1.6653×10⁻¹⁵ passed. The complete<50Ha band S-dual residual **1.8192200×10⁻⁷Ha** failed the unchanged1×10⁻⁸ gate; κ−5 was not reached. No gate was relaxed. All eight fullFock h/S fields,19original waves/nuclear grid and the complete κ4 spectrum were saved in `actual/failed_arrays.npz` (SHA8e32133bd4c917d51f12b9f286d53acc1cbcdb30afbd046ec43fb268c0adb37e).

Counts:8Fock/rawhS,152direct,230exchange,1Dirac/LAPACK,8metric solves,0SCF/CI/optimizer. Cache array payload385,123,824bytes; uncontrolled elapsed1.78s/RSS646.78MiB are execution observations, not timing claims. `failure.json` SHA629b14b3f718499e7c634118ae04d6dbde90f4c515d96792b647df6220850501 retains the original failed check and all occupied diagnostics.

Explicit003 successor keeps the same bank/H/S/complete50Ha band and1×10⁻⁸ gate. It calls existing native bordered refinement within the original complete electron span, tracks every selected column/gap/overlap and validates both fixedg partners before aggregate residual reporting. This changes numerical preparation, not the retained physical model. If the same gate still fails, retain it and stop; no further strategy is dispatched.
