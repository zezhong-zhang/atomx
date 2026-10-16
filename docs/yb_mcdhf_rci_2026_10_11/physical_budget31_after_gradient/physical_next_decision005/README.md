# Five-state constrained derivative discriminator

This continues the [completed four-point response](../physical_next_decision004/execution002/RESULTS.md)
in Atomx's native MCDHF→RCI route. The immediate question is whether the
five individual state derivatives agree with fresh constrained CI energies,
and which orbital contributions survive upper-minus-lower cancellation.
The earlier aggregate central-difference failures remain archived; their
independently centered diagnostic passed and did not certify the ordinary
reported-energy callback.

Before any new energy evaluation, root fixes **h=0.002 and h/2=0.001** along
the complete **−checkpoint13.full_normalized_pg** direction. Retain literal
accepted13, f043 native source, all31 shared orbitals/3689 coordinates,
original seven nuclear positive-energy projectors and metric/inverse blocks,
Z70/A175 Fermi nucleus, complete A91NR/1595CSF and B100NR/2574CSF models,
five signed physical UIDs and equal0.2 objective weights. Every trial uses
the existing per-κ polar retraction. No NIST-based direction, step or weight
selection is allowed.

The experiment has two fresh retained-owner R0 CI calculations and eight
complete displaced CI calculations: ±h and ±h/2 in each model. At R0,
contract each full signed CI vector with itself, divide by its actual norm,
and differentiate using the shared native radial engine. Ten full state
gradients yield absolute-state, four-line and aggregate EOL directional
actions. Their weighted sum must close against the already saved aggregate
gradient in the original fixed metric. Keep all31 orbital contributions.
The stationary target remains the fixed weighted EOL functional. Individual
state gradients need not vanish in a shared orbital bank; their differences
measure state balance and line sensitivity, not independent convergence
failures or a license to change the objective weights.

For each h, compare ordinary reported central differences with the analytic
actions using the unchanged **max(1e−8,1e−5·|action|)** tolerance in Ha per
directional coordinate. Record per-state and line outcomes at both steps.
The predeclared Richardson estimate `(4 D(h/2)−D(h))/3` is a separate
diagnostic; it cannot replace or retrospectively pass an ordinary-FD gate.
Record all finite-response outcomes before drawing the derivative verdict.
Identity, complete CI residual/Gram, no-pair and resource failures stop
immediately. No automatic step scan, model reduction or gate adjustment.

Matching derivatives would establish this local direction for the measured
states, not full-space stationarity or experimental accuracy. A persistent
mismatch requires a producer/constraint/reporting diagnosis. Reference-metric
gradient norms and orbital decompositions are diagnostics, not spectral
uncertainty bounds. Execution/resource details and actual results belong to
the new execution directory; earlier sealed evidence remains unchanged.
