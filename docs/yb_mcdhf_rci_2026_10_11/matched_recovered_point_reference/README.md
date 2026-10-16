# Matched recovered-point old-source reference

The rejected baseline compared gradients at two different floating-point radial points. A fresh, resource-bounded process imported only the frozen original005353 source epoch and evaluated the same complete bare-five equal-weight DC objective exactly twice: first on original005 saved physical spinors, then on the authoritative direct original P×archived recovered x spinors. No optimizer was imported or called, and no correlation space, weight, operator, source or tolerance was changed.

[contract.json](contract.json) and [prospective resource admission](prospective_resource_admission.json) were written before either gradient. [run.py](run.py) applied CPU affinity1, all numerical threads1, a3.95GiB address-space cap and a60s hard alarm. The completed process used6.2075s and peaked at797.06MiB RSS. These are resource observations, not a controlled performance comparison. The69-electron/24-orbital/ns123/Fermi-A175 model has exactly five CSFs in five distinct J/parity blocks and equal weights0.2; it is the same bare objective as the stopped public driver.

## Two actual points

Both gradients are projected with the same archived coordinate x, original layout/no-pair P and original scaled inverse blocks. The physical saved waves are not falsely declared bitwise equal to unpack(x).

| Old-source evaluation point | EOL energy / Ha | max g−archived g | max pg−archived pg | Original metric residual |
|---|---:|---:|---:|---:|
| Original005 saved physical waves | −14067.391460357385 | 0 | 0 | 5.14300923282462e−8 |
| Exact float64 originalP×recoveredx | −14067.391460357383 | 2.6310068574076167e−8 | 9.185507287403148e−15 | 5.143010780099849e−8 |

The two sets of coefficients differ by at most2.986638714119749e−13. Original saved radial gradients also reproduce the native005 gradient archive with maximum error0. The EOL energies, original residual/energy indicator, and both discrepancy maxima at the direct P×x point agree **bitwise** with all five scalar controls retained by the stopped new-source process. Both bare points pass the original1e−7 orbital/1e−8Ha indicator gates; this does not certify expanded531 MCDHF or experiment.

The old `recovered_coordinates.npz` stores a recovered coordinate x and a gradient evaluated on the original saved physical waves before recovery. Those quantities are related but do not represent a bitwise identical radial evaluation point. Its old physical-wave gradient is therefore an invalid strict whole-vector comparator for a new objective evaluation on direct P×x. This diagnoses the rejected comparison without relaxing its1e−9 raw-coordinate-g or1e−10 preconditioned-pg limits.

The stopped new-source process did not archive its complete g vector. Matching its retained scalar controls is not a direct elementwise cross-source proof. The later consumer must recompute the actual same-point bare gradient and compare every g/pg element against the new source-bound matched reference under the unchanged1e−9/1e−10 gates. Its complete H/CI/gradient and physical state checks remain necessary.

## Frozen reference payload

[arrays.npz](arrays.npz) is the matched directP×recoveredx reference. It exposes `x`, `g`, `preconditioned_gradient`, `P`, `Q`, `orbital_keys`, full `radial_gradient`, `ci_energies_Ha`, `ci_mixing`, equal `state_weights` and `EOL_energy_Ha`. It also includes each original `projector_i`, `inverse_blocks_i`, actual `metric_i` and nuclear `one_body_i`, plus `inverse_diagonal`. The coordinate vector is the exact archived2856-entry float64 x; P/Q are24×123 and reproduce the same direct matrix expression used by the stopped driver.

[physical_point_arrays.npz](physical_point_arrays.npz) independently retains the first actual saved-wave evaluation, with the same complete numerical payload and coordinate-tangent reference. [result.json](result.json) binds both payload hashes, original005 source-map SHA9dff33…, coordinate archive SHAfa3b24…, original complete metric SHAaae3a596…, exact original layout SHA, x raw-byte SHA and all loaded original-source module paths/hashes. No old native/probe/publicrun proof is modified.

[verify.py](verify.py) performed a separate read-only saved-array replay:625 checks passed in [verification.json](verification.json). It validates all scientific/source identities, exact P/x/inverse bytes, actual radial points and S-Gram matrices, equal weights and CI Gram/objective, reconstructs coordinate gradients from saved complete radial gradients, and independently assembles full block-diagonal inverse actions. It performs zero new gradients, CI solves or optimizer calls. The intentional old-physical-g versus recovered-point mismatch is an expected rejection control, not an accepted same-point gate.

[run.log](run.log) and [verify.log](verify.log) preserve execution output. [attempt001_pre_gradient_guard/failure.json](attempt001_pre_gradient_guard/failure.json) records an earlier zero-gradient input-identity rejection caused by nested tuple/list JSON representation; the exact attempted source and admission are retained there. The sole correction normalized the complete coupling representation through JSON before equality, matching the existing public driver's recursive container normalization. It changed no physics or gate, and only two full gradients were executed across all attempts.

## Documentation impact and handoff

This is a new matched-point reference for the next explicitly source/input-bound public continuation. Preserve the [original005 stationarity replay](../native_run/nuclear_eol005_projected_increment/stationarity_replay002/result.json) and the [stopped new-source baseline](../public_mixed_resume_run/new_source_baseline_controls.json) unchanged. The public optimizer API and [its frozen proof](../public_optimizer_resume/README.md) are unaffected. Root owns successor contracts, source gates and later publication; this subtask changes no shared source or existing guide. EARS is in ignored `trace.md`.

Distilled boundary: an archive containing x and g does not by itself prove that g was evaluated on exact unpack(x). Carry and verify the actual radial point through recovery and cross-source gradient comparisons, retain separate historical evidence, and correct the comparator before considering any tolerance change.
