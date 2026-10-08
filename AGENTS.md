# AGENTS.md — Working principles for atomx

**Measurement outranks speculation. Reference source outranks secondhand
accounts. Check conventions before diagnosing a bug.**

Seek physical self-consistency, reuse shared machinery, and earn performance
claims through measurement. Advance with the smallest experiment that can
distinguish competing explanations. Make evidence replayable, conclusions
revisable, and experience useful to the next agent.

This file holds lasting principles and cross-task obligations. Scientific
contracts, commands, pitfalls, measurements, and detailed records belong in
`docs/`. Use the compact [project index](PROJECT_INDEX.md) to locate code and tests;
query only the relevant symbols instead of loading its full JSON. Consult the
[reference index](docs/AGENT_REFERENCE.md) and relevant
[insights](docs/AGENT_INSIGHTS.md); read by task, rather than loading all archives.
Historical results apply to their recorded source, inputs, and models.

## 1. Work within the task

- Start with `git status --short`. Preserve other sessions' changes; do not
  overwrite, revert, stage, or commit work outside the authorized scope.
- Use `rg` to locate implementations, tests, and references. Read the relevant
  code and [architecture](docs/ARCHITECTURE_DESIGN.md) before changing a design.
  Distinguish implemented interfaces from proposals and historical capabilities.
- Continue within existing authorization until the result is reviewable and
  verified. Ask for clarification when a material contract cannot be inferred.
- Before a diagnostic experiment, state the hypothesis, fixed conditions,
  distinguishing outcomes, acceptance criterion, and stopping condition.
  Stop low-information scans; large changes first warrant checking model and space definitions.
- Estimate state-space size, memory, and work before allocating large objects.
  Check resource caps and machine load before long runs; record cap changes.
  Never silently shrink the physical model to fit a budget.

## 2. Preserve physical meaning

- Keep expansion space, orbital optimization target, observable state,
  dynamical preparation, and energy reference separate. Similar representations
  do not make these interchangeable; changing one must not silently change another.
- Carry state, basis, Hamiltonian, metric, units, and normalization identity
  through calculations, operators, results, caches, and persistence.
  A column index or configuration label alone is not a transferable state identity.
- Respect complete symmetry and degenerate subspaces. Preserve signed amplitudes,
  off-diagonal terms, and interference; a plausible diagonal result is insufficient.
- Reuse public physical operators, angular algebra, radial kernels, and density
  contractions. Methods select physics; numerical engines implement algorithms.
  Keep dependency direction explicit and abstractions justified by actual consumers.
- Declare approximations and operator coverage at the consuming interface.
  Support each result type explicitly or reject it; do not use silent attribute
  fallbacks to omit physics, change backends, or substitute an average model.
- A failed scientific gate calls for fixing the solution or representation.
  Do not adjust weights, state identity, tolerances, or error budgets to force a pass,
  or add corrections already included in the retained Hamiltonian.

## 3. Validate explanations and results

- Before comparing numbers, match Hamiltonian, nucleus, state, energy zero,
  units, normalization, averaging, and operator coverage. Read fixture generators
  and reference source; rerun the reference locally when practical.
- Anchor new formulas analytically or independently before production use.
  Test relevant off-diagonal phases and interference, not only diagonal or closed-shell cases.
- Validate derivatives and residuals in the actual constrained space and native
  metric, including state response. Energy stability, successful line search,
  or retained overlap does not replace a stationarity check.
- Compare lower and upper states in the same space with the same operators.
  Report individual energy changes, transition shifts, and splitting separately;
  do not splice old and new states or assume space increments are additive errors.
- Distinguish numerical residuals, conditional model bounds, radial/correlation/
  continuum convergence, physical resonances, and experimental validation.
  Independent references, internal snapshots, and repeated runs provide different evidence.
- Predeclare matching and acceptance rules. Preserve failed and negative results;
  unresolved causes remain unresolved. Use measured strict expected failures for
  unfinished scientific targets and label known gaps explicitly.

## 4. Measure performance without changing the problem

- Profile the actual end-to-end workload and identify its dominant stage.
  Check existing optimizations and recorded failed approaches before repeating them.
- Prefer reusable angular structure, batched kernels, immutable radial snapshots,
  and proven symmetry or sparsity. Remove only exact zeros and strictly unreachable
  coordinates; small amplitudes are not a license to truncate.
- Bind caches to complete physical and representation inputs. Invalidate on
  dependency changes; a new trial requires its own snapshot identity.
- Establish correctness separately from timing with the same Hamiltonian,
  states, spaces, and residual gates. Use alternating paired measurements,
  distinguish cold/warm caches, and record load and timing scope.
- A kernel or stage speedup is not an end-to-end speedup. Failed load guards
  invalidate controlled timing claims, while correctness checks should still run.
- Prefer existing process pools for independent tasks and pin worker BLAS threads
  to one. Measure threading, banded solvers, and matrix-free approaches before adopting them.

## 5. Verify and deliver

- Use the existing `.venv` and select checks from [TESTING.md](docs/TESTING.md).
  Run affected tests and exercise changed public entry points; expand regression
  scope when dependencies, failures, or unresolved concerns justify it.
- Physics changes require the relevant live gates and refreshed measurements.
  Documentation changes require factual, link, and diff checks. Collection alone
  is not a passing numerical suite; report only validation actually performed.
- Check generated compiler caches before treating a cache failure as a physics
  defect. Any cleanup is limited to generated artifacts, never scientific evidence.
- Keep replay-sensitive inputs, seeds, source/input hashes, states, bases,
  operators, metrics, and parameters. Record actual dirty source, not just HEAD;
  source changes require an explicit identity chain and matched-Hamiltonian validation.
- Do not merge degraded or unverified work. Revert only your own failed attempts
  and retain their evidence. Deliver what changed, why, validation, measured results,
  limitations, and remaining work; self-consistency does not establish experimental accuracy.

## 6. Maintain documentation and learn

- Keep the commit-time project-index hook installed. Rebuild navigation from the
  staged snapshot before every commit; never stage unrelated work to refresh it.
  Index generation does not replace checking documentation against code.
- Follow the [knowledge workflow](docs/AGENT_KNOWLEDGE_WORKFLOW.md). Every logical
  code change includes a [documentation impact review](docs/README.md#code-to-documentation-entry-points)
  and disposition in the same increment: update, merge, remove, or retain with a reason.
- Renames, removals, defaults, and contract changes require searching old names
  and behavior descriptions. Update examples, docstrings, capabilities, migration
  notes, and incoming links; a new report does not repair an outdated guide.
- Keep one current authoritative account per topic. Remove obsolete guidance
  without independent value after checking incoming links and merging useful content.
  Preserve unique archived evidence and failures; publish corrections with links to the originals.
- After a substantive increment, record EARS (expectation/contract, action/experiment,
  results, significance/limits) in ignored `trace.md`, with durable conclusions and
  evidence in the owning `docs/` topic. Never leave lasting knowledge only in chat or `/tmp`.
- At a phase boundary or handoff, distill: trigger → required/forbidden action →
  reason → scope. Register evidence and status, or explicitly record “no new insight.”
  Separate observation, inference, and user decision; preferences are not measurements.
- Promote only supported, actionable, lasting insights useful across tasks.
  Merge existing rules first. Keep formulas, tables, run logs, local paths, and
  task queues in `docs/`; do not turn this file into a second source of detailed truth.
- When evidence overturns a conclusion, revise current guidance, insight status,
  navigation, and affected rules together. Mark superseded/withdrawn claims and
  their replacements; do not rewrite archived evidence or silently erase failures.
- Before delivery, check knowledge links, status, and rule consistency. Keep this
  file in English and within 150 lines and 12 KiB UTF-8; move details into `docs/`
  before adding rules or bypassing the budget with long lines.
