# Physics priority in the actual card headline

The previous annotation only selected the current physical-completeness priority when no finite audit was queued. A source update could queue bounded consumer replays while leaving the old expansion headline visible.

The annotation now preserves the production physics priority in both cases and explicitly says that queued source-change finite replays run first. It changes no job, physical gate, executor, cap or dispatch.

[Contract](contract.json) and [minimal diff](minimal.patch) bind the change. The two cases additionally reject tampered five-state inputs and preserve the job payload exactly. [Affected regression log](pytest.log): 35 passed, zero skipped. [Actual render](render.log) and the [real v2 delivery](../delivery_v2/delivery.json) exercise the current priority.
