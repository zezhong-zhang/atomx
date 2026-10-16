# Native Yb endpoint160 and matched static RCI publication

The [remote Yb page](https://atomx-atomic-spectrum-atlas.black-fairy-6027.chatgpt.site/#Yb:1:models)
was published as version 3 from the reviewed website commit
`fe4fb81bff25179b48cc1f91c2dddb236ee283a2`.
The [deployment](deployment.json) succeeded at 2026-10-11T10:03:09Z;
the [access reread](access_after_deploy.json) confirms the previous owner-private
audience and revision 1 remained unchanged.

Two native full531 models and eight rows were appended: accepted160 DC and its
same-point static Breit RCI. The page explicitly retains the original DC
stationarity failure and all four experimental failures. It has 12 separately
identified models and 48 rows, preserving the previous 10 models and 40 rows.
The latest accepted320 DC point is reported separately in the hourly evidence;
this publication does not claim a static RCI calculation at point320.

The [frozen handoff](handoff_manifest.json), [root preflight](root_publication_preflight.json)
and [push/archive receipt](root_push_and_archive.json) bind the exact source,
all 12 new/changed website files and the deployment archive. Local checks passed
6,611 data checks, 50 browser checks and 51 documentation checks. The original
6,019 NIST gzip files and previous scientific records were preserved.
Only the separate website repository was committed and pushed; no Atomx source
commit or access change was made by this publication.

The initially attempted credential reader received EOF before a credential was
provided. The retry disabled terminal echo and passed the ephemeral credential
through stdin; the actual push succeeded and persisted no token. This did not
change the reviewed source or archive.
