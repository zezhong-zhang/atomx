# Removed long test campaigns — 10 October 2026

These three original campaign modules were removed from active `tests/` at the
user's request for an essential suite targeting about three minutes. Their
source bytes are unchanged; [manifest.json](manifest.json) records original
paths, SHA-256 hashes and the last commit containing them at those paths.

| Archived source | Why removed | Retained routine coverage |
|---|---|---|
| [Pr/Eu/Nd transfer](test_slater_scaling_transfer.py) | Multi-ion calibration and transfer campaign; historical timings exceed the routine budget | Complete small 4f² Slater/sign/J-sector gates |
| [Large-grid DHF banks](test_dhf_semiseparable_exchange.py) | Two full ns=256/320 bank comparisons | Radial kernel identities and focused subsystem checks |
| [Fe XVII LMM](test_fe_lmm_mscheme_panel.py) | Full six-configuration manifold; original module reports 4–6 minutes | Signed FAC/GRASP operator matrices and small live multi-electron CI |

These replacements protect smaller contracts; they do not establish the removed
campaigns' large-space convergence or experimental agreement. Historical failed
and successful evidence remains in its original reports and Git history.

For an exact historical replay, create an isolated checkout at
`7a488454214dac3795d8b0f02b0163c8f162db78` and explicitly run the module at its
original `tests/` path with the locked environment and serial math threads.
Do not execute these archived copies in place: their original relative fixture
paths assume the original layout. Revalidating a changed implementation needs
an explicit port and new source/input identity, rather than treating an old
result as current evidence. No scheduled workflow runs these campaigns.

The maintained suite and explicit diagnostic commands are in
[Testing](../../../TESTING.md).
