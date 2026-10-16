The browser launcher is prepared; no server or browser has run. The existing browser contract, site assets and local data proof remain unchanged. Root must provide a separate exact execution seal and release the exclusive resource window before the one foreground attempt.

Run `launch_browser.py --execution-contract <root seal> --execution-sha256 <seal SHA>` with Python. It validates the root seal, every bound preparation input and the local asset manifest, then pins itself and its descendants to CPU 30 with thread environment values of 1. An occupied port 39319 or an existing attempt/artifact refuses execution. No credentials, installation, publication, scientific imports or registry mutation are involved.

The whole foreground budget is 90 seconds, including a three-second cleanup reserve. Aggregate descendant and supervisor RSS is sampled every 0.2 seconds; an observed value above 2 GiB or sample gap above 0.25 seconds stops the attempt. This is a sampled stop rule, not a kernel memory cap; brief overshoot can occur and the RSS sum counts shared pages more than once. No RLIMIT_AS or systemd/cgroup modification is applied. Timing is uncontrolled browser QA.

The supervisor acts as a Linux child subreaper, retains PID start identities, and uses TERM followed by KILL on its own descendants in every controlled exit. Adopted detached Chromium children are included. Full server/browser stdout and stderr, RSS samples, artifact hashes, actual exits and cleanup proof are retained. An external uncatchable SIGKILL cannot execute Python cleanup, so the root must retain foreground supervision. Browser failure, timeout or unresolved cleanup stays failed; no automatic retry is allowed.

Fifteen static/source preparation checks and Ruff passed. They are preparation evidence, not browser acceptance. Root execution and resulting artifacts will be new evidence; the old preparation handoff is not rewritten.

The root review corrected the preparation consumer to read the actual `all_saved_data_and_preservation_checks_passed` field. The unexecuted prior launcher and proof are preserved in `browser_launcher_draft001/`; no browser attempt occurred.
