# Retired campaign script tests

These 19 modules exercised campaign/import/viewer scripts removed in commit
`bb9dff3` (2026-09-29). They are preserved verbatim as historical contracts,
outside the current `tests/` collection. They are not passing verification
of the current package. Reintroducing a corresponding supported workflow
requires migrating its applicable contracts to that workflow and running them.

The retired scripts include the legacy NIST archive importers, viewer, DHF
campaign/repair commands and FAC branch-count inventory. Current reference
matching and comparison services have their own package tests; those tests
do not establish completeness of the removed archive/viewer workflows.
