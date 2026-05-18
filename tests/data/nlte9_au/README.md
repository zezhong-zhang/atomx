# NLTE-9 Au Mean-Charge Fixture

`mean_charge.csv` is the reduced Au hohlraum mean-charge panel used by
`tests/test_au_nlte9_workshop.py`.

The fixture tracks the NLTE workshop `zbar` summary quantity for the Au
ICF-hohlraum density range.  The gate intentionally uses the workshop-scale
cross-code envelope, not a single-code exact match: the test accepts atomx when
its UTA-mode CRM mean charge is within 20 percent of the reduced workshop mean.
