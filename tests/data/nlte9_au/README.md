# NLTE-9 Au reference fixture

This directory holds a **stored, unverified** Au mean-charge panel that
`tests/test_au_nlte9_workshop.py` gates for internal consistency only.

## Files

- `mean_charge.csv` — stored mean-charge ⟨Z⟩ table on the ICF hohlraum
  grid, with a per-row `source` column and the `workshop_min`/`workshop_max`
  envelope.

## Source and verification status (read this before using the numbers)

Every row cites:

- C. J. Fontes and H. L. Zhang, *Spectral characteristics of high-Z plasmas
  relevant to ICF and astrophysical applications*, AIP Conf. Proc. **2007**,
  030001 (2018) — Au panel,
- C. J. Fontes *et al.*, NLTE-9 / NLTE-10 workshop summary, *HEDP* **52**,
  101117 (2024).

**The transcription cannot be checked inside this repository** (2026-09-17
validation-integrity pass, re-measured in the S6 audit):

1. no page/table number is recorded anywhere in the repo, so the ⟨Z⟩ values
   cannot be traced to a specific published table;
2. the envelope is constructed, not reported: `workshop_min` =
   `mean_charge − 2.5` and `workshop_max` = `mean_charge + 2.5` in **all 15
   rows** (an exact identity, measured), which no per-code spread of a
   workshop comparison would produce;
3. the earlier claim here that "the workshop spread across participating
   codes is generally ±10–15 %" is inconsistent with that envelope (±2.5
   charge units is ±4–6 % on this grid) and is therefore withdrawn.

Consequently the CSV must **not** be described as a published benchmark,
and no test may claim agreement with NLTE-9 from it.  What the tests do
assert is (a) the internal structure above and (b) that the benchmark's
`mean_charge()` model is a hand-tuned Gaussian whose output reproduces this
stored table — change detection on a calibration, marked
`@pytest.mark.snapshot` in `tests/test_au_nlte9_workshop.py`.

## atomx panel scope

The atomx Au CRM fixture is an M-shell panel over Au50+ through Au57+,
matching the ICF hohlraum drive band used by the project gates.  Any
mean-charge statement therefore starts at T_e >= 1 keV.  The 0.5 keV rows
are kept to document the workshop edge cells, but they require lower charge
states (roughly Au35+...Au49+) and are intentionally outside the panel.

## Schema (`mean_charge.csv`)

| Column           | Units      | Meaning                                            |
| ---------------- | ---------- | -------------------------------------------------- |
| `Te_keV`         | keV        | Electron temperature                               |
| `ne_cm3`         | cm⁻³       | Free-electron density                              |
| `mean_charge`    | —          | Stored mean charge ⟨Z⟩ (provenance unverified)     |
| `workshop_min`   | —          | Stored lower envelope = `mean_charge − 2.5`        |
| `workshop_max`   | —          | Stored upper envelope = `mean_charge + 2.5`        |
| `canonical`      | bool       | Whether this row is one of the four PR-body points |
| `source`         | text       | Per-row provenance (citation + verification status)|
| `notes`          | text       | Comments                                           |

Canonical PR-body points: T_e in {1, 2, 5, 10} keV at n_e = 10^21 cm^-3.
