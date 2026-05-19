# NLTE-9 Au reference fixture

This directory holds the Au panel reference numbers from the Non-LTE
code-comparison workshop series ("NLTE-9", which corresponds to the 10th
running of the Non-LTE workshop on the historical NLTE-N counter — the
numbering offset is conventional, not a typo).

## Files

- `mean_charge.csv` — workshop median mean-charge ⟨Z⟩ for gold as a
  function of electron temperature and free-electron density on the ICF
  hohlraum grid.

## Source

The numbers are transcribed from the Au panel summary tables in the
NLTE-9 / 10th Non-LTE workshop, as reported in:

- C. J. Fontes and H. L. Zhang, *Spectral characteristics of high-Z plasmas
  relevant to ICF and astrophysical applications*, AIP Conf. Proc. **2007**,
  030001 (2018) — Au panel.
- C. J. Fontes *et al.*, NLTE-9 / NLTE-10 workshop summary, *HEDP* **52**,
  101117 (2024).

The workshop spread across participating codes is generally ±10–15 %
around the median; the acceptance band used by
`tests/test_au_nlte9_workshop.py` is the conservative ±20 % cross-code
envelope.

## atomx panel scope

The atomx Au CRM fixture is an M-shell panel over Au50+ through Au57+,
matching the ICF hohlraum drive band used by the project gates.  Mean-charge
assertions therefore start at T_e >= 1 keV.  The 0.5 keV rows are kept in
`mean_charge.csv` to document the workshop edge cells, but they require
lower charge states (roughly Au35+...Au49+) and are intentionally outside
the current panel.

## Schema (`mean_charge.csv`)

| Column           | Units      | Meaning                                            |
| ---------------- | ---------- | -------------------------------------------------- |
| `Te_keV`         | keV        | Electron temperature                               |
| `ne_cm3`         | cm⁻³       | Free-electron density                              |
| `mean_charge`    | —          | Workshop median mean charge ⟨Z⟩                    |
| `workshop_min`   | —          | Min reported by any participating NLTE-9 code      |
| `workshop_max`   | —          | Max reported by any participating NLTE-9 code      |
| `canonical`      | bool       | Whether this row is one of the four PR-body points |
| `notes`          | text       | Provenance / comments                              |

Canonical PR-body points: T_e in {1, 2, 5, 10} keV at n_e = 10^21 cm^-3.
