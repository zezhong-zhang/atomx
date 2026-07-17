# Style & Chapter Contract

This document is binding on every chapter of the series. It is the referee for
all disputes about content, tone, or format. When a draft conflicts with this
document, the draft loses; when this document is wrong, change it in a commit
that says why.

---

## 1. The bar

Every *section* (not chapter — section) must pass three questions:

1. **Falsifiability.** Could a reader, or their AI assistant, check this claim
   against a computation, a dataset, or a cited measurement? If not, rewrite
   it until they can, or cut it.
2. **Novelty.** Does this say something, or show something, that Bethe–Salpeter,
   Cowan, Grant, Johnson, Sobelman, Friedrich, or Drake's handbook does not?
   Coverage is not a reason to exist. Being *executable* counts as novelty only
   when the execution reveals something (a convergence rate, a failure mode, a
   size) that prose alone could not.
3. **Reality.** Is every number in it real — produced by a run, or quoted from
   a cited source? A plausible number is a lie with good manners.

## 2. Voice

- Precise, confident, plain. Short sentences carry the physics; long sentences
  carry the connections.
- First person plural ("we") for choices and derivations; never "one".
- **Banned phrases:** "it is easy to see", "it can be shown", "straightforward
  algebra", "obviously", "clearly", "note that" as filler, "small correction"
  or any unquantified size word. Effects have *sizes*: state the size, in the
  unit the reader is thinking in, at a stated Z and n.
- War stories are told plainly, in past tense, with dates, without drama. The
  facts are dramatic enough.
- Humor is allowed once per chapter, maximum, and must never be in a Define or
  Verify section.

## 3. Chapter anatomy — the five movements

Fixed order. Named section headings may vary; the movements may not.

0. **The opening problem** (≤ 1 page). A real number, spectrum, or failure that
   the chapter exists to explain. It must be concrete (an element, an energy, a
   date) and it must be *resolved or re-examined* in the chapter's final section.
1. **Define.** Operators, states, and conventions, in boxes (see §4). No
   derivations here. Every symbol used later appears here first.
2. **Derive.** Only load-bearing derivations, in full, with no skipped signs.
   A derivation is load-bearing if (a) its *result* is used downstream in the
   book, and (b) seeing the mechanism changes how the reader uses the result.
   Everything else is stated with a citation and, where possible, a symbolic
   companion script. Every derivation ends with a **Checkpoints** list: limiting
   cases, units, symmetries — each one testable.
3. **Calculate.** Worked atomx examples on real elements. Code shown, output
   shown, both real (see §5, §6).
4. **Verify.** A table against exact results, NIST ASD, FAC, or a published
   measurement, with quantified parity and provenance for every column.
5. **Ablate.** Break it on purpose: turn the effect off, degrade the basis,
   adopt the wrong convention, and *predict the damage before observing it*.
   Includes the chapter's ladder table where applicable. Exercises live here;
   each exercise states its verification criterion ("your value must match X
   to within Y").

Then: **Sources** — short, annotated (one line each on *why* the source is
listed), never a bibliography dump.

**Bridge boxes** (Vol. I only): ≤ half a page, marked `> **Bridge.**`, pointing
at where a chapter object reappears in Vols. II–IV. At most two per chapter.

## 4. Contract boxes

The series' central device. A *contract* is a convention choice that (a) other
codes or books make differently, and (b) silently corrupts results when
mismatched.

Format (MyST admonition):

```markdown
:::{admonition} Contract C2.3 — Energy normalization of continuum states
:class: important
⟨ε′ℓ|εℓ⟩ = δ(ε − ε′), which fixes the asymptotic amplitude to √(2/(πk(r))).
Codes that normalize per unit momentum differ by a factor dk/dε = 1/k.
**Enforced by:** `tests/test_continuum_vs_fac.py::test_amplitude_convention`
:::
```

Rules:
- Numbered `C<chapter>.<n>`, cited by that number everywhere (text, other
  chapters, appendix A).
- Every contract names at least one *enforcing test* that exists in the repo.
  If the test does not exist yet, writing the chapter includes writing the test.
- State what the *other* convention would give, not just ours. A contract that
  doesn't name the rival convention doesn't immunize anyone.
- All contracts are mirrored into Appendix A (generated, not hand-copied).

## 5. Numbers policy

- **No hand-typed physics numbers in prose.** Every number is either
  (a) computed by a script in the chapter's `calc/` directory, or
  (b) quoted from a citable source (CODATA, NIST ASD, a paper, a FAC run).
- Every inline number carries a provenance marker in the source:
  `<!-- num: calc/convergence.py:E_1s -->` or `<!-- num: NIST ASD H I 2022 -->`.
  Markers are invisible in rendered output; the build tooling (later) will
  check them. Until then, reviewers check them by hand.
- Round honestly: quote the digits the computation supports, state the digit
  where agreement fails. "−0.5000067 Eₕ (analytic: −0.5000066566)" beats
  "≈ −0.5 Eₕ" every time.
- Parity language is standardized: "agrees to N digits", "rms x% over range R",
  "ratio r = a/b ∈ [lo, hi]". Never "excellent agreement".

## 6. Code blocks

- atomx **public API only** (`from atomx import ...`); if a chapter needs a
  private function, that is an atomx API bug — file it, fix it, then write the
  chapter against the fix.
- Every block is self-contained (runs from a bare interpreter in the repo) and
  ≤ 20 lines. Longer computations live in `calc/` scripts and the text shows
  the interesting excerpt.
- Every block that prints output shows the *actual* output, in a fenced block
  immediately after, produced by an actual run. Reviewers re-run every block.
- Style: match atomx's own conventions; no defensive try/except in book code;
  no comments that narrate ("# solve the atom").

## 7. Figures

- Every figure is generated by a script in `calc/`, saved to the chapter's
  `figures/` directory, and regenerable with one command. Hand-drawn diagrams
  (level schemes, apparatus sketches) are allowed but live as versioned SVG
  with sources.
- Every axis labeled with quantity *and unit*. Captions state what to look at
  and what it implies, in ≤ 3 sentences.
- Use matplotlib defaults plus the book's rcParams file (to be added at
  `docs/book/bookstyle.mplstyle`); no per-figure style improvisation.

## 8. Mathematics and notation

- **Hartree atomic units throughout** (Contract C1.1); SI or eV appear only in
  Verify tables and opening problems, always labeled.
- Notation is owned by Appendix B's symbol table; chapters may not repurpose a
  symbol. Core choices: `E` total energies, `ε` orbital/one-particle energies,
  `κ` relativistic angular quantum number, `Ω` collision strength, `σ` cross
  sections, `α` fine-structure constant only (never a generic constant), `Z`
  nuclear charge, `P(r), Q(r)` large/small radial components, `R^k` Slater
  integrals, `⟨‖·‖⟩` reduced matrix elements (Wigner–Eckart per Contract C6.x).
- Display equations are numbered per chapter `(2.14)`; only referenced
  equations get numbers.
- Derivation steps that invoke a contract cite it inline: "using C21.1".

## 9. Files

```
docs/book/vol1/chNN_slug/
  text.md          # the chapter (MyST markdown)
  calc/            # every script behind every number and figure
  figures/         # generated output (committed)
  notebook.ipynb   # added at build-tooling stage, not during drafting
```

`text.md` begins with YAML front matter: `title`, `part`, `chapter`, `status`
(draft/reviewed/final), `depends_on` (chapter list — this is the concept-DAG
metadata the AI reading paths consume), `contracts` (list of contract IDs
defined here).

## 10. Length and depth

- Target 6,000–12,000 words per chapter (≈ 25–45 book pages) excluding code
  and tables. Under 6k means the chapter is a section of another chapter; over
  12k means it is two chapters.
- Depth rule: basics are treated *fast but completely* (hydrogen radial
  equation: every step, four pages, no detours); frontier material is treated
  *honestly* (state what is not known, cite the 2024–2026 literature, and say
  what a calculation can and cannot currently deliver).

## 11. Forbidden moves

1. A number without provenance.
2. An API call that was not executed. (Hallucinated atomx usage is the
   book-killing failure mode. When in doubt, run it again.)
3. A convention stated without its rival and its enforcing test.
4. A derivation that skips a sign "for clarity".
5. A claim of agreement without a range and an rms/ratio.
6. Forward references that do the current chapter's work ("as we will see" as
   a substitute for an argument).
7. Editing a Verify table to match prose. The table is generated; the prose
   yields.
