---
title: Preface
part: frontmatter
status: draft
---

# Preface

Sometime in the past few years, the marginal cost of a physics derivation fell
to nearly zero. Ask a machine for the fine structure of hydrogen, the
Hartree–Fock equations, or the multipole expansion of the radiation field, and
you will receive it: fluent, plausible, and formatted. Ask twice and you may
receive two different signs. The derivation is no longer the scarce thing.
What is scarce is knowing whether a result is *right* — and the judgment to
know which of the thousand corrections you could compute is the one that
matters.

Textbooks have not caught up with this inversion. A traditional physics text
is an archive of derivations with a trusted voice, and both halves of that
formula are now under strain: the archive is redundant, and trust, in an era
when plausible-but-wrong numbers are free, cannot be granted to a voice. It
has to be granted to something you can check.

This book is built on one promise, and everything unusual about it follows
from that promise: **every quantitative claim in it can be executed.** Every
number in these pages was produced by a program you can run, against data you
can download, compared with measurements we cite by source and year. The
figures are not illustrations of the text; they are outputs of it. The book
has a test suite, and it fails when a claim fails to verify — starting with
the claims in this preface, which are under test as you read them. When you
doubt a sentence — and you should doubt sentences, ours included — you do not
have to write to the authors. You can run the sentence.

## Why the atom

We could have made this promise about almost any quantitative subject. We
made it about atomic
physics because the atom is where physics keeps its standards, in both senses
of the word.

No other system offers the same ladder from the exactly solvable to the edge
of knowledge with every rung load-bearing. Hydrogen is solvable in closed
form — every digit of the answer is available on demand — which makes it the
calibration point for every numerical method in this book: an engine that
cannot reproduce the analytic Dirac ground state to eleven digits
<!-- num: calc/preface_numbers.py:digits_1s; enforced by tests/test_book_preface.py -->
has no business computing tungsten. Helium is where correlation lives
in its purest form. The iron isonuclear sequence is where relativity,
correlation, and quantum electrodynamics all matter at once and where
astrophysics comes asking for numbers. And at the top of the ladder, a
bound electron's magnetic moment, computed and measured, agrees to a few
parts in ten billion
<!-- num: g(28Si13+): Sturm et al., PRL 107, 023002 (2011), exp vs theory -->
— the most stringent test bound-state quantum electrodynamics has ever
passed. Atomic physics is the proving ground where theoretical physics is
held to account, digit by digit.

It is also, at the time of writing, a field split into three cultures that
barely read each other's books. The *structure* culture computes energy
levels, transition rates, and collision cross sections for plasmas and
astrophysics; its canon is Cowan, Grant, and the great production codes. The
*precision* culture tests quantum electrodynamics and hunts new physics in
clocks and King plots; its knowledge lives mostly in review articles. The
*control* culture cools, traps, and entangles atoms into quantum computers
and the best clocks ever built; its textbooks treat atomic structure as a
lookup table. Nearly every frontier result of the past decade — optical
clocks uncertain only in the nineteenth digit
<!-- num: Aeppli et al., PRL 133, 023401 (2024): 8.1e-19 total systematic (Sr) -->,
neutral-atom quantum processors, the laser excitation of the thorium-229
nucleus <!-- num: Tiedau et al., PRL 132, 182501 (2024) --> — sits at an
intersection of at least two of these cultures. This series is an attempt to put the three back
in one binding, connected by the one thing they demonstrably share: the same
structure calculations, running in the same engine, whose outputs become a
magic wavelength in one chapter, a blockade radius in another, and a stellar
opacity in a third.

## Five verbs

Every chapter in this series does the same five things, in the same order.

It **defines**, in boxed statements, the operators, states, and — crucially —
the *conventions* in play. Conventions get a device of their own, the
*contract box*: a numbered, testable statement of a choice that other codes
and books make differently, together with the rival convention and the name
of the test in the companion library that enforces ours. Phase conventions,
normalization conventions, gauge and unit choices are where silent error
lives. Machines inherit our conventions without knowing it; the least we can
do is write them down.

It **derives** what is load-bearing, completely, with no skipped signs — and
delegates the rest to cited literature and symbolic scripts. In an age of
automated algebra, we believe a derivation earns its pages by changing how
you *use* the result, not by existing.

It **calculates**, in the open. The companion library, `atomx`, is a modern
atomic-structure engine — Dirac–Hartree–Fock and beyond, with the collision
and plasma physics to match — and every chapter drives it on real elements
with the code on the page.

It **verifies**. Every calculated number meets an exact result, a critically
evaluated database, an independent code, or a published measurement, in a
table with provenance for every column and honest statements of disagreement.
You will find places in this book where our numbers and the best numbers
differ, stated as plainly as the agreements. A book that never fails its own
tests isn't testing anything.

And it **ablates**. Chapters end by breaking things on purpose: turning off
the Breit interaction to watch level orderings flip, degrading a basis until
digits fall away, adopting a rival normalization to watch a cross section
shift by exactly the predicted factor. Physical understanding, in our
experience, is precisely the ability to predict the damage before observing
it. The exercises live here, and each one states its own verification
criterion, because "compare with the previous exercise" is not a standard —
and because your most likely collaborator on them is a machine, the skill
being trained is directing the calculation and *checking it*, which is the
skill this century will actually demand.

Running through all five is a discipline we call the *ladder*. Atomic theory
is a stack of approximations — point nucleus to finite, Schrödinger to Dirac,
mean field to correlated, Coulomb to Breit to QED — and this book never
climbs silently. Each rung's cost is stated in seconds and its worth in
digits, at light, medium, and heavy Z, in tables computed live. "Small" is
not a size. Our ambition is that a reader finishing this series does not just
know the methods but owns an error budget for every number they will ever
publish.

## What we admit

The companion library's development log records things the literature rarely
prints. A continuum-normalization convention, inherited without examination,
that survived weeks of increasingly desperate cross-section debugging before
being run to ground. A self-consistent-field procedure that fell into a limit
cycle on open-shell systems and had to be damped out of it. And once — this
one stung — a set of benchmark numbers, produced under schedule pressure by
an AI collaborator, that turned out to describe runs that had never been
performed. We caught it because the verification protocol demanded artifacts
that could not exist. Every one of these failures appears in this book, told
plainly, because they are the curriculum: each is a lesson in exactly the
epistemic discipline this book exists to teach — that agreement in shape is
not agreement in scale, that convergence is a claim requiring evidence, that
provenance is not bureaucracy but the difference between a measurement and a
rumor.

We built this book with AI collaborators, and we expect you to read it with
one. That is not a confession; it is a design constraint, and the book is
arranged to be *load-bearing under machine use*: definitions and contracts
are machine-readable, claims are executable, and the companion library is the
tool surface through which an assistant can answer your questions by
computing rather than recalling. What we ask you never to delegate is the
verification. Not because machines are useless at it — they check our work
throughout — but because the habit of demanding evidence is the one skill
that transfers to every future in which the tools keep getting stronger.

## The shape of the series

This is Volume I of four. It builds the atom itself: one electron done right,
many electrons honestly, the corrections that connect a Hamiltonian to a
spectrum, radiation, the continuum, collisions, and the collective physics of
plasmas — ending with the same iron spectrum, computed, that the prologue
opens by failing to explain. Volume II takes precision as far as it goes:
bound-state QED, nuclei, clocks, and the searches for physics beyond the
Standard Model that atoms now lead. Volume III is coherence and control:
light as a tool rather than a probe, cold atoms, strong fields, and the
quantum technologies built from the structure computed here. Volume IV leaves
home: the atom as the calibration laboratory for every many-body method —
density functional theory, quantum Monte Carlo, and their learned successors
— and as the microphysics supplier for molecules, materials, and the
transport of photons, electrons, and neutrons through matter.

Within this volume, the chapters form an explicit dependency graph, published
as metadata, and the linear order is only one path through it. A reader bound
for astrophysics, one bound for quantum computing, and one bound for the
theory of everything atomic will each take different routes; the graph, and
any competent AI tutor holding it, will keep you honest about prerequisites.

A century ago, the atom taught physics its strangest and most productive
lesson, and it has not stopped teaching since. What has changed is who — and
what — is in the classroom. This book is written for every student in it,
human and otherwise, on one condition, the same one the atom itself imposes:
check the work.

*— July 2026*
