"""Source-bound removal of numerator magnitude screening in EXP-T diveps."""

import difflib
import hashlib

ORIGINAL_SHA256 = 'd8a57389490352b214e188a336735288a87776e12b78ed2277e23c467671384c'
QUALIFIED_SHA256 = 'dd17fb3cb207bbe7d6430761c0bc4698abbd49fc994d6bc74f00622dded3b547'
CAPABILITY = 'ATOMX_EXACT_NONZERO_DIVEPS_001'


def patch_exact_nonzero_division(source):
    """Divide every nonzero numerator; preserve denominators and equations.

    A small numerator can yield a resolvable amplitude near a small nonzero
    denominator. Exact complex zeros remain skipped to avoid 0/0 in closed
    model-space coordinates. Unknown or already patched sources reject.
    """
    path = source / 'src/rcc/engine/diveps.c'
    original = path.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('unknown pinned numerator-division source')
    declaration = 'static const double ZERO_THRESH = 1e-14;'
    condition = 'if (cabs(t) < ZERO_THRESH) {'
    entry = 'diagram_t *diagram_diveps(diagram_t *dg)\n{'
    if (original.count(declaration) != 1 or original.count(condition) != 4
            or original.count(entry) != 1):
        raise ValueError('unknown pinned numerator-division structure')
    updated = original.replace(declaration,
        '/* Every nonzero numerator must be divided, including near a small D. */')
    updated = updated.replace(condition, 'if (t == 0.0) {')
    updated = updated.replace(entry, entry + '''
    static int nonzero_marker_printed = 0;
    if (!nonzero_marker_printed) {
        printf("ATOMX_EXACT_NONZERO_DIVEPS_001 all nonzero numerators divided\\n");
        nonzero_marker_printed = 1;
    }''')
    if hashlib.sha256(updated.encode()).hexdigest() != QUALIFIED_SHA256:
        raise ValueError('numerator-division patch differs from qualified source')
    difference = ''.join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile='a/src/rcc/engine/diveps.c', tofile='b/src/rcc/engine/diveps.c'))
    path.write_text(updated)
    return difference
