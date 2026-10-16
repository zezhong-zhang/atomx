"""Reviewed open Fov*T2 antisymmetry fix for the pinned 0h1p equations.

The input T2 is already antisymmetric in its two outgoing particles. Applying
P(34) again doubles this one open core-valence Fock contraction. Retain the
complete Fock operator and change only the duplicate permutation, as tested
against the full signed amplitude/Bloch oracle.
"""

import difflib

OLD = """    // dgd2b_2
    reorder("t2c", "r1", "2341");
    mult("vh", "r1", "r2", 1);
    perm("r2", "(34)");
    update("s2_0", -1.0, "r2");
    restore_stack_pos(pos);
"""
NEW = """    // dgd2b_2: open core-valence Fock contraction.
    // ATOMX_FOV_T2_001: t2c already antisymmetric in outgoing particles.
    // A second P(34) would double this contribution in noncanonical orbitals.
    reorder("t2c", "r1", "2341");
    mult("vh", "r1", "r2", 1);
    update("s2_0", -1.0, "r2");
    restore_stack_pos(pos);
"""


def patch_noncanonical_attachment(source):
    """Reject changed conventions; never touch a validated executable or tree."""
    relative = "src/rcc/models/sector01.c"
    path = source / relative
    before = path.read_text()
    if before.count(OLD) != 1:
        raise ValueError("pinned noncanonical attachment diagram differs")
    after = before.replace(OLD, NEW)
    path.write_text(after)
    return "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/" + relative,
            tofile="b/" + relative,
        )
    )
