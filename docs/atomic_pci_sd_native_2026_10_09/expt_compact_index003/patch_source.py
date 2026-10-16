"""Private exact EXP-T block-index storage experiment, not new CC equations.

Preserve every upstream symmetry test, block and creation order. Replace the
dense Cartesian inverse index with sorted (full key, original block number)
pairs; sparse lookup never assumes that block order is sorted. Binary diagram
files use an explicit private version and rebuild the index after block reads.
The Cartesian constructor traversal is unchanged and remains a scaling gap.
"""

import difflib


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"exact upstream source boundary differs: {old[:80]!r}")
    return source.replace(old, new)


def patch_diagram(source, *, reachable_constructor=False):
    original = source
    source = once(
        source,
        "#include <assert.h>",
        "#include <assert.h>\n#include <stdint.h>\n#include <limits.h>",
    )
    source = once(
        source,
        "    int max_sbs;            // max number of sym blocks = (# spinor blocks)^rank",
        "    size_t block_capacity = 64; // storage for actual created blocks only",
    )
    source = once(
        source,
        "    max_sbs = (int) pow(n_spinor_blocks, rank);\n    block_list = (block_t **) cc_malloc(sizeof(block_t *) * max_sbs);",
        "    block_list = (block_t **) cc_malloc(sizeof(block_t *) * block_capacity);",
    )
    source = once(
        source,
        "        block_list[blocks_counter] = block;",
        """        if ((size_t) blocks_counter == block_capacity) {
            if (block_capacity > (size_t) INT_MAX / 2 ||
                block_capacity > SIZE_MAX / (2 * sizeof(block_t *))) {
                errquit("Atomx compact block pointer extent exceeds exact backend range");
            }
            size_t new_capacity = block_capacity * 2;
            block_t **grown = cc_malloc(sizeof(block_t *) * new_capacity);
            memcpy(grown, block_list, sizeof(block_t *) * block_capacity);
            cc_free(block_list);
            block_list = grown;
            block_capacity = new_capacity;
        }
        block_list[blocks_counter] = block;""",
    )
    start = source.index(
        "void diagram_init_inverse_index(diagram_t *dg, size_t n_blocks, block_t **block_list)\n{"
    )
    end = source.index("\n\n\n/**", start)
    source = (
        source[:start]
        + """/* ATOMX_COMPACT_INDEX_V1: the existing size_t pointer holds pairs.
 * No public struct ABI change; every consuming function is replaced below.
 */
static size_t atomx_block_key(int rank, const int *indices)
{
    if (rank < 1 || rank > CC_DIAGRAM_MAX_RANK || n_spinor_blocks == 0) {
        errquit("Atomx compact index has invalid rank or block base");
    }
    size_t key = 0;
    for (int i = 0; i < rank; i++) {
        if (indices[i] < 0 || (size_t) indices[i] >= n_spinor_blocks ||
            key > (SIZE_MAX - (size_t) indices[i]) / n_spinor_blocks) {
            errquit("Atomx compact index coordinate or key extent invalid");
        }
        key = key * n_spinor_blocks + (size_t) indices[i];
    }
    return key;
}

static int atomx_compare_index_pair(const void *a, const void *b)
{
    size_t ka = ((const size_t *) a)[0], kb = ((const size_t *) b)[0];
    return (ka > kb) - (ka < kb);
}

void diagram_init_inverse_index(diagram_t *dg, size_t n_blocks, block_t **block_list)
{
    if (n_blocks > SIZE_MAX / (2 * sizeof(size_t))) {
        errquit("Atomx complete compact index exceeds exact byte extent");
    }
    dg->inv_index = NULL;
    if (n_blocks == 0) return;
    dg->inv_index = cc_malloc(2 * sizeof(size_t) * n_blocks);
    for (size_t i = 0; i < n_blocks; i++) {
        dg->inv_index[2 * i] = atomx_block_key(dg->rank, block_list[i]->spinor_blocks);
        dg->inv_index[2 * i + 1] = i;
    }
    qsort(dg->inv_index, n_blocks, 2 * sizeof(size_t), atomx_compare_index_pair);
    for (size_t i = 1; i < n_blocks; i++) {
        if (dg->inv_index[2 * i] == dg->inv_index[2 * (i - 1)]) {
            errquit("Atomx complete compact index contains duplicate block identity");
        }
    }
}"""
        + source[end:]
    )
    start = source.index(
        "block_t *diagram_get_block(diagram_t *dg, int *spinor_blocks_nums)\n{"
    )
    end = source.index("\n\n\n/**", start)
    source = (
        source[:start]
        + """block_t *diagram_get_block(diagram_t *dg, int *spinor_blocks_nums)
{
    if (dg->n_blocks == 0) return NULL;
    size_t key = atomx_block_key(dg->rank, spinor_blocks_nums);
    size_t low = 0, high = dg->n_blocks;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        size_t found = dg->inv_index[2 * mid];
        if (found < key) low = mid + 1;
        else high = mid;
    }
    if (low == dg->n_blocks || dg->inv_index[2 * low] != key) return NULL;
    size_t block_index = dg->inv_index[2 * low + 1];
    if (block_index >= dg->n_blocks) errquit("Atomx compact block index invalid");
    block_t *ret = dg->blocks[block_index];
    for (int i = 0; i < dg->rank; i++) {
        if (ret->spinor_blocks[i] != spinor_blocks_nums[i]) {
            errquit("Atomx compact block identity differs");
        }
    }
    return ret;
}"""
        + source[end:]
    )
    source = once(
        source,
        "    io_write(f, dg->inv_index, sizeof(size_t) * int_pow(n_spinor_blocks, dg->rank));",
        """    /* Explicit private diagram format: old dense files are never guessed. */
    const char atomx_magic[8] = "ATXDG002";
    io_write(f, atomx_magic, sizeof(atomx_magic));""",
    )
    source = once(
        source,
        "    dg->inv_index = (size_t *) cc_malloc(sizeof(size_t) * int_pow(n_spinor_blocks, dg->rank));\n    io_read(f, dg->inv_index, sizeof(size_t) * int_pow(n_spinor_blocks, dg->rank));",
        """    char atomx_magic[8];
    io_read(f, atomx_magic, sizeof(atomx_magic));
    if (memcmp(atomx_magic, "ATXDG002", sizeof(atomx_magic)) != 0) {
        errquit("Atomx compact backend requires its explicit ATXDG002 diagram format");
    }""",
    )
    source = once(
        source,
        "        dg->blocks[i] = block_read_binary(f);\n    }\n\n    io_close(f);",
        "        dg->blocks[i] = block_read_binary(f);\n    }\n    diagram_init_inverse_index(dg, dg->n_blocks, dg->blocks);\n\n    io_close(f);",
    )
    # All original inverse-index consumers must be accounted for, with no
    # residual dense indexing or serialization path hidden in the variant.
    if (
        "int_pow(n_spinor_blocks, dg->rank)" in source
        or "pow(n_spinor_blocks, rank)" in source
    ):
        raise ValueError("unconverted dense Cartesian block-index consumer")
    if reachable_constructor:
        # The pinned original is_symblock_zero is the OR of independent
        # per-axis empty spinor lists. Query that exact predicate at rank one;
        # retain its original full-rank check and all DPD/permutation tests.
        source = once(
            source,
            "    // dynamically-nested loop over tensor's dimensions\n    for (i = 0; i < rank; i++) {\n        ijkl[i] = 0;\n    }\n    while (ijkl[0] < n_spinor_blocks) {",
            """    int *axis_blocks[CC_DIAGRAM_MAX_RANK];
    size_t axis_sizes[CC_DIAGRAM_MAX_RANK] = {0};
    size_t axis_positions[CC_DIAGRAM_MAX_RANK] = {0};
    int empty_product = 0;
    size_t visited = 0;
    if (n_spinor_blocks > INT_MAX || n_spinor_blocks > SIZE_MAX / sizeof(int)) {
        errquit("Atomx reachable block-axis extent exceeds exact backend range");
    }
    for (i = 0; i < rank; i++) {
        axis_blocks[i] = cc_malloc(sizeof(int) * n_spinor_blocks);
        for (size_t b = 0; b < n_spinor_blocks; b++) {
            int index = (int) b;
            if (!is_symblock_zero(1, &index, &dg->qparts[i],
                                  &dg->valence[i], &dg->t3space[i])) {
                axis_blocks[i][axis_sizes[i]++] = index;
            }
        }
        if (axis_sizes[i] == 0) empty_product = 1;
        else ijkl[i] = axis_blocks[i][0];
    }
    while (!empty_product && axis_positions[0] < axis_sizes[0]) {
        if (visited == SIZE_MAX) errquit("Atomx complete candidate count overflow");
        visited++;""",
        )
        source = once(
            source,
            """        ijkl[rank - 1]++;
        for (i = rank - 1; i > 0; i--) {
            if (ijkl[i] >= n_spinor_blocks) {
                ijkl[i] = 0;
                ijkl[i - 1]++;
            }
            else {
                break;
            }
        }""",
            """        axis_positions[rank - 1]++;
        for (i = rank - 1; i > 0; i--) {
            if (axis_positions[i] >= axis_sizes[i]) {
                axis_positions[i] = 0;
                axis_positions[i - 1]++;
            }
            else break;
        }
        if (axis_positions[0] < axis_sizes[0]) {
            for (i = 0; i < rank; i++) {
                ijkl[i] = axis_blocks[i][axis_positions[i]];
            }
        }""",
        )
        source = once(
            source,
            "    diagram_bind_blocks(dg, blocks_counter, block_list);\n\n    // cleanup",
            """    diagram_bind_blocks(dg, blocks_counter, block_list);
    printf("ATOMX_REACHABLE_BLOCKS rank=%d base=%zu visited=%zu created=%d\\n",
           rank, n_spinor_blocks, visited, blocks_counter);
    for (i = 0; i < rank; i++) cc_free(axis_blocks[i]);

    // cleanup""",
        )
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            source.splitlines(True),
            fromfile="a/src/rcc/engine/diagram.c",
            tofile="b/src/rcc/engine/diagram.c",
        )
    )
    return source, patch
