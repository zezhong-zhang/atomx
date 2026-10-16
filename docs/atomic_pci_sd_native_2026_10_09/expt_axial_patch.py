"""Experimental exact axial/parity bridge over the owned EXP-T ABI001 patch.

Uses the upstream Dinfh product table and tensor blocks without changing CC
equations. The optional sidecar is only produced after complete signed integral
symmetry checks by the finite control. Not a public or production ABI yet.
"""

import difflib
from pathlib import Path


def patch_axial(source):
    path = Path(source) / "src/rcc/interfaces/pyscf_interface.c"
    old = text = path.read_text()
    begin = text.index("    /*\n     * symmetry group: double relativistic C1")
    end = text.index("    create_spinor_info(nspinors, irreps, eps, occ);", begin)
    text = (
        text[:begin]
        + r"""
    /* Optional exact signed axial M and inversion parity. Original C1 remains
     * the fallback when no experimental sidecar is declared. */
    int point_group_type = CC_GROUP_QUATERNION;
    int nsym = 2;
    int irrep_a1 = 1;
    char point_group_name[MAX_IRREP_NAME];
    strcpy(point_group_name, "C1");
    int *mult_table = NULL;
    char **rep_names = NULL;
    int *irreps = cc_calloc(nspinors, sizeof(int));
    /* EXP-T enters scratch before this interface. Resolve alongside the
     * absolute one-electron integral path, retaining its input identity. */
    char axial_path[CC_MAX_PATH_LENGTH];
    const char *last_slash = strrchr(opts->integral_file_1, '/');
    size_t parent_length = last_slash == NULL ? 0 : (size_t)(last_slash - opts->integral_file_1 + 1);
    if (parent_length + sizeof("ATOMX_AXIAL") > sizeof(axial_path)) {
        errquit("experimental Atomx axial sidecar path exceeds capacity");
    }
    memcpy(axial_path, opts->integral_file_1, parent_length);
    strcpy(axial_path + parent_length, "ATOMX_AXIAL");
    FILE *axial = fopen(axial_path, "r");
    if (axial == NULL) {
        mult_table = cc_calloc(nsym * nsym, sizeof(int));
        mult_table[0] = 1; mult_table[3] = 1;
        rep_names = cc_calloc(nsym, sizeof(char *));
        rep_names[0] = cc_strdup("A"); rep_names[1] = cc_strdup("a");
    } else {
        char magic[32];
        int count;
        if (fscanf(axial, "%31s %d", magic, &count) != 2 ||
            strcmp(magic, "ATOMX_AXIAL_001") != 0 || count != nspinors) {
            errquit("experimental Atomx axial sidecar count/version mismatch");
        }
        point_group_type = CC_GROUP_COMPLEX;
        strcpy(point_group_name, "Dinfh");
        int max_omega_x2 = CC_MAX_NUM_IRREPS / 4 - 2;
        rep_names = generate_irreps_Dinfh(max_omega_x2, &nsym);
        irrep_a1 = search_string("0g", rep_names, nsym);
        mult_table = construct_direct_product_table(nsym, rep_names, multiply_irreps_Dinfh);
        for (int i=0; i<nspinors; i++) {
            int m2, parity;
            char name[MAX_IRREP_NAME];
            if (fscanf(axial, "%d %d", &m2, &parity) != 2 ||
                m2 == 0 || abs(m2) % 2 != 1 ||
                6 * abs(m2) > max_omega_x2 || (parity != 1 && parity != -1)) {
                errquit("invalid axial spinor irrep or insufficient rank-six irrep range");
            }
            sprintf(name, "%d/2%s%s", abs(m2), parity == 1 ? "g" : "u", m2 > 0 ? "+" : "-");
            irreps[i] = search_string(name, rep_names, nsym);
            if (irreps[i] < 0) errquit("missing exact axial spinor representation");
        }
        char excess[2];
        if (fscanf(axial, "%1s", excess) != EOF) errquit("extra axial spinor data");
        fclose(axial);
        printf("\nATOMX_AXIAL_V1 exact M/parity tensor blocking; all spinors retained\n");
    }
    setup_symmetry(point_group_type, point_group_name, nsym, rep_names, irrep_a1, mult_table);
"""
        + text[end:]
    )
    path.write_text(text)
    return "".join(
        difflib.unified_diff(
            old.splitlines(True),
            text.splitlines(True),
            fromfile="a/" + str(path.relative_to(source)),
            tofile="b/" + str(path.relative_to(source)),
        )
    )
