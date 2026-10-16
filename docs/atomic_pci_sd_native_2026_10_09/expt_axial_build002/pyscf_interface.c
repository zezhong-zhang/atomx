/*
 *  EXP-T -- A Relativistic Fock-Space Multireference Coupled Cluster Program
 *  Copyright (C) 2018-2025 The EXP-T developers.
 *
 *  This file is part of EXP-T.
 *
 *  EXP-T is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  EXP-T is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with EXP-T.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  E-mail:        exp-t-program@googlegroups.com
 *  Google Groups: https://groups.google.com/d/forum/exp-t-program
 */

#include "interfaces.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "io.h"
#include "engine.h"
#include "error.h"
#include "options.h"
#include "spinors.h"
#include "symmetry.h"
#include "timer.h"


void read_pyscf_integrals(char *path_integrals_1e, char *path_integrals_2e,
                          int *nspinors, int *nocc, double *enuc, double *escf, double **eps, int **occ,
                          double complex **eri, size_t **eri_index);

double complex get_eri(int nspinors, double complex *eris, size_t *eri_index, int p, int q, int r, int s);

void check_irrep_names();

double complex *pyscf_unique_eri = NULL;
size_t *pyscf_eri_index = NULL;
double complex *atomx_fock = NULL;


void pyscf_interface(cc_options_t *opts)
{
    timer_new_entry("pyscf", "Interface to pyscf");
    timer_start("pyscf");
    printf("\n ATOMX_INTEGRALS_V1 signed-antisymmetric complex Fock+ERI\n");
    printf(" file with molecular integrals: %s\n", opts->integral_file_1);

    int nspinors = 0;
    int nocc = 0;
    double enuc = 0.0;
    double escf = 0.0;
    double *eps = NULL;
    int *occ = NULL;

    read_pyscf_integrals(opts->integral_file_1, opts->integral_file_2,
                         &nspinors, &nocc, &enuc, &escf, &eps, &occ,
                         &pyscf_unique_eri, &pyscf_eri_index);

    printf(" number of spinors = %d\n", nspinors);
    printf(" number of occupied spinors = %d\n", nocc);
    printf(" nuclear repulsion energy = %.16f\n", enuc);
    printf(" scf energy = %.16f\n", escf);
    printf(" end interface to pyscf\n\n");

    opts->enuc = enuc;
    opts->escf = escf;

    printf(" retained requested tile size %d\n", opts->tile_size);

    printf(" switching openmp parallalization algorithm to 'internal'\n");
    opts->openmp_algorithm = CC_OPENMP_ALGORITHM_INTERNAL;

    /*
     * complex arithmetic
     */
    arith = CC_ARITH_COMPLEX;
    WORKING_TYPE = CC_DOUBLE_COMPLEX;
    SIZEOF_WORKING_TYPE = sizeof(double complex);


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
    create_spinor_info(nspinors, irreps, eps, occ);

    create_spinor_blocks(opts->tile_size);
    setup_occupation_numbers(opts, nspinors, spinor_info);
    setup_active_space(opts);
    setup_fast_access_spinor_lists();

    /*
     * final printout
     */
    print_symmetry_info();
    print_spinor_info_table();

    /*
     * additional checks:
     * 1. names of irreps in some directives of the input file can be incorrect
     */
    check_irrep_names();

    /*
     * cleanup
     */
    cc_free(occ);
    cc_free(eps);
    cc_free(mult_table);
    cc_free(irreps);

    timer_stop("pyscf");
}


void read_pyscf_integrals(char *path_integrals_1e, char *path_integrals_2e,
                          int *nspinors, int *nocc, double *enuc, double *escf, double **eps, int **occ,
                          double complex **eri, size_t **eri_index)
{
    /*
     * open file
     */
    int f = io_open(path_integrals_1e, "r");
    if (f == -1) {
        errquit("cannot open file with pyscf 1e integrals '%s'", path_integrals_1e);
    }

    /*
     * total number of spinors
     */
    char magic[8];
    io_read(f, magic, 8);
    if (memcmp(magic, "ATXCC001", 8) != 0 || cc_opts->use_goldstone) {
        errquit("Atomx backend requires ATXCC001 antisymmetrized integrals");
    }
    *nspinors = 0;
    io_read(f, nspinors, sizeof(int));
    if (*nspinors < 2 || *nspinors > CC_MAX_SPINORS) {
        errquit("Atomx spinor count exceeds this EXP-T build limit");
    }
    int64_t nspinors64 = *nspinors;

    /*
     * get nuclear repulsion and scf energies
     */
    io_read(f, enuc, sizeof(double));
    io_read(f, escf, sizeof(double));

    /*
     * get information about spinors: their occupation numbers of one-electron energies
     */
    *eps = cc_calloc(nspinors64, sizeof(double));
    *occ = cc_calloc(nspinors64, sizeof(int));

    *nocc = 0;
    for (int i = 0; i < nspinors64; i++) {
        io_read(f, *occ + i, sizeof(int));
        io_read(f, *eps + i, sizeof(double));
        *nocc += (*occ)[i];
    }

    atomx_fock = cc_calloc(nspinors64 * nspinors64, sizeof(double complex));
    io_read(f, atomx_fock, nspinors64 * nspinors64 * sizeof(double complex));

    /*
     * create index for fast random access to two-electron integrals
     */
    int n = *nspinors;
    int n_pq = n * (n - 1) / 2;
    *eri_index = cc_calloc(n_pq, sizeof(size_t));

    size_t num_unique_integrals = 0;
    size_t i_pq = 0;

    if (cc_opts->use_goldstone) {
        num_unique_integrals = n * n * n * n;
    }
    else {
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                size_t pq_size = (n - p) * (n - p - 1) / 2;
                num_unique_integrals += pq_size;

                if (i_pq < n_pq - 1) {
                    (*eri_index)[i_pq + 1] = (*eri_index)[i_pq] + pq_size;
                }

                i_pq += 1;
            }
        }
    }

    /*
     * allocate space for unique two-electron integrals
     */
    *eri = cc_calloc(num_unique_integrals, sizeof(double complex));
    if (*eri == NULL) {
        errquit("cannot allocate memory for two-electron integrals (required space: %.3f gb)",
                ((double) num_unique_integrals) / (1024.0 * 1024.0 * 1024.0));
    }

    /*
     * read two-electron integrals from disk
     */
    io_close(f);

    printf("begin reading two electron part\n");
    f = io_open(path_integrals_2e, "r");
    if (f == -1) {
        errquit("cannot open file with pyscf 2e integrals '%s'", path_integrals_2e);
    }
    io_read(f, *eri, num_unique_integrals * sizeof(double complex));
    io_close(f);
}


double complex get_eri_goldstone(int nspinors, double complex *eris, int p, int q, int r, int s)
{
    size_t nspinors_2 = nspinors * nspinors;
    size_t nspinors_3 = nspinors_2 * nspinors;

    // old version
//    return eris[nspinors_3 * p + nspinors_2 * q + nspinors * r + s];// - eris[nspinors_3 * p + nspinors_2 * q + nspinors * s + r];

    // new version: numpy.ndarray.tofile()
    return eris[nspinors_3 * p + nspinors_2 * r + nspinors * q + s];
}


double complex get_eri(int nspinors, double complex *eris, size_t *eri_index, int p, int q, int r, int s)
{
    if (p == q || r == s) {
        return 0.0 + 0.0 * I;
    }

    int p0 = p;
    int q0 = q;
    int r0 = r;
    int s0 = s;
    double sign = 1.0;

    if (p0 > q0) {
        p = q0;
        q = p0;
        sign *= -1.0;
    }

    if (r0 > s0) {
        r = s0;
        s = r0;
        sign *= -1.0;
    }

    int do_conj = 0;
    if (p > r) {
        do_conj = 1;
        p0 = p;
        q0 = q;
        r0 = r;
        s0 = s;

        p = r0;
        q = s0;
        r = p0;
        s = q0;
    }

    size_t n = nspinors;
    size_t i_pq = n * p + (q + 1) - (p + 1) * (p + 2) / 2 - 1;
    size_t m = n - p;
    size_t rx = r - p;
    size_t sx = s - p;
    size_t i_rs = m * rx + (sx + 1) - (rx + 1) * (rx + 2) / 2 - 1;
    size_t i = eri_index[i_pq] + i_rs;

    if (do_conj) {
        return sign * conj(eris[i]);
    }
    else {
        return sign * eris[i];
    }
}





