"""Optional exact M/parity integral bank over the pinned Atomx EXP-T ABI.

The sidecar selects the new layout explicitly. Legacy files/build defaults
are retained. This changes integral storage and indexing, never CC equations.
"""
import difflib
from pathlib import Path


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('compact integral patch does not match pinned interface')
    return text.replace(old, new)


def patch_compact_eri(source):
    path = Path(source)/'src/rcc/interfaces/pyscf_interface.c'
    old = text = path.read_text()
    if 'ATOMX_AXIAL_V1' not in text:
        raise ValueError('compact integral storage requires exact axial support')
    text = _once(text, '#include <time.h>', '#include <time.h>\n#include <sys/stat.h>')
    text = _once(text, 'double complex *atomx_fock = NULL;',
                 'double complex *atomx_fock = NULL;\n'+_INDEX)
    text = _once(text, '    *eri = cc_calloc(num_unique_integrals, sizeof(double complex));',
                 '''    num_unique_integrals = atomx_compact_eri_init(
        path_integrals_1e, n, *eri_index, num_unique_integrals);
    *eri = cc_calloc(num_unique_integrals, sizeof(double complex));''')
    text = _once(text, '    io_read(f, *eri, num_unique_integrals * sizeof(double complex));',
                 '''    if (atomx_compact_eri_mode) {
        struct stat st;
        if (stat(path_integrals_2e, &st) != 0 || st.st_size < 0 ||
            (uint64_t)st.st_size != (uint64_t)num_unique_integrals * sizeof(double complex)) {
            errquit("compact axial integral file length mismatch");
        }
    }
    io_read(f, *eri, num_unique_integrals * sizeof(double complex));''')
    text = _once(text, '    size_t i = eri_index[i_pq] + i_rs;',
                 '''    size_t i;
    if (atomx_compact_eri_mode) {
        int g = atomx_eri_group(p, q);
        if (g != atomx_eri_group(r, s)) return 0.0 + 0.0 * I;
        i = eri_index[i_pq] + atomx_pair_rank[n*r+s]
            - atomx_pair_prefix[p*atomx_pair_groups+g];
    } else {
        i = eri_index[i_pq] + i_rs;
    }''')
    path.write_text(text)
    patch = ''.join(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
        fromfile='a/src/rcc/interfaces/pyscf_interface.c',
        tofile='b/src/rcc/interfaces/pyscf_interface.c'))
    path = Path(source)/'src/rcc/sorting/sort_pyscf.c'
    old = text = path.read_text()
    text = _once(text, 'void pyscf_data_free()\n{',
                 'void atomx_free_compact_eri_index(void);\n\nvoid pyscf_data_free()\n{\n    atomx_free_compact_eri_index();')
    path.write_text(text)
    return patch+''.join(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
        fromfile='a/src/rcc/sorting/sort_pyscf.c', tofile='b/src/rcc/sorting/sort_pyscf.c'))


_INDEX = r'''
/* Exact pair lists ordered by their first index. Each outer (p,q) retains
 * the suffix r>=p of its M/parity group, including all original duplicates. */
static int atomx_compact_eri_mode = 0;
static int atomx_pair_groups = 0;
static int atomx_pair_max_m = 0;
static int *atomx_pair_m = NULL;
static int *atomx_pair_parity = NULL;
static size_t *atomx_pair_rank = NULL;
static size_t *atomx_pair_prefix = NULL;

static int atomx_eri_group(int p, int q)
{
    return 2*(atomx_pair_m[p]+atomx_pair_m[q]+2*atomx_pair_max_m)
        + (atomx_pair_parity[p] != atomx_pair_parity[q]);
}

void atomx_free_compact_eri_index(void)
{
    cc_free(atomx_pair_m); atomx_pair_m = NULL;
    cc_free(atomx_pair_parity); atomx_pair_parity = NULL;
    cc_free(atomx_pair_rank); atomx_pair_rank = NULL;
    cc_free(atomx_pair_prefix); atomx_pair_prefix = NULL;
    atomx_compact_eri_mode = 0;
}

static void atomx_sidecar_path(const char *one, const char *name, char *path)
{
    const char *slash = strrchr(one, '/');
    size_t parent = slash == NULL ? 0 : (size_t)(slash-one+1);
    if (parent+strlen(name)+1 > CC_MAX_PATH_LENGTH) {
        errquit("compact integral sidecar path exceeds capacity");
    }
    memcpy(path, one, parent);
    strcpy(path+parent, name);
}

static size_t atomx_compact_eri_init(const char *one, int n,
                                    size_t *offsets, size_t old_count)
{
    char path[CC_MAX_PATH_LENGTH], magic[32], excess[2];
    int count;
    size_t declared;
    atomx_sidecar_path(one, "ATOMX_ERI_COMPACT", path);
    FILE *f = fopen(path, "r");
    if (f == NULL) return old_count;
    if (fscanf(f, "%31s %d %zu", magic, &count, &declared) != 3 ||
        strcmp(magic, "ATOMX_AXIAL_ERI_001") != 0 || count != n ||
        fscanf(f, "%1s", excess) != EOF) {
        errquit("compact axial integral sidecar count/version mismatch");
    }
    fclose(f);
    atomx_sidecar_path(one, "ATOMX_AXIAL", path);
    f = fopen(path, "r");
    if (f == NULL || fscanf(f, "%31s %d", magic, &count) != 2 ||
        strcmp(magic, "ATOMX_AXIAL_001") != 0 || count != n) {
        errquit("compact integrals require the exact axial spinor sidecar");
    }
    atomx_pair_m = cc_calloc(n, sizeof(int));
    atomx_pair_parity = cc_calloc(n, sizeof(int));
    for (int p=0; p<n; p++) {
        int m, parity;
        if (fscanf(f, "%d %d", &m, &parity) != 2 || m == 0 ||
            abs(m)%2 != 1 || 6*abs(m) > CC_MAX_NUM_IRREPS/4-2 ||
            (parity != 1 && parity != -1)) {
            errquit("invalid compact integral spinor symmetry");
        }
        atomx_pair_m[p] = m;
        atomx_pair_parity[p] = parity;
        if (abs(m) > atomx_pair_max_m) atomx_pair_max_m = abs(m);
    }
    if (fscanf(f, "%1s", excess) != EOF) errquit("extra compact integral spinor data");
    fclose(f);
    atomx_pair_groups = 2*(4*atomx_pair_max_m+1);
    size_t *sizes = cc_calloc(atomx_pair_groups, sizeof(size_t));
    atomx_pair_rank = cc_calloc((size_t)n*n, sizeof(size_t));
    atomx_pair_prefix = cc_calloc((size_t)n*atomx_pair_groups, sizeof(size_t));
    for (int p=0; p<n; p++) {
        memcpy(atomx_pair_prefix+p*atomx_pair_groups, sizes,
               atomx_pair_groups*sizeof(size_t));
        for (int q=p+1; q<n; q++) {
            int g = atomx_eri_group(p, q);
            atomx_pair_rank[(size_t)n*p+q] = sizes[g]++;
        }
    }
    size_t total = 0, pq = 0;
    for (int p=0; p<n-1; p++) {
        for (int q=p+1; q<n; q++) {
            int g = atomx_eri_group(p, q);
            offsets[pq++] = total;
            total += sizes[g]-atomx_pair_prefix[p*atomx_pair_groups+g];
        }
    }
    cc_free(sizes);
    if (total != declared) errquit("compact integral exact coordinate count differs");
    atomx_compact_eri_mode = 1;
    printf("\nATOMX_AXIAL_ERI_V1 exact signed compact integral bank: %zu values\n", total);
    return total;
}
'''
