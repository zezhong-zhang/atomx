
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include "diagram.h"
size_t n_spinor_blocks;
static size_t allocated;
void *cc_malloc(size_t n) { allocated += n; void *p=malloc(n); if(!p) exit(90); return p; }
void cc_free(void *p) { free(p); }
void errquit(const char *fmt, ...) { fprintf(stderr,"EXPECTED_REJECTION: %s\n",fmt); exit(87); }
void diagram_init_inverse_index(diagram_t *, size_t, block_t **);
static void coordinates(size_t key,int rank,int base,int *a) {
    for(int j=rank-1;j>=0;j--) { a[j]=key%base; key/=base; }
}
int main(int argc,char **argv) {
    if(argc>1) {
        diagram_t d={0}; block_t b[2]={0}; block_t *v[]={b,b+1};
        d.rank=6; d.n_blocks=2; d.blocks=v; n_spinor_blocks=3;
        if(!strcmp(argv[1],"overflow")) { n_spinor_blocks=SIZE_MAX; b[0].spinor_blocks[0]=2; }
        diagram_init_inverse_index(&d,2,v); return 91;
    }
    size_t queries=0;
    for(int rank=2;rank<=6;rank+=2) for(int base=2;base<=5;base++) {
        n_spinor_blocks=base; size_t total=1; for(int k=0;k<rank;k++) total*=base;
        diagram_t d={0}; d.rank=rank;
        block_t *all=calloc(total,sizeof(block_t));
        block_t **v=calloc(total,sizeof(block_t *)); size_t count=0;
        /* Deliberately reverse creation order and omit exact structural zeros. */
        for(size_t k=total;k-->0;) if(k%7!=0) {
            coordinates(k,rank,base,all[k].spinor_blocks); v[count++]=all+k;
        }
        d.blocks=v; d.n_blocks=count; allocated=0;
        diagram_init_inverse_index(&d,count,v);
        if(allocated!=2*sizeof(size_t)*count) return 92;
        for(size_t k=0;k<total;k++) {
            int a[CC_DIAGRAM_MAX_RANK]={0}; coordinates(k,rank,base,a);
            block_t *expected=k%7 ? all+k : NULL;
            if(diagram_get_block(&d,a)!=expected) return 93;
            queries++;
        }
        cc_free(d.inv_index); free(v); free(all);
    }
    diagram_t d={0}; d.rank=6; d.n_blocks=5; n_spinor_blocks=64;
    block_t b[5]={0}; block_t *v[5]; size_t total=(size_t)64*64*64*64*64*64;
    size_t keys[]={total-1,0,4095,15,1};
    for(int i=0;i<5;i++) { coordinates(keys[i],6,64,b[i].spinor_blocks); v[i]=b+i; }
    d.blocks=v; allocated=0; diagram_init_inverse_index(&d,5,v);
    if(allocated!=80) return 94;
    for(size_t k=0;k<4096;k++) {
        int a[CC_DIAGRAM_MAX_RANK]={0}; coordinates(k,6,64,a); block_t *expected=NULL;
        for(int i=0;i<5;i++) if(keys[i]==k) expected=b+i;
        if(diagram_get_block(&d,a)!=expected) return 95;
        queries++;
    }
    int a[CC_DIAGRAM_MAX_RANK]={63,63,63,63,63,63};
    if(diagram_get_block(&d,a)!=b) return 96;
    cc_free(d.inv_index); d.n_blocks=0;
    if(diagram_get_block(&d,a)!=NULL) return 97;
    printf("{\"status\":\"pass\",\"complete_small_queries\":%zu,\"large_base\":64,\"rank\":6,\"cartesian_extent\":%zu,\"dense_index_bytes\":%zu,\"compact_five_block_bytes\":%zu}\n",queries,total,total*sizeof(size_t),allocated);
    return 0;
}
