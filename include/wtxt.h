#ifndef WTXT_H
#define WTXT_H

#include "model.h"

typedef struct {
    float *wte, *wpe, *lnfg, *lnfb;
    float *ln1g[12], *ln1b[12], *wqkv[12], *bqkv[12];
    float *wproj[12], *bproj[12], *ln2g[12], *ln2b[12];
    float *wfc[12], *bfc[12], *wfp[12], *bfp[12];
} Wtxt;

int wtxt_load(Wtxt *w, const char *dir);
void wtxt_free(Wtxt *w);
void wtxt_wire(GPT2 *g, Wtxt *w);

#endif
