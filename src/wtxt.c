#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wtxt.h"

static float *get(const char *dir, const char *name, int n) {
    char path[256];
    float *d = malloc(sizeof(float) * n);
    FILE *f;

    if (!d) return NULL;
    snprintf(path, sizeof(path), "%s/%s.txt", dir, name);
    f = fopen(path, "r");
    if (!f) {
        free(d);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%f", &d[i]) != 1) {
            fclose(f);
            free(d);
            return NULL;
        }
    }
    fclose(f);
    return d;
}

int wtxt_load(Wtxt *w, const char *dir) {
    char name[64];
    int got = 0;

    memset(w, 0, sizeof(*w));
    if ((w->wte = get(dir, "transformer.wte.weight", N_VOCAB * N_EMBD))) got++;
    if ((w->wpe = get(dir, "transformer.wpe.weight", N_CTX * N_EMBD))) got++;
    if ((w->lnfg = get(dir, "transformer.ln_f.weight", N_EMBD))) got++;
    if ((w->lnfb = get(dir, "transformer.ln_f.bias", N_EMBD))) got++;
    for (int l = 0; l < 12; l++) {
        snprintf(name, sizeof(name), "transformer.h.%d.ln_1.weight", l);
        if ((w->ln1g[l] = get(dir, name, N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.ln_1.bias", l);
        if ((w->ln1b[l] = get(dir, name, N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.attn.c_attn.weight", l);
        if ((w->wqkv[l] = get(dir, name, N_EMBD * 3 * N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.attn.c_attn.bias", l);
        if ((w->bqkv[l] = get(dir, name, 3 * N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.attn.c_proj.weight", l);
        if ((w->wproj[l] = get(dir, name, N_EMBD * N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.attn.c_proj.bias", l);
        if ((w->bproj[l] = get(dir, name, N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.ln_2.weight", l);
        if ((w->ln2g[l] = get(dir, name, N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.ln_2.bias", l);
        if ((w->ln2b[l] = get(dir, name, N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_fc.weight", l);
        if ((w->wfc[l] = get(dir, name, N_EMBD * 4 * N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_fc.bias", l);
        if ((w->bfc[l] = get(dir, name, 4 * N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_proj.weight", l);
        if ((w->wfp[l] = get(dir, name, 4 * N_EMBD * N_EMBD))) got++;
        snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_proj.bias", l);
        if ((w->bfp[l] = get(dir, name, N_EMBD))) got++;
    }
    if (got != 148) {
        fprintf(stderr, "loaded %d/148 tensors\n", got);
        wtxt_free(w);
        return -1;
    }
    return 0;
}

void wtxt_free(Wtxt *w) {
    float **p = (float **)w;

    for (int i = 0; i < (int)(sizeof(*w) / sizeof(float *)); i++) free(p[i]);
    memset(w, 0, sizeof(*w));
}

void wtxt_wire(GPT2 *g, Wtxt *w) {
    for (int l = 0; l < 12; l++) {
        BlockW *b = &g->blocks[l];

        b->ln1g = w->ln1g[l];
        b->ln1b = w->ln1b[l];
        b->wqkv = w->wqkv[l];
        b->bqkv = w->bqkv[l];
        b->wproj = w->wproj[l];
        b->bproj = w->bproj[l];
        b->ln2g = w->ln2g[l];
        b->ln2b = w->ln2b[l];
        b->wfc = w->wfc[l];
        b->bfc = w->bfc[l];
        b->wfp = w->wfp[l];
        b->bfp = w->bfp[l];
    }
    g->lnfg = w->lnfg;
    g->lnfb = w->lnfb;
    g->wte = w->wte;
    g->wpe = w->wpe;
}
