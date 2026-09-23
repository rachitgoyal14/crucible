#ifndef MODEL_H
#define MODEL_H

#define N_EMBD 768
#define N_CTX 1024
#define N_HEAD 12
#define HEAD_DIM 64

int embed(const float *wte, const float *wpe, const int *ids, float *out, int seq_len);
int attn_head(const float *x, const float *wq, const float *wk, const float *wv, float *out, int seq_len, int dim, int hd);
int attn_raw(const float *q, const float *k, const float *v, float *out, int seq_len, int hd);
int mha(const float *x, const float *wqkv, const float *bqkv, const float *wproj, const float *bproj, float *out, int seq_len);

typedef struct {
    const float *ln1g, *ln1b;
    const float *wqkv, *bqkv, *wproj, *bproj;
    const float *ln2g, *ln2b;
    const float *wfc, *bfc, *wfp, *bfp;
} BlockW;

int block(const float *x, const BlockW *w, float *out, int seq_len);

#define N_LAYER 12
#define N_VOCAB 50257

typedef struct {
    BlockW blocks[N_LAYER];
    const float *lnfg, *lnfb;
    const float *wte, *wpe;
} GPT2;

int forward(GPT2 *g, const int *ids, float *logits, int seq_len);
int sample(const float *logits, int n, float temp, int topk);

#endif
