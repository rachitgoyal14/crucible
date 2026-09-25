#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "ops.h"

int embed(const float *wte, const float *wpe, const int *ids, float *out, int seq_len) {
    if (seq_len > N_CTX) return -1;
    for (int i = 0; i < seq_len; i++) {
        memcpy(out + i * N_EMBD, wte + ids[i] * N_EMBD, sizeof(float) * N_EMBD);
        for (int j = 0; j < N_EMBD; j++) {
            out[i * N_EMBD + j] += wpe[i * N_EMBD + j];
        }
    }
    return 0;
}

int attn_head(const float *x, const float *wq, const float *wk, const float *wv, float *out, int seq_len, int dim, int hd) {
    float *q = malloc(sizeof(float) * seq_len * hd);
    float *k = malloc(sizeof(float) * seq_len * hd);
    float *v = malloc(sizeof(float) * seq_len * hd);
    if (!q || !k || !v) {
        free(q);
        free(k);
        free(v);
        return -1;
    }

    matmul(x, wq, q, seq_len, dim, hd);
    matmul(x, wk, k, seq_len, dim, hd);
    matmul(x, wv, v, seq_len, dim, hd);

    int rc = attn_raw(q, k, v, out, seq_len, hd);

    free(q);
    free(k);
    free(v);
    return rc;
}

int attn_raw(const float *q, const float *k, const float *v, float *out, int seq_len, int hd) {
    float *s = malloc(sizeof(float) * seq_len * seq_len);
    if (!s) return -1;

    float scale = 1.0f / sqrtf((float)hd);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                s[i * seq_len + j] = -INFINITY;
                continue;
            }
            float d = 0.0f;
            for (int h = 0; h < hd; h++) {
                d += q[i * hd + h] * k[j * hd + h];
            }
            s[i * seq_len + j] = d * scale;
        }
        softmax(s + i * seq_len, s + i * seq_len, seq_len);
    }

    matmul(s, v, out, seq_len, seq_len, hd);

    free(s);
    return 0;
}

int mha(const float *x, const float *wqkv, const float *bqkv, const float *wproj, const float *bproj, float *out, int seq_len) {
    float *qkv = malloc(sizeof(float) * seq_len * 3 * N_EMBD);
    float *cat = malloc(sizeof(float) * seq_len * N_EMBD);
    float *qh = malloc(sizeof(float) * seq_len * HEAD_DIM);
    float *kh = malloc(sizeof(float) * seq_len * HEAD_DIM);
    float *vh = malloc(sizeof(float) * seq_len * HEAD_DIM);
    float *oh = malloc(sizeof(float) * seq_len * HEAD_DIM);
    if (!qkv || !cat || !qh || !kh || !vh || !oh) {
        free(qkv);
        free(cat);
        free(qh);
        free(kh);
        free(vh);
        free(oh);
        return -1;
    }

    matmul(x, wqkv, qkv, seq_len, N_EMBD, 3 * N_EMBD);
    for (int i = 0; i < seq_len; i++) {
        add(qkv + i * 3 * N_EMBD, bqkv, qkv + i * 3 * N_EMBD, 3 * N_EMBD);
    }

    for (int h = 0; h < N_HEAD; h++) {
        for (int i = 0; i < seq_len; i++) {
            memcpy(qh + i * HEAD_DIM, qkv + i * 3 * N_EMBD + h * HEAD_DIM, sizeof(float) * HEAD_DIM);
            memcpy(kh + i * HEAD_DIM, qkv + i * 3 * N_EMBD + N_EMBD + h * HEAD_DIM, sizeof(float) * HEAD_DIM);
            memcpy(vh + i * HEAD_DIM, qkv + i * 3 * N_EMBD + 2 * N_EMBD + h * HEAD_DIM, sizeof(float) * HEAD_DIM);
        }
        if (attn_raw(qh, kh, vh, oh, seq_len, HEAD_DIM) != 0) {
            free(qkv);
            free(cat);
            free(qh);
            free(kh);
            free(vh);
            free(oh);
            return -1;
        }
        for (int i = 0; i < seq_len; i++) {
            memcpy(cat + i * N_EMBD + h * HEAD_DIM, oh + i * HEAD_DIM, sizeof(float) * HEAD_DIM);
        }
    }

    matmul(cat, wproj, out, seq_len, N_EMBD, N_EMBD);
    for (int i = 0; i < seq_len; i++) {
        add(out + i * N_EMBD, bproj, out + i * N_EMBD, N_EMBD);
    }

    free(qkv);
    free(cat);
    free(qh);
    free(kh);
    free(vh);
    free(oh);
    return 0;
}

int block(const float *x, const BlockW *w, float *out, int seq_len) {
    float *n1 = malloc(sizeof(float) * seq_len * N_EMBD);
    float *a = malloc(sizeof(float) * seq_len * N_EMBD);
    float *n2 = malloc(sizeof(float) * seq_len * N_EMBD);
    float *m = malloc(sizeof(float) * seq_len * 4 * N_EMBD);
    if (!n1 || !a || !n2 || !m) {
        free(n1);
        free(a);
        free(n2);
        free(m);
        return -1;
    }

    for (int i = 0; i < seq_len; i++) {
        layernorm(x + i * N_EMBD, w->ln1g, w->ln1b, n1 + i * N_EMBD, N_EMBD, 1e-5f);
    }
    if (mha(n1, w->wqkv, w->bqkv, w->wproj, w->bproj, a, seq_len) != 0) {
        free(n1);
        free(a);
        free(n2);
        free(m);
        return -1;
    }
    for (int i = 0; i < seq_len; i++) {
        add(x + i * N_EMBD, a + i * N_EMBD, a + i * N_EMBD, N_EMBD);
        layernorm(a + i * N_EMBD, w->ln2g, w->ln2b, n2 + i * N_EMBD, N_EMBD, 1e-5f);
    }

    matmul(n2, w->wfc, m, seq_len, N_EMBD, 4 * N_EMBD);
    for (int i = 0; i < seq_len; i++) {
        add(m + i * 4 * N_EMBD, w->bfc, m + i * 4 * N_EMBD, 4 * N_EMBD);
    }
    gelu(m, m, seq_len * 4 * N_EMBD);
    matmul(m, w->wfp, n1, seq_len, 4 * N_EMBD, N_EMBD);
    for (int i = 0; i < seq_len; i++) {
        add(n1 + i * N_EMBD, w->bfp, n1 + i * N_EMBD, N_EMBD);
        add(a + i * N_EMBD, n1 + i * N_EMBD, out + i * N_EMBD, N_EMBD);
    }

    free(n1);
    free(a);
    free(n2);
    free(m);
    return 0;
}

int forward(GPT2 *g, const int *ids, float *logits, int seq_len) {
    float *x = malloc(sizeof(float) * seq_len * N_EMBD);
    float *y = malloc(sizeof(float) * seq_len * N_EMBD);
    if (!x || !y) {
        free(x);
        free(y);
        return -1;
    }

    if (embed(g->wte, g->wpe, ids, x, seq_len) != 0) {
        free(x);
        free(y);
        return -1;
    }
    for (int l = 0; l < N_LAYER; l++) {
        if (block(x, &g->blocks[l], y, seq_len) != 0) {
            free(x);
            free(y);
            return -1;
        }
        float *tmp = x;
        x = y;
        y = tmp;
    }
    for (int i = 0; i < seq_len; i++) {
        layernorm(x + i * N_EMBD, g->lnfg, g->lnfb, y + i * N_EMBD, N_EMBD, 1e-5f);
    }
    for (int v = 0; v < N_VOCAB; v++) {
        float d = 0.0f;
        for (int j = 0; j < N_EMBD; j++) {
            d += y[(seq_len - 1) * N_EMBD + j] * g->wte[v * N_EMBD + j];
        }
        logits[v] = d;
    }

    free(x);
    free(y);
    return 0;
}

int sample(const float *logits, int n, float temp, int topk) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    if (topk <= 0 || topk > n) topk = n;
    if (temp < 1e-5f) temp = 1e-5f;
    int *idx = malloc(sizeof(int) * n);
    if (!idx) return -1;
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < topk; i++) {
        for (int j = i + 1; j < n; j++) {
            if (logits[idx[j]] > logits[idx[i]]) {
                int t = idx[i];
                idx[i] = idx[j];
                idx[j] = t;
            }
        }
    }

    float max = logits[idx[0]];
    float sum = 0.0f;
    for (int i = 0; i < topk; i++) {
        sum += expf((logits[idx[i]] - max) / temp);
    }
    float r = (float)rand() / (float)RAND_MAX * sum;
    float acc = 0.0f;
    int pick = idx[topk - 1];
    for (int i = 0; i < topk; i++) {
        acc += expf((logits[idx[i]] - max) / temp);
        if (r < acc) {
            pick = idx[i];
            break;
        }
    }
    free(idx);
    return pick;
}
