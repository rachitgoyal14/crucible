#include <math.h>
#include <stdio.h>
#include <string.h>
#include "model.h"
#include "wtxt.h"

static int test_qkv(void) {
    static float w[N_EMBD * 3 * N_EMBD];
    static float b[3 * N_EMBD];
    static float p[N_EMBD * N_EMBD];
    static float x[N_EMBD];
    static float o[N_EMBD];

    for (int i = 0; i < N_EMBD; i++) {
        for (int j = 0; j < N_EMBD; j++) {
            w[i * 3 * N_EMBD + j] = (i == j);
            w[i * 3 * N_EMBD + N_EMBD + j] = 2 * (i == j);
            w[i * 3 * N_EMBD + 2 * N_EMBD + j] = 3 * (i == j);
            p[i * N_EMBD + j] = (i == j);
        }
    }
    x[0] = 1.0f;
    if (mha(x, w, b, p, b, o, 1) != 0) return -1;
    if (o[0] != 3.0f) return -1;
    for (int i = 1; i < N_EMBD; i++) {
        if (o[i] != 0.0f) return -1;
    }
    return 0;
}

static int test_sample(void) {
    float l[4] = {1.0f, 5.0f, 3.0f, 2.0f};
    if (sample(l, 4, 0.0f, 40) != 1) return -1;
    for (int i = 0; i < 200; i++) {
        int s = sample(l, 4, 1.0f, 2);
        if (s != 1 && s != 2) return -1;
    }
    for (int i = 0; i < 50; i++) {
        int s = sample(l, 4, 1e-9f, 2);
        if (s != 1 && s != 2) return -1;
    }
    return 0;
}

static int test_repeat(void) {
    static float a[N_VOCAB];
    static float b[N_VOCAB];
    static int ids4[4] = {15496, 11, 995, 0};
    Wtxt w;
    GPT2 g;

    if (wtxt_load(&w, "models") != 0) {
        printf("FAIL wtxt load\n");
        return -1;
    }
    wtxt_wire(&g, &w);
    if (forward(&g, ids4, a, 4) != 0) {
        printf("FAIL forward\n");
        return -1;
    }
    if (forward(&g, ids4, b, 4) != 0) {
        printf("FAIL forward repeat\n");
        return -1;
    }
    if (memcmp(a, b, sizeof(a)) != 0) {
        printf("FAIL repeat parity\n");
        return -1;
    }
    wtxt_free(&w);
    return 0;
}

int main(void) {
    static float wte[3 * N_EMBD];
    static float wpe[2 * N_EMBD];
    static float out[2 * N_EMBD];

    for (int j = 0; j < N_EMBD; j++) {
        for (int r = 0; r < 3; r++) wte[r * N_EMBD + j] = r * 10 + j;
        for (int i = 0; i < 2; i++) wpe[i * N_EMBD + j] = i + j;
    }

    int ids[2] = {2, 0};
    if (embed(wte, wpe, ids, out, 2) != 0) {
        printf("FAIL rc\n");
        return 1;
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < N_EMBD; j++) {
            float want = ids[i] * 10 + i + 2 * j;
            if (out[i * N_EMBD + j] != want) {
                printf("FAIL [%d][%d] %f != %f\n", i, j, out[i * N_EMBD + j], want);
                return 1;
            }
        }
    }

    if (embed(wte, wpe, ids, out, N_CTX + 1) == 0) {
        printf("FAIL over-context accepted\n");
        return 1;
    }

    float x[4] = {1, 0, 0, 1};
    float eye[4] = {1, 0, 0, 1};
    float ao[4] = {0};
    float want_a[4] = {1.0f, 0.0f, 0.330238f, 0.669762f};
    if (attn_head(x, eye, eye, eye, ao, 2, 2, 2) != 0) {
        printf("FAIL attn rc\n");
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        float d = ao[i] - want_a[i];
        if (d < 0) d = -d;
        if (d > 1e-4f) {
            printf("FAIL attn[%d] %f != %f\n", i, ao[i], want_a[i]);
            return 1;
        }
    }

    static float wqkv[N_EMBD * 3 * N_EMBD];
    static float bqkv[3 * N_EMBD];
    static float wproj[N_EMBD * N_EMBD];
    static float bproj[N_EMBD];
    static float mx[2 * N_EMBD];
    static float mo[2 * N_EMBD];
    for (int i = 0; i < N_EMBD; i++) {
        for (int j = 0; j < N_EMBD; j++) {
            wqkv[i * 3 * N_EMBD + j] = (i == j);
            wqkv[i * 3 * N_EMBD + N_EMBD + j] = (i == j);
            wqkv[i * 3 * N_EMBD + 2 * N_EMBD + j] = (i == j);
            wproj[i * N_EMBD + j] = (i == j);
        }
    }
    for (int i = 0; i < 3 * N_EMBD; i++) bqkv[i] = 0.5f;
    mx[0] = 1.0f;
    if (mha(mx, wqkv, bqkv, wproj, bproj, mo, 1) != 0) {
        printf("FAIL mha rc\n");
        return 1;
    }
    for (int h = 0; h < N_HEAD; h++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            float want = (h == 0 && j == 0) ? 1.5f : 0.5f;
            float got = mo[h * HEAD_DIM + j];
            if (got != want) {
                printf("FAIL mha[%d][%d] %f != %f\n", h, j, got, want);
                return 1;
            }
        }
    }

    mx[N_EMBD + 1] = 1.0f;
    if (mha(mx, wqkv, bqkv, wproj, bproj, mo, 2) != 0) {
        printf("FAIL mha2 rc\n");
        return 1;
    }
    for (int i = 0; i < 2 * N_EMBD; i++) {
        if (!isfinite(mo[i])) {
            printf("FAIL mha2 non-finite [%d]\n", i);
            return 1;
        }
    }

    static float zone[2 * N_EMBD];
    static float bone[2 * N_EMBD];
    static float bln[2 * N_EMBD];
    static float bwqkv[N_EMBD * 3 * N_EMBD];
    static float bbqkv[3 * N_EMBD];
    static float bwproj[N_EMBD * N_EMBD];
    static float bbproj[N_EMBD];
    static float bwfc[N_EMBD * 4 * N_EMBD];
    static float bbfc[4 * N_EMBD];
    static float bwfp[4 * N_EMBD * N_EMBD];
    static float bbfp[N_EMBD];
    for (int i = 0; i < N_EMBD; i++) {
        bln[i] = 1.0f;
        bbproj[i] = 0.0f;
        for (int j = 0; j < N_EMBD; j++) {
            bwproj[i * N_EMBD + j] = (i == j);
        }
    }
    for (int i = 0; i < N_EMBD; i++) {
        for (int j = 0; j < 4 * N_EMBD; j++) {
            bwfc[i * 4 * N_EMBD + j] = (i == j);
            bwfp[j * N_EMBD + i] = (i == j);
        }
    }
    BlockW bw = {
        bln, zone, bwqkv, bbqkv, bwproj, bbproj,
        bln, zone, bwfc, bbfc, bwfp, bbfp,
    };
    if (block(zone, &bw, bone, 2) != 0) {
        printf("FAIL block rc\n");
        return 1;
    }
    for (int i = 0; i < 2 * N_EMBD; i++) {
        if (bone[i] != 0.0f) {
            printf("FAIL block zero [%d] %f\n", i, bone[i]);
            return 1;
        }
    }
    for (int i = 0; i < N_EMBD; i++) bbproj[i] = 1.0f;
    if (block(zone, &bw, bone, 2) != 0) {
        printf("FAIL block rc2\n");
        return 1;
    }
    for (int i = 0; i < 2 * N_EMBD; i++) {
        if (bone[i] != 1.0f) {
            printf("FAIL block ones [%d] %f\n", i, bone[i]);
            return 1;
        }
    }

    if (test_qkv() != 0) {
        printf("FAIL qkv\n");
        return 1;
    }

    if (test_sample() != 0) {
        printf("FAIL sample\n");
        return 1;
    }

    if (test_repeat() != 0) {
        printf("FAIL repeat\n");
        return 1;
    }

    printf("all pass\n");
    return 0;
}
