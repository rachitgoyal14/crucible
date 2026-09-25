#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat.h"

int chat_init(Chat *c) {
    c->ids = malloc(sizeof(int) * CHAT_WINDOW);
    if (!c->ids) return -1;
    c->len = 0;
    c->cap = CHAT_WINDOW;
    c->dropped = 0;
    return 0;
}

void chat_free(Chat *c) {
    free(c->ids);
    c->ids = NULL;
    c->len = 0;
    c->cap = 0;
}

void chat_reset(Chat *c) {
    c->len = 0;
}

int chat_add(Chat *c, const int *ids, int n) {
    if (n >= c->cap) {
        memcpy(c->ids, ids + n - c->cap, sizeof(int) * c->cap);
        c->len = c->cap;
        c->dropped += n - c->cap;
        fprintf(stderr, "truncated %d tokens\n", n - c->cap);
        return 0;
    }
    if (c->len + n > c->cap) {
        int drop = c->len + n - c->cap;
        memmove(c->ids, c->ids + drop, sizeof(int) * (c->len - drop));
        c->len -= drop;
        c->dropped += drop;
        fprintf(stderr, "truncated %d tokens\n", drop);
    }
    memcpy(c->ids + c->len, ids, sizeof(int) * n);
    c->len += n;
    return 0;
}

/* earliest fabricated turn header inside a reply, or NULL */
static char *turn_stop(char *s) {
    char *a = strstr(s, "\nAssistant:");
    char *u = strstr(s, "\nUser:");
    if (a && u) return a < u ? a : u;
    return a ? a : u;
}

void strip_echo(char *s) {
    char *p = turn_stop(s);
    if (p) *p = 0;
}

static tok_cb tokfn = NULL;
static void *tokctx = NULL;

void chat_on_token(tok_cb cb, void *ctx) {
    tokfn = cb;
    tokctx = ctx;
}

int looping(const int *ids, int len) {
    for (int p = 1; p <= 4; p++) {
        if (len < 4 * p) continue;
        int ok = 1;
        for (int i = 0; i < 3 * p; i++) {
            if (ids[len - 1 - i] != ids[len - 1 - i - p]) {
                ok = 0;
                break;
            }
        }
        if (ok) return 1;
    }
    return 0;
}

int chat_turn(Chat *c, GPT2 *g, BPE *b, float *logits, const char *msg, char *out, int outcap, int max_tokens, float temp) {
    char prompt[2048];
    int ids[N_CTX];

    snprintf(prompt, sizeof(prompt), "User: %s\nAssistant:", msg);
    int n = bpe_encode(b, prompt, ids, N_CTX);
    if (n < 0) return -1;
    if (chat_add(c, ids, n) != 0) return -1;

    int made = 0, sent = 0;
    int ntok = max_tokens > 0 ? max_tokens : 1;
    int *fresh = malloc(sizeof(int) * ntok);
    char *acc = malloc((size_t)ntok * 32 + 128);
    if (!fresh || !acc) {
        free(fresh);
        free(acc);
        return -1;
    }
    acc[0] = 0;
    int fabricated = 0;
    while (made < max_tokens && c->len < N_CTX) {
        if (forward(g, c->ids, logits, c->len) != 0) {
            free(fresh);
            free(acc);
            return -1;
        }
        int next = sample(logits, N_VOCAB, temp, 40);
        if (next < 0 || next == N_VOCAB - 1) break;
        if (chat_add(c, &next, 1) != 0) {
            free(fresh);
            free(acc);
            return -1;
        }
        fresh[made++] = next;
        if (bpe_decode(b, fresh, made, acc, ntok * 32 + 128) < 0) break;
        char *p = turn_stop(acc);
        if (p) {
            *p = 0;
            fabricated = 1;
        }
        int n = (int)strlen(acc);
        if (tokfn && n > sent) tokfn(acc + sent, tokctx);
        sent = n;
        if (fabricated) break;
        if (looping(c->ids, c->len)) break;
    }

    if (fabricated) {
        /* keep only the clean reply in context so the next turn starts fresh */
        int tmp[N_CTX];
        int nclean = bpe_encode(b, acc, tmp, N_CTX);
        c->len -= made;
        if (nclean > 0) chat_add(c, tmp, nclean);
    }

    snprintf(out, outcap, "%s", acc);
    free(fresh);
    free(acc);
    strip_echo(out);
    return 0;
}

int chat_save(Chat *c, BPE *b, const char *path) {
    char *text = malloc(N_CTX * 8 + 1);
    FILE *f;
    if (!text) return -1;
    if (bpe_decode(b, c->ids, c->len, text, N_CTX * 8 + 1) < 0) {
        free(text);
        return -1;
    }
    f = fopen(path, "w");
    if (!f) {
        free(text);
        return -1;
    }
    fputs(text, f);
    fclose(f);
    free(text);
    return 0;
}

int chat_load(Chat *c, BPE *b, const char *path) {
    FILE *f = fopen(path, "rb");
    long size;
    char *text;
    int ids[N_CTX];
    int n;

    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    text = malloc(size + 1);
    if (!text || fread(text, 1, size, f) != (size_t)size) {
        free(text);
        fclose(f);
        return -1;
    }
    text[size] = 0;
    fclose(f);

    long keep = size;
    n = -1;
    while (keep > 0) {
        n = bpe_encode(b, text + size - keep, ids, N_CTX);
        if (n >= 0) break;
        keep /= 2;
        fprintf(stderr, "log too long, kept last %ld bytes\n", keep);
    }
    free(text);
    if (n < 0) return -1;
    chat_reset(c);
    return chat_add(c, ids, n);
}

int chat_log(const char *user, const char *reply) {
    FILE *f = fopen("chat.log", "a");
    if (!f) return -1;
    fprintf(f, "User: %s\nAssistant: %s\n", user, reply);
    fclose(f);
    return 0;
}
