#ifndef CHAT_H
#define CHAT_H

#include "bpe.h"
#include "model.h"

#define CHAT_WINDOW 900

typedef struct {
    int *ids;
    int len;
    int cap;
    long dropped;
} Chat;

int chat_init(Chat *c);
void chat_free(Chat *c);
void chat_reset(Chat *c);
int chat_add(Chat *c, const int *ids, int n);
void strip_echo(char *s);
int looping(const int *ids, int len);
typedef void (*tok_cb)(const char *piece, void *ctx);
void chat_on_token(tok_cb cb, void *ctx);
int chat_turn(Chat *c, GPT2 *g, BPE *b, float *logits, const char *msg, char *out, int outcap, int max_tokens, float temp);
int chat_save(Chat *c, BPE *b, const char *path);
int chat_load(Chat *c, BPE *b, const char *path);
int chat_log(const char *user, const char *reply);

#endif
