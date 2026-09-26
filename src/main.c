#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "wtxt.h"
#include "bpe.h"
#include "model.h"
#include "chat.h"
#include "term.h"
#include "tui.h"

static int dump_weights(void) {
    Wtxt w;

    if (wtxt_load(&w, "models") != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    printf("wte.weight first 5:");
    for (int i = 0; i < 5; i++) {
        printf(" %f", w.wte[i]);
    }
    printf("\n148 tensors loaded\n");
    wtxt_free(&w);
    return 0;
}

static int encode_text(const char *text) {
    BPE b;
    int ids[1024];

    if (bpe_init(&b, "models/vocab.json", "models/merges.txt") != 0) {
        fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }

    int n = bpe_encode(&b, text, ids, 1024);
    bpe_free(&b);

    if (n < 0) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        printf("%s%d", i ? " " : "", ids[i]);
    }
    printf("\n");

    return 0;
}

typedef struct {
    Wtxt w;
    BPE b;
    GPT2 g;
    float *logits;
} Engine;

static int engine_load(Engine *e) {
    e->logits = malloc(sizeof(float) * N_VOCAB);
    if (!e->logits) {
        fprintf(stderr, "out of memory\n");
        return -1;
    }
    if (wtxt_load(&e->w, "models") != 0) {
        fprintf(stderr, "load failed\n");
        free(e->logits);
        return -1;
    }
    if (bpe_init(&e->b, "models/vocab.json", "models/merges.txt") != 0) {
        fprintf(stderr, "tokenizer load failed\n");
        wtxt_free(&e->w);
        free(e->logits);
        return -1;
    }
    wtxt_wire(&e->g, &e->w);
    return 0;
}

static void engine_free(Engine *e) {
    bpe_free(&e->b);
    wtxt_free(&e->w);
    free(e->logits);
}

static int forward_text(const char *text) {
    Engine e;
    int ids[N_CTX];

    if (engine_load(&e) != 0) {
        return 1;
    }

    int seq_len = bpe_encode(&e.b, text, ids, N_CTX);
    if (seq_len <= 0) {
        fprintf(stderr, "encode failed\n");
        engine_free(&e);
        return 1;
    }
    if (forward(&e.g, ids, e.logits, seq_len) != 0) {
        fprintf(stderr, "forward failed\n");
        engine_free(&e);
        return 1;
    }

    int top[5] = {-1, -1, -1, -1, -1};
    for (int v = 0; v < N_VOCAB; v++) {
        for (int k = 0; k < 5; k++) {
            if (top[k] < 0 || e.logits[v] > e.logits[top[k]]) {
                for (int m = 4; m > k; m--) top[m] = top[m - 1];
                top[k] = v;
                break;
            }
        }
    }
    char piece[64];
    for (int k = 0; k < 5; k++) {
        piece[0] = 0;
        bpe_decode(&e.b, &top[k], 1, piece, sizeof(piece));
        printf("%d %f [%s]\n", top[k], e.logits[top[k]], piece);
    }

    engine_free(&e);
    return 0;
}

static int generate_text(const char *text, int max) {
    Engine e;
    int ids[N_CTX];
    char out[4096];

    srand((unsigned)time(NULL));
    if (engine_load(&e) != 0) {
        return 1;
    }

    int seq_len = bpe_encode(&e.b, text, ids, N_CTX);
    if (seq_len <= 0) {
        fprintf(stderr, "encode failed\n");
        engine_free(&e);
        return 1;
    }
    int prompt_len = seq_len;
    for (int i = 0; i < max && seq_len < N_CTX; i++) {
        if (forward(&e.g, ids, e.logits, seq_len) != 0) {
            fprintf(stderr, "forward failed\n");
            engine_free(&e);
            return 1;
        }
        int next = sample(e.logits, N_VOCAB, 0.0f, 40);
        if (next < 0 || next == N_VOCAB - 1) break;
        ids[seq_len++] = next;
    }

    if (bpe_decode(&e.b, ids + prompt_len, seq_len - prompt_len, out, sizeof(out)) < 0) {
        fprintf(stderr, "decode failed\n");
        engine_free(&e);
        return 1;
    }
    printf("%s\n", out);

    engine_free(&e);
    return 0;
}

static void print_help(void) {
    printf("/quit         leave\n");
    printf("/reset        forget this conversation\n");
    printf("/temp N        sampling temperature (0 = greedy)\n");
    printf("/tokens       context usage\n");
    printf("/save FILE    save transcript\n");
    printf("/load FILE    restore transcript\n");
    printf("/help         this list\n");
}

typedef struct {
    int started;
    int count;
    clock_t t0;
} Stream;

static void on_piece(const char *piece, void *ctx) {
    Stream *s = ctx;
    if (!s->started) {
        printf("\r\033[Kai: ");
        s->started = 1;
    }
    printf("%s", piece);
    fflush(stdout);
    s->count++;
}

static int chat_repl(int max) {
    Engine e;
    Chat c;
    History hist = {0};
    char line[1024];
    char reply[2048];
    float temp = 0.8f;
    int tty = isatty(STDIN_FILENO);

    if (engine_load(&e) != 0) {
        return 1;
    }
    if (chat_init(&c) != 0) {
        fprintf(stderr, "out of memory\n");
        engine_free(&e);
        return 1;
    }

    for (;;) {
        if (tty) {
            int n = term_getline("you: ", line, sizeof(line), &hist);
            if (n < 0) break;
            if (n == 0) continue;
        } else {
            printf("you: ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (line[0] == 0) continue;
        }
        if (line[0] == '/') {
            if (strcmp(line, "/quit") == 0) break;
            if (strcmp(line, "/reset") == 0) {
                chat_reset(&c);
                continue;
            }
            if (strcmp(line, "/help") == 0) {
                print_help();
                continue;
            }
            if (strcmp(line, "/tokens") == 0) {
                printf("%d / %d tokens, %ld dropped\n", c.len, CHAT_WINDOW, c.dropped);
                continue;
            }
            if (strncmp(line, "/temp ", 6) == 0) {
                temp = (float)atof(line + 6);
                printf("temp %g\n", temp);
                continue;
            }
            if (strncmp(line, "/save ", 6) == 0) {
                printf(chat_save(&c, &e.b, line + 6) == 0 ? "saved\n" : "save failed\n");
                continue;
            }
            if (strncmp(line, "/load ", 6) == 0) {
                printf(chat_load(&c, &e.b, line + 6) == 0 ? "loaded\n" : "load failed\n");
                continue;
            }
            print_help();
            continue;
        }
        printf("ai: thinking...");
        fflush(stdout);
        Stream s = {0, 0, clock()};
        chat_on_token(on_piece, &s);
        if (chat_turn(&c, &e.g, &e.b, e.logits, line, reply, sizeof(reply), max, temp) != 0) {
            chat_on_token(NULL, NULL);
            fprintf(stderr, "turn failed\n");
            break;
        }
        chat_on_token(NULL, NULL);
        printf("\r\033[Kai: %s\n", reply[0] ? reply : "(no response)");
        printf("done - %d tokens, %.1fs\n", s.count, (double)(clock() - s.t0) / CLOCKS_PER_SEC);
        if (chat_log(line, reply) != 0) {
            fprintf(stderr, "log failed\n");
        }
    }

    chat_free(&c);
    engine_free(&e);
    return 0;
}

int main(int argc, char **argv) {
    int tui = isatty(STDIN_FILENO);
    int w = 1;

    for (int r = 1; r < argc; r++) {
        if (strcmp(argv[r], "--tui") == 0) tui = 1;
        else if (strcmp(argv[r], "--no-tui") == 0) tui = 0;
        else argv[w++] = argv[r];
    }
    argc = w;

    if (argc == 3 && strcmp(argv[1], "encode") == 0) {
        return encode_text(argv[2]);
    }

    if (argc == 3 && strcmp(argv[1], "forward") == 0) {
        return forward_text(argv[2]);
    }

    if ((argc == 3 || argc == 4) && strcmp(argv[1], "generate") == 0) {
        int max = (argc == 4) ? atoi(argv[3]) : 30;
        return generate_text(argv[2], max);
    }

    if ((argc == 2 || argc == 3) && strcmp(argv[1], "chat") == 0) {
        int max = (argc == 3) ? atoi(argv[2]) : 120;
        return tui ? tui_chat(max) : chat_repl(max);
    }

    if (argc == 1) {
        return tui ? tui_chat(120) : chat_repl(120);
    }

    return dump_weights();
}
