#include <string.h>
#include "tui_tr.h"

TrMsg tr[TR_N];
int trn, troff;

static void push(char t, const char *s) {
    if (trn == TR_N) {
        memmove(tr, tr + 1, sizeof(tr) - sizeof(tr[0]));
        trn--;
        if (troff > 0) troff--;
    }
    tr[trn].tag = t;
    strncpy(tr[trn].text, s, TR_W - 1);
    tr[trn].text[TR_W - 1] = 0;
    trn++;
}

void tr_push(char tag, const char *s) {
    push(tag, s);
}

void tr_append(char tag, const char *s) {
    char *dst;
    int len;

    if (trn == 0 || tr[trn - 1].tag != tag) push(tag, "");
    dst = tr[trn - 1].text;
    len = (int)strlen(dst);
    for (; *s && len < TR_W - 1; s++) dst[len++] = *s;
    dst[len] = 0;
}

void tr_setlast(const char *s) {
    if (trn == 0) return;
    strncpy(tr[trn - 1].text, s, TR_W - 1);
    tr[trn - 1].text[TR_W - 1] = 0;
}

void tr_reset(void) {
    trn = troff = 0;
}
