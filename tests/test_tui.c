#include <stdio.h>
#include <string.h>
#include "tui_tr.h"

int main(void) {
    tr_reset();
    tr_push('u', "hello");
    if (trn != 1 || tr[0].tag != 'u' || strcmp(tr[0].text, "hello") != 0) {
        printf("FAIL basic\n");
        return 1;
    }

    /* streamed pieces must append into one assistant message */
    tr_append('a', "the quick");
    if (trn != 2 || tr[1].tag != 'a' || strcmp(tr[1].text, "the quick") != 0) {
        printf("FAIL stream first\n");
        return 1;
    }
    tr_append('a', " fox");
    if (trn != 2 || strcmp(tr[1].text, "the quick fox") != 0) {
        printf("FAIL stream append\n");
        return 1;
    }
    tr_append('a', "tail\nline two");
    if (trn != 2 || strcmp(tr[1].text, "the quick foxtail\nline two") != 0) {
        printf("FAIL stream newline\n");
        return 1;
    }

    tr_setlast("final reply");
    if (trn != 2 || strcmp(tr[1].text, "final reply") != 0) {
        printf("FAIL setlast\n");
        return 1;
    }

    /* a different tag starts a new message */
    tr_append('u', "next");
    if (trn != 3 || tr[2].tag != 'u' || strcmp(tr[2].text, "next") != 0) {
        printf("FAIL new tag\n");
        return 1;
    }

    char big[TR_W + 100];
    memset(big, 'x', sizeof(big) - 1);
    big[sizeof(big) - 1] = 0;
    tr_push('s', big);
    if (strlen(tr[3].text) != TR_W - 1) {
        printf("FAIL chop %lu\n", strlen(tr[3].text));
        return 1;
    }

    troff = 10;
    for (int i = 0; i < TR_N + 6; i++) tr_push('s', "filler");
    if (trn != TR_N || troff != 0 || strcmp(tr[TR_N - 1].text, "filler") != 0) {
        printf("FAIL scroll %d %d\n", trn, troff);
        return 1;
    }

    tr_reset();
    if (trn != 0 || troff != 0) {
        printf("FAIL reset\n");
        return 1;
    }

    /* append on an empty ring starts the message */
    tr_append('a', "first");
    if (trn != 1 || strcmp(tr[0].text, "first") != 0) {
        printf("FAIL append empty\n");
        return 1;
    }

    printf("all pass\n");
    return 0;
}
