#include <stdio.h>
#include <string.h>
#include "bpe.h"

int main(void) {
    BPE b;
    int ids[16];

    if (bpe_init(&b, "models/vocab.json", "models/merges.txt") != 0) {
        printf("FAIL init\n");
        return 1;
    }

    int n = bpe_encode(&b, "Hello, world!", ids, 16);
    int want[4] = {15496, 11, 995, 0};

    if (n != 4) {
        printf("FAIL count %d\n", n);
        bpe_free(&b);
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        if (ids[i] != want[i]) {
            printf("FAIL ids[%d] %d != %d\n", i, ids[i], want[i]);
            bpe_free(&b);
            return 1;
        }
    }

    n = bpe_encode(&b, "Hello", ids, 16);
    if (n != 1 || ids[0] != 15496) {
        printf("FAIL hello %d %d\n", n, ids[0]);
        bpe_free(&b);
        return 1;
    }

    /* ground truth from HF GPT2Tokenizer; guards the pretokenizer rules */
    struct {
        const char *text;
        int ids[6];
        int n;
    } cases[] = {
        {"a  b", {64, 220, 275}, 3},
        {"a   b", {64, 220, 220, 275}, 4},
        {"a , b", {64, 837, 275}, 3},
        {"don't", {9099, 470}, 2},
        {"it's fine", {270, 338, 3734}, 3},
        {" $19.99", {720, 1129, 13, 2079}, 4},
    };
    for (int c = 0; c < 6; c++) {
        n = bpe_encode(&b, cases[c].text, ids, 16);
        if (n != cases[c].n) {
            printf("FAIL count %d != %d [%s]\n", n, cases[c].n, cases[c].text);
            bpe_free(&b);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (ids[i] != cases[c].ids[i]) {
                printf("FAIL ids [%s][%d] %d != %d\n", cases[c].text, i, ids[i], cases[c].ids[i]);
                bpe_free(&b);
                return 1;
            }
        }
    }

    const char *sentences[] = {
        "Hello, world!",
        "It costs $19.99.",
        "two  spaces",
        "12345",
        "caf\u00e9, na\u00efve.",
    };
    char buf[256];
    for (int s = 0; s < 5; s++) {
        n = bpe_encode(&b, sentences[s], ids, 16);
        if (n < 0) {
            printf("FAIL encode %d\n", s);
            bpe_free(&b);
            return 1;
        }
        if (bpe_decode(&b, ids, n, buf, sizeof(buf)) < 0 || strcmp(buf, sentences[s]) != 0) {
            printf("FAIL roundtrip %d: [%s]\n", s, buf);
            bpe_free(&b);
            return 1;
        }
    }

    bpe_free(&b);
    printf("all pass\n");
    return 0;
}
