#include <stdio.h>
#include <string.h>
#include "chat.h"

static int test_window(void) {
    Chat c;
    int ids[CHAT_WINDOW + 10];

    if (chat_init(&c) != 0) {
        printf("FAIL init\n");
        return 1;
    }
    for (int i = 0; i < CHAT_WINDOW + 10; i++) ids[i] = i;
    if (chat_add(&c, ids, CHAT_WINDOW + 10) != 0) {
        printf("FAIL add\n");
        return 1;
    }
    if (c.len != CHAT_WINDOW || c.ids[0] != 10 || c.ids[CHAT_WINDOW - 1] != CHAT_WINDOW + 9) {
        printf("FAIL window %d %d %d\n", c.len, c.ids[0], c.ids[CHAT_WINDOW - 1]);
        return 1;
    }
    chat_free(&c);

    char s[] = "hello\nUser: impostor";
    strip_echo(s);
    if (strcmp(s, "hello") != 0) {
        printf("FAIL strip [%s]\n", s);
        return 1;
    }
    char t[] = "no echo here";
    strip_echo(t);
    if (strcmp(t, "no echo here") != 0) {
        printf("FAIL strip clean\n");
        return 1;
    }

    /* fabricated assistant turns get cut, genuine newlines survive */
    char a[] = "answer\nAssistant: fake turn\nAssistant: more";
    strip_echo(a);
    if (strcmp(a, "answer") != 0) {
        printf("FAIL strip assistant [%s]\n", a);
        return 1;
    }
    char b[] = "answer\nUser: fake\nAssistant: also fake";
    strip_echo(b);
    if (strcmp(b, "answer") != 0) {
        printf("FAIL strip user first [%s]\n", b);
        return 1;
    }
    char m[] = "para one\npara two";
    strip_echo(m);
    if (strcmp(m, "para one\npara two") != 0) {
        printf("FAIL strip multiline [%s]\n", m);
        return 1;
    }

    int l1[] = {7, 7, 7, 7};
    int l2[] = {1, 2, 1, 2, 1, 2, 1, 2};
    int l3[] = {1, 2, 3, 4, 5};
    int l4[] = {9, 9};
    int l5[] = {5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7};
    if (!looping(l1, 4) || !looping(l2, 8) || !looping(l5, 12)) {
        printf("FAIL loop miss\n");
        return 1;
    }
    if (looping(l3, 5) || looping(l4, 2)) {
        printf("FAIL loop false alarm\n");
        return 1;
    }

    return 0;
}

static int test_persist(void) {
    BPE b;
    Chat c;
    int ids[16];
    int n;

    if (bpe_init(&b, "models/vocab.json", "models/merges.txt") != 0) {
        printf("FAIL bpe init\n");
        return 1;
    }
    if (chat_init(&c) != 0) {
        printf("FAIL init\n");
        bpe_free(&b);
        return 1;
    }
    n = bpe_encode(&b, "hi there", ids, 16);
    if (n < 0 || chat_add(&c, ids, n) != 0) {
        printf("FAIL add\n");
        return 1;
    }
    if (chat_save(&c, &b, "tests/.tmp-save.txt") != 0) {
        printf("FAIL save\n");
        return 1;
    }
    chat_reset(&c);
    if (c.len != 0) {
        printf("FAIL reset\n");
        return 1;
    }
    if (chat_load(&c, &b, "tests/.tmp-save.txt") != 0) {
        printf("FAIL load\n");
        return 1;
    }
    if (c.len != n) {
        printf("FAIL len %d != %d\n", c.len, n);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (c.ids[i] != ids[i]) {
            printf("FAIL ids[%d]\n", i);
            return 1;
        }
    }
    if (chat_load(&c, &b, "tests/.tmp-nope.txt") == 0) {
        printf("FAIL missing file accepted\n");
        return 1;
    }
    remove("tests/.tmp-save.txt");
    chat_free(&c);
    bpe_free(&b);
    return 0;
}

int main(void) {
    if (test_window() != 0) return 1;
    if (test_persist() != 0) return 1;
    printf("all pass\n");
    return 0;
}
