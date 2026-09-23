#ifndef BPE_H
#define BPE_H

typedef struct {
    char *str;
    int id;
} VocabEntry;

typedef struct {
    VocabEntry *vocab;
    int nvocab;
    char **id2str;
    int nidstr;
    char **ma;
    char **mb;
    int nmerge;
} BPE;

int bpe_init(BPE *b, const char *vocab_path, const char *merges_path);
int bpe_encode(BPE *b, const char *text, int *out, int cap);
int bpe_decode(BPE *b, const int *ids, int n, char *out, int cap);
void bpe_free(BPE *b);

#endif
