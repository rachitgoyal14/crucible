#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe.h"

static char *read_file(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (!buf || fread(buf, 1, n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[n] = 0;
    fclose(f);
    if (len) *len = n;
    return buf;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int utf8len(const char *s, unsigned *cp) {
    unsigned char c = s[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0) {
        *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        *cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    *cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    return 4;
}

static int utf8enc(unsigned cp, char *out) {
    if (cp < 0x80) {
        out[0] = cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = 0xC0 | (cp >> 6);
        out[1] = 0x80 | (cp & 0x3F);
        return 2;
    }
    out[0] = 0xE0 | (cp >> 12);
    out[1] = 0x80 | ((cp >> 6) & 0x3F);
    out[2] = 0x80 | (cp & 0x3F);
    return 3;
}

static const char *parse_str(const char *p, char **out) {
    int cap = 64, len = 0;
    char *s = malloc(cap);
    if (!s) return NULL;
    p++;
    for (;;) {
        char c = *p++;
        if (c == '"') break;
        if (c == 0) {
            free(s);
            return NULL;
        }
        unsigned cp;
        if (c == '\\') {
            char e = *p++;
            if (e == '"' || e == '\\' || e == '/') {
                cp = e;
            } else if (e == 'b') {
                cp = '\b';
            } else if (e == 'f') {
                cp = '\f';
            } else if (e == 'n') {
                cp = '\n';
            } else if (e == 'r') {
                cp = '\r';
            } else if (e == 't') {
                cp = '\t';
            } else if (e == 'u') {
                cp = 0;
                for (int i = 0; i < 4; i++) {
                    int h = hexval(*p++);
                    if (h < 0) {
                        free(s);
                        return NULL;
                    }
                    cp = (cp << 4) | h;
                }
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (p[0] == '\\' && p[1] == 'u') {
                        unsigned lo = 0;
                        p += 2;
                        for (int i = 0; i < 4; i++) {
                            int h = hexval(*p++);
                            if (h < 0) {
                                free(s);
                                return NULL;
                            }
                            lo = (lo << 4) | h;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                }
            } else {
                free(s);
                return NULL;
            }
        } else {
            p--;
            int n = utf8len(p, &cp);
            p += n;
        }
        char tmp[4];
        int n = utf8enc(cp, tmp);
        while (len + n >= cap) {
            cap *= 2;
            char *ns = realloc(s, cap);
            if (!ns) {
                free(s);
                return NULL;
            }
            s = ns;
        }
        memcpy(s + len, tmp, n);
        len += n;
    }
    s[len] = 0;
    *out = s;
    return p;
}

static int cmp_entry(const void *a, const void *b) {
    return strcmp(((const VocabEntry *)a)->str, ((const VocabEntry *)b)->str);
}

static int load_vocab(BPE *b, const char *path) {
    char *js = read_file(path, NULL);
    if (!js) return -1;
    const char *p = js;
    while (*p && *p != '{') p++;
    if (*p) p++;

    int cap = 1 << 16, n = 0;
    VocabEntry *v = malloc(sizeof(VocabEntry) * cap);
    if (!v) {
        free(js);
        return -1;
    }
    int maxid = -1;
    for (;;) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == '}' || *p == 0) break;
        if (*p != '"') goto fail;
        char *key = NULL;
        p = parse_str(p, &key);
        if (!p) goto fail;
        while (*p == ' ') p++;
        if (*p != ':') {
            free(key);
            goto fail;
        }
        p++;
        while (*p == ' ') p++;
        int id = (int)strtol(p, (char **)&p, 10);
        if (n >= cap) goto fail;
        v[n].str = key;
        v[n].id = id;
        if (id > maxid) maxid = id;
        n++;
    }
    free(js);

    qsort(v, n, sizeof(VocabEntry), cmp_entry);
    b->vocab = v;
    b->nvocab = n;
    b->id2str = calloc(maxid + 1, sizeof(char *));
    if (!b->id2str) return -1;
    b->nidstr = maxid + 1;
    for (int i = 0; i < n; i++) {
        b->id2str[v[i].id] = v[i].str;
    }
    return 0;

fail:
    for (int i = 0; i < n; i++) free(v[i].str);
    free(v);
    free(js);
    return -1;
}

static int load_merges(BPE *b, const char *path) {
    char *tx = read_file(path, NULL);
    if (!tx) return -1;
    int cap = 1 << 16, n = 0;
    char **ma = malloc(sizeof(char *) * cap);
    char **mb = malloc(sizeof(char *) * cap);
    if (!ma || !mb) {
        free(ma);
        free(mb);
        free(tx);
        return -1;
    }
    char *line = tx;
    int first = 1;
    for (char *p = tx; ; p++) {
        if (*p != '\n' && *p != 0) continue;
        char save = *p;
        *p = 0;
        if (line[0] == '#' && first) {
            /* version header, not a merge */
        } else if (*p == 0 && line[0] == 0) {
            /* trailing blank line */
        } else {
            char *sp = strchr(line, ' ');
            if (!sp) {
                free(tx);
                return -1;
            }
            *sp = 0;
            char *rest = sp + 1;
            size_t rl = strlen(rest);
            if (rl > 0 && rest[rl - 1] == '\r') rest[rl - 1] = 0;
            ma[n] = strdup(line);
            mb[n] = strdup(rest);
            if (!ma[n] || !mb[n]) {
                free(tx);
                return -1;
            }
            n++;
        }
        first = 0;
        if (save == 0) break;
        line = p + 1;
    }
    free(tx);
    b->ma = ma;
    b->mb = mb;
    b->nmerge = n;
    return 0;
}

static int rank(BPE *b, const char *a, const char *c) {
    for (int i = 0; i < b->nmerge; i++) {
        if (strcmp(b->ma[i], a) == 0 && strcmp(b->mb[i], c) == 0) {
            return i;
        }
    }
    return INT_MAX;
}

static int vocab_id(BPE *b, const char *s) {
    VocabEntry q = { .str = (char *)s, .id = 0 };
    VocabEntry *f = bsearch(&q, b->vocab, b->nvocab, sizeof(VocabEntry), cmp_entry);
    return f ? f->id : -1;
}

int bpe_init(BPE *b, const char *vocab_path, const char *merges_path) {
    memset(b, 0, sizeof(*b));
    if (load_vocab(b, vocab_path) != 0) return -1;
    if (load_merges(b, merges_path) != 0) {
        bpe_free(b);
        return -1;
    }
    return 0;
}

void bpe_free(BPE *b) {
    for (int i = 0; i < b->nvocab; i++) free(b->vocab[i].str);
    free(b->vocab);
    free(b->id2str);
    for (int i = 0; i < b->nmerge; i++) {
        free(b->ma[i]);
        free(b->mb[i]);
    }
    free(b->ma);
    free(b->mb);
    memset(b, 0, sizeof(*b));
}

static int bmap[256];
static int bmap_done = 0;

static void bmap_init(void) {
    if (bmap_done) return;
    int n = 0;
    for (int i = 0; i < 256; i++) {
        if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255)) {
            bmap[i] = i;
        } else {
            bmap[i] = 256 + n++;
        }
    }
    bmap_done = 1;
}

static int is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static int is_letter(char c) {
    unsigned char u = c;
    return (u >= 'A' && u <= 'Z') || (u >= 'a' && u <= 'z') || u >= 0x80;
}

static int is_digit(char c) {
    return c >= '0' && c <= '9';
}

static int match_contraction(const char *s, int *len) {
    static const char *forms[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d", NULL};
    for (int i = 0; forms[i]; i++) {
        int n = strlen(forms[i]);
        if (strncmp(s, forms[i], n) == 0) {
            *len = n;
            return 1;
        }
    }
    return 0;
}

static int next_piece(const char *s, int *len) {
    int n;
    if (match_contraction(s, &n)) {
        *len = n;
        return 1;
    }
    int i = 0;
    if (s[i] == ' ') i++;
    if (is_letter(s[i])) {
        do {
            i++;
        } while (is_letter(s[i]));
        *len = i;
        return 1;
    }
    i = (s[0] == ' ');
    if (is_digit(s[i])) {
        do {
            i++;
        } while (is_digit(s[i]));
        *len = i;
        return 1;
    }
    if (s[0] == ' ' && s[1] && !is_space(s[1]) && !is_letter(s[1]) && !is_digit(s[1])) {
        i = 2;
        while (s[i] && !is_space(s[i]) && !is_letter(s[i]) && !is_digit(s[i])) i++;
        *len = i;
        return 1;
    }
    if (!is_space(s[0]) && !is_letter(s[0]) && !is_digit(s[0])) {
        i = 0;
        while (s[i] && !is_space(s[i]) && !is_letter(s[i]) && !is_digit(s[i])) i++;
        *len = i;
        return 1;
    }
    /* whitespace run; give one back when text follows, like the regex */
    i = 0;
    while (is_space(s[i])) i++;
    if (s[i] != 0 && i > 1) i--;
    *len = i;
    return 1;
}

static int bpe_piece(BPE *b, const unsigned char *p, int len, int *out, int cap) {
    int nsym = 0;
    char **sym = malloc(sizeof(char *) * (len + 1));
    if (!sym) return -1;
    for (int i = 0; i < len; i++) {
        char tmp[4];
        int n = utf8enc(bmap[p[i]], tmp);
        sym[nsym] = malloc(n + 1);
        if (!sym[nsym]) goto fail;
        memcpy(sym[nsym], tmp, n);
        sym[nsym][n] = 0;
        nsym++;
    }

    for (;;) {
        int best = INT_MAX, at = -1;
        for (int i = 0; i + 1 < nsym; i++) {
            int r = rank(b, sym[i], sym[i + 1]);
            if (r < best) {
                best = r;
                at = i;
            }
        }
        if (at < 0) break;
        int nl = strlen(sym[at]) + strlen(sym[at + 1]) + 1;
        char *joined = malloc(nl);
        if (!joined) goto fail;
        strcpy(joined, sym[at]);
        strcat(joined, sym[at + 1]);
        free(sym[at]);
        free(sym[at + 1]);
        sym[at] = joined;
        for (int i = at + 1; i + 1 < nsym; i++) {
            sym[i] = sym[i + 1];
        }
        nsym--;
        void *ns = realloc(sym, sizeof(char *) * (nsym + 1));
        if (ns) sym = ns;
    }

    int count = 0;
    for (int i = 0; i < nsym; i++) {
        int id = vocab_id(b, sym[i]);
        free(sym[i]);
        if (id < 0 || count >= cap) {
            for (int j = i + 1; j < nsym; j++) free(sym[j]);
            free(sym);
            return -1;
        }
        out[count++] = id;
    }
    free(sym);
    return count;

fail:
    for (int i = 0; i < nsym; i++) free(sym[i]);
    free(sym);
    return -1;
}

int bpe_encode(BPE *b, const char *text, int *out, int cap) {
    bmap_init();
    int count = 0;
    const char *s = text;
    while (*s) {
        int len = 0;
        next_piece(s, &len);
        if (len <= 0) return -1;
        int left = cap - count;
        int n = bpe_piece(b, (const unsigned char *)s, len, out + count, left);
        if (n < 0) return -1;
        count += n;
        s += len;
    }
    return count;
}

int bpe_decode(BPE *b, const int *ids, int n, char *out, int cap) {
    static int revmap[324];
    static int revmap_done = 0;

    bmap_init();
    if (!revmap_done) {
        for (int i = 0; i < 324; i++) revmap[i] = -1;
        for (int i = 0; i < 256; i++) revmap[bmap[i]] = i;
        revmap_done = 1;
    }

    int len = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= b->nidstr || !b->id2str[ids[i]]) return -1;
        const char *s = b->id2str[ids[i]];
        while (*s) {
            unsigned cp = 0;
            s += utf8len(s, &cp);
            if (cp >= 324 || revmap[cp] < 0 || len >= cap - 1) return -1;
            out[len++] = revmap[cp];
        }
    }
    out[len] = 0;
    return len;
}
