#ifndef TERM_H
#define TERM_H

typedef struct {
    char lines[50][1024];
    int count;
} History;

void term_raw(void);
void term_restore(void);
int term_getline(const char *prompt, char *buf, int cap, History *h);

#endif
