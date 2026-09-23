#ifndef TUI_TR_H
#define TUI_TR_H

#define TR_N 500
#define TR_W 4096

typedef struct {
    char text[TR_W];
    char tag; /* 'u' user, 'a' assistant, 's' status */
} TrMsg;

extern TrMsg tr[TR_N];
extern int trn, troff;

void tr_push(char tag, const char *s);   /* new message */
void tr_append(char tag, const char *s); /* append to last msg if same tag, else new */
void tr_setlast(const char *s);          /* replace last message text */
void tr_reset(void);

#endif
