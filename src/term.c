#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include "term.h"

static struct termios saved;
static int active = 0;

void term_restore(void) {
    if (!active) return;
    active = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &saved);
}

void term_raw(void) {
    struct termios raw;

    if (active) return;
    if (tcgetattr(STDIN_FILENO, &saved) != 0) return;
    raw = saved;
    raw.c_lflag &= ~(ICANON | ECHO | ISIG);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return;
    active = 1;
}

static void on_fatal(int sig) {
    term_restore();
    signal(sig, SIG_DFL);
    raise(sig);
}

static void on_exit_term(void) {
    term_restore();
}

static void on_quit(int sig) {
    term_restore();
    _exit(128 + sig);
}

void term_watch(void) {
    static int armed = 0;
    if (armed) return;
    armed = 1;
    atexit(on_exit_term);
    signal(SIGINT, on_quit);
    signal(SIGTERM, on_quit);
    signal(SIGSEGV, on_fatal);
    signal(SIGABRT, on_fatal);
}

static int read_esc(void) {
    fd_set fds;
    struct timeval tv = {0, 50000};
    unsigned char c;

    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) <= 0) return -1;
    if (read(STDIN_FILENO, &c, 1) != 1 || c != '[') return -1;
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) <= 0) return -1;
    if (read(STDIN_FILENO, &c, 1) != 1) return -1;
    return c;
}

static void redraw(const char *prompt, const char *buf) {
    printf("\r\033[K%s%s", prompt, buf);
    fflush(stdout);
}

int term_getline(const char *prompt, char *buf, int cap, History *h) {
    int len = 0;
    int hist = h ? h->count : 0;

    term_watch();
    term_raw();
    printf("%s", prompt);
    fflush(stdout);

    for (;;) {
        unsigned char c;
        if (read(STDIN_FILENO, &c, 1) != 1) {
            term_restore();
            return -1;
        }
        if (c == '\r' || c == '\n') {
            printf("\r\n");
            break;
        }
        if (c == 3) {
            if (len == 0) {
                term_restore();
                return -1;
            }
            printf("^C\n");
            len = 0;
            buf[0] = 0;
            redraw(prompt, buf);
            continue;
        }
        if (c == 4) {
            if (len == 0) {
                term_restore();
                return -1;
            }
            continue;
        }
        if (c == 127 || c == 8) {
            if (len > 0) {
                buf[--len] = 0;
                redraw(prompt, buf);
            }
            continue;
        }
        if (c == 27) {
            int k = read_esc();
            if (h && (k == 'A' || k == 'B')) {
                if (k == 'A' && hist > 0) hist--;
                if (k == 'B' && hist < h->count) hist++;
                if (hist < h->count) {
                    strncpy(buf, h->lines[hist], cap - 1);
                    buf[cap - 1] = 0;
                } else {
                    buf[0] = 0;
                }
                len = strlen(buf);
                redraw(prompt, buf);
            }
            continue;
        }
        if (c >= 32 && len + 1 < cap) {
            buf[len++] = c;
            buf[len] = 0;
            printf("%c", c);
            fflush(stdout);
        }
    }

    term_restore();
    if (h && len > 0) {
        if (h->count >= 50) {
            memmove(h->lines, h->lines + 1, sizeof(h->lines) - sizeof(h->lines[0]));
            h->count = 49;
        }
        strcpy(h->lines[h->count++], buf);
    }
    return len;
}
