#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "tui.h"
#include "wtxt.h"
#include "bpe.h"
#include "model.h"
#include "chat.h"
#include "tui_tr.h"

#if __has_include(<ncurses.h>)
#include <ncurses.h>
#define HAVE_NCURSES 1
#else
#define HAVE_NCURSES 0
#endif

#if HAVE_NCURSES

static WINDOW *tw, *sw, *iw;
static int th, ww, use_color;
static int done;

static void cleanup(void) {
    if (!done) {
        done = 1;
        endwin();
    }
}

static void on_sig(int s) {
    cleanup();
    signal(s, SIG_DFL);
    raise(s);
}

static void watch(void) {
    static int armed = 0;
    if (armed) return;
    armed = 1;
    atexit(cleanup);
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    signal(SIGSEGV, on_sig);
    signal(SIGABRT, on_sig);
}

static int cpn(const char *s, int n) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80) c++;
    return c;
}

static int msg_rows(const char *s, int w) {
    int rows = 0;
    const char *seg = s;
    for (;;) {
        const char *nl = strchr(seg, '\n');
        int len = nl ? (int)(nl - seg) : (int)strlen(seg);
        int cp = cpn(seg, len);
        rows += cp > 0 ? (cp + w - 1) / w : 1;
        if (!nl) break;
        seg = nl + 1;
    }
    return rows;
}

static void draw_msg(int idx, int w, int skip, int maxr, int *pr) {
    const char *seg = tr[idx].text;
    int row = 0;
    int col = tr[idx].tag == 'u' ? 0 : 0;
    (void)col;
    if (use_color) {
        if (tr[idx].tag == 'u') wattron(tw, COLOR_PAIR(1));
        else if (tr[idx].tag == 'a') wattron(tw, COLOR_PAIR(2));
        else wattron(tw, COLOR_PAIR(3));
    }
    for (;;) {
        const char *nl = strchr(seg, '\n');
        int len = nl ? (int)(nl - seg) : (int)strlen(seg);
        int off = 0;
        while (off < len) {
            int take = len - off;
            if (take > w) {
                take = w;
                while (take > 0 && ((unsigned char)seg[off + take] & 0xC0) == 0x80) take--;
            }
            if (row >= skip && *pr < maxr) {
                char line[TR_W];
                memcpy(line, seg + off, take);
                line[take] = 0;
                wmove(tw, *pr, 0);
                waddstr(tw, line);
                wclrtoeol(tw);
                (*pr)++;
            }
            row++;
            off += take;
        }
        if (!nl) break;
        seg = nl + 1;
    }
    if (use_color) {
        wattroff(tw, COLOR_PAIR(1));
        wattroff(tw, COLOR_PAIR(2));
        wattroff(tw, COLOR_PAIR(3));
    }
}

static void tr_draw(void) {
    int w = ww < TR_W ? ww : TR_W - 1;
    long total = 0;
    int printed = 0;
    long skip, row = 0;

    werase(tw);
    if (w < 1 || th < 1) return;
    for (int i = 0; i < trn; i++) total += msg_rows(tr[i].text, w);
    skip = total - troff - th;
    if (skip < 0) skip = 0;
    for (int i = 0; i < trn && printed < th; i++) {
        int r = msg_rows(tr[i].text, w);
        if (row + r <= skip) {
            row += r;
            continue;
        }
        draw_msg(i, w, (int)(skip - row), th, &printed);
        row += r;
    }
    if (skip > 0) mvwaddstr(tw, 0, w - 9 > 0 ? w - 9 : 0, "-- more --");
    wrefresh(tw);
}

static void st_draw(Chat *c, float temp, int ntok, double secs, int busy) {
    char line[1024];
    werase(sw);
    if (use_color) wbkgd(sw, COLOR_PAIR(4));
    snprintf(line, sizeof(line), " %d/%d tok +%ld drop | temp %g | %s %d tok %.1fs",
        c->len, CHAT_WINDOW, c->dropped, temp, busy ? "gen" : "last", ntok, secs);
    mvwaddnstr(sw, 0, 0, line, ww - 1);
    wrefresh(sw);
}

static void in_draw(const char *buf, int cur) {
    int plen = 5, hcp;
    char head[1100];
    werase(iw);
    mvwaddstr(iw, 0, 0, "you: ");
    if (cur > 0) {
        memcpy(head, buf, cur);
        head[cur] = 0;
        mvwaddstr(iw, 0, plen, head);
    }
    hcp = cpn(buf, cur);
    wmove(iw, 0, plen + hcp);
    wrefresh(iw);
}

static void layout(void) {
    int H, W;
    getmaxyx(stdscr, H, W);
    if (H < 4) H = 4;
    if (W < 8) W = 8;
    th = H - 2;
    ww = W;
    if (tw) delwin(tw);
    if (sw) delwin(sw);
    if (iw) delwin(iw);
    tw = newwin(th, W, 0, 0);
    sw = newwin(1, W, H - 2, 0);
    iw = newwin(1, W, H - 1, 0);
    scrollok(tw, FALSE);
    keypad(iw, TRUE);
}

static void help_box(void) {
    int H, W, h = 14, w = 42;
    WINDOW *hp;
    getmaxyx(stdscr, H, W);
    if (h > H) h = H;
    if (w > W) w = W;
    hp = newwin(h, w, (H - h) / 2, (W - w) / 2);
    box(hp, 0, 0);
    mvwaddstr(hp, 1, 2, "crucible commands");
    mvwaddstr(hp, 3, 2, "/quit         leave");
    mvwaddstr(hp, 4, 2, "/reset        forget chat");
    mvwaddstr(hp, 5, 2, "/temp N       temperature");
    mvwaddstr(hp, 6, 2, "/tokens       context usage");
    mvwaddstr(hp, 7, 2, "/save FILE    save transcript");
    mvwaddstr(hp, 8, 2, "/load FILE    restore transcript");
    mvwaddstr(hp, 10, 2, "PgUp/PgDn     scroll");
    mvwaddstr(hp, 11, 2, "Up/Down       history");
    mvwaddstr(hp, 12, 2, "any key closes");
    wrefresh(hp);
    wgetch(hp);
    delwin(hp);
}

typedef struct {
    int ntok;
    Chat *c;
    float temp;
    clock_t t0;
} StreamCtx;

static void on_piece(const char *p, void *ctx) {
    StreamCtx *s = ctx;
    s->ntok++;
    tr_append('a', p);
    tr_draw();
    st_draw(s->c, s->temp, s->ntok, (double)(clock() - s->t0) / CLOCKS_PER_SEC, 1);
}

static const char *cmds[] = {"/quit", "/reset", "/temp ", "/tokens", "/save ", "/load ", "/help", NULL};

int tui_chat(int max) {
    Chat c;
    char hist[50][1024] = {{0}};
    int hcount = 0, hi = 0;
    char buf[1024] = {0};
    char reply[2048];
    int len = 0, cur = 0;
    float temp = 0.8f;
    int ntok = 0;
    double secs = 0;
    Wtxt w;
    BPE b;
    GPT2 g;
    float *logits;

    if (!isatty(STDIN_FILENO)) {
        fprintf(stderr, "tui needs a tty\n");
        return 1;
    }
    logits = malloc(sizeof(float) * N_VOCAB);
    if (!logits) return 1;
    if (wtxt_load(&w, "models") != 0 || bpe_init(&b, "models/vocab.json", "models/merges.txt") != 0) {
        fprintf(stderr, "load failed\n");
        wtxt_free(&w);
        free(logits);
        return 1;
    }
    wtxt_wire(&g, &w);
    if (chat_init(&c) != 0) {
        fprintf(stderr, "out of memory\n");
        bpe_free(&b);
        wtxt_free(&w);
        free(logits);
        return 1;
    }
    watch();
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    use_color = !getenv("NO_COLOR") && has_colors();
    if (use_color) {
        start_color();
        use_default_colors();
        init_pair(1, COLOR_CYAN, -1);
        init_pair(2, COLOR_GREEN, -1);
        init_pair(3, COLOR_BLACK, -1);
        init_pair(4, COLOR_WHITE, COLOR_BLACK);
    }
    layout();
    tr_push('s', "crucible — /help for commands, /quit to leave");

    for (;;) {
        int k;
        tr_draw();
        st_draw(&c, temp, ntok, secs, 0);
        in_draw(buf, cur);
        k = wgetch(iw);
        if (k == ERR) break;
        if (k == KEY_RESIZE) {
            layout();
            continue;
        }
        if (k == 4) {
            if (len == 0) break;
            len = cur = 0;
            buf[0] = 0;
            continue;
        }
        if (k == '\r' || k == '\n' || k == KEY_ENTER) {
            buf[len] = 0;
            if (len > 0) {
                if (hcount < 50) strcpy(hist[hcount++], buf);
                hi = hcount;
                if (buf[0] == '/') {
                    if (strcmp(buf, "/quit") == 0) break;
                    else if (strcmp(buf, "/reset") == 0) {
                        chat_reset(&c);
                        tr_push('s', "context cleared");
                    } else if (strcmp(buf, "/help") == 0) help_box();
                    else if (strcmp(buf, "/tokens") == 0) {
                        char t[128];
                        snprintf(t, sizeof(t), "%d / %d tokens, %ld dropped", c.len, CHAT_WINDOW, c.dropped);
                        tr_push('s', t);
                    } else if (strncmp(buf, "/temp ", 6) == 0) {
                        char t[64];
                        temp = (float)atof(buf + 6);
                        snprintf(t, sizeof(t), "temp %g", temp);
                        tr_push('s', t);
                    } else if (strncmp(buf, "/save ", 6) == 0) {
                        tr_push('s', chat_save(&c, &b, buf + 6) == 0 ? "saved" : "save failed");
                    } else if (strncmp(buf, "/load ", 6) == 0) {
                        tr_push('s', chat_load(&c, &b, buf + 6) == 0 ? "loaded" : "load failed");
                    } else tr_push('s', "unknown command — /help");
                } else {
                    char msg[1024];
                    StreamCtx sx;
                    clock_t t0;
                    strcpy(msg, buf);
                    tr_push('u', msg);
                    troff = 0;
                    len = cur = 0;
                    buf[0] = 0;
                    t0 = clock();
                    sx.ntok = 0;
                    sx.c = &c;
                    sx.temp = temp;
                    sx.t0 = t0;
                    chat_on_token(on_piece, &sx);
                    if (chat_turn(&c, &g, &b, logits, msg, reply, sizeof(reply), max, temp) != 0) {
                        chat_on_token(NULL, NULL);
                        ntok = sx.ntok;
                        tr_push('s', "turn failed");
                    } else {
                        chat_on_token(NULL, NULL);
                        secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
                        ntok = sx.ntok;
                        if (sx.ntok > 0) tr_setlast(reply[0] ? reply : "(no response)");
                        else tr_push('a', reply[0] ? reply : "(no response)");
                        chat_log(msg, reply);
                    }
                }
            }
            len = cur = 0;
            buf[0] = 0;
            continue;
        }
        if (k == KEY_BACKSPACE || k == 127 || k == 8) {
            if (cur > 0) {
                int cl = 1;
                while (cur - cl > 0 && ((unsigned char)buf[cur - cl] & 0xC0) == 0x80) cl++;
                memmove(buf + cur - cl, buf + cur, len - cur + 1);
                cur -= cl;
                len -= cl;
            }
            continue;
        }
        if (k == KEY_DC) {
            if (cur < len) {
                int cl = 1;
                while (cur + cl < len && ((unsigned char)buf[cur + cl] & 0xC0) == 0x80) cl++;
                memmove(buf + cur, buf + cur + cl, len - cur - cl + 1);
                len -= cl;
            }
            continue;
        }
        if (k == KEY_LEFT) {
            if (cur > 0) {
                cur--;
                while (cur > 0 && ((unsigned char)buf[cur] & 0xC0) == 0x80) cur--;
            }
            continue;
        }
        if (k == KEY_RIGHT) {
            if (cur < len) {
                cur++;
                while (cur < len && ((unsigned char)buf[cur] & 0xC0) == 0x80) cur++;
            }
            continue;
        }
        if (k == KEY_HOME || k == 1) {
            cur = 0;
            continue;
        }
        if (k == KEY_END || k == 5) {
            cur = len;
            continue;
        }
        if (k == 11) {
            buf[cur] = 0;
            len = cur;
            continue;
        }
        if (k == 21) {
            memmove(buf, buf + cur, len - cur + 1);
            len -= cur;
            cur = 0;
            continue;
        }
        if (k == 23) {
            int p = cur;
            while (p > 0 && buf[p - 1] == ' ') p--;
            while (p > 0 && buf[p - 1] != ' ') p--;
            memmove(buf + p, buf + cur, len - cur + 1);
            len -= cur - p;
            cur = p;
            continue;
        }
        if (k == KEY_UP) {
            if (hi > 0) {
                hi--;
                strcpy(buf, hist[hi]);
                len = cur = strlen(buf);
            }
            continue;
        }
        if (k == KEY_DOWN) {
            if (hi < hcount) hi++;
            if (hi < hcount) strcpy(buf, hist[hi]);
            else buf[0] = 0;
            len = cur = strlen(buf);
            continue;
        }
        if (k == KEY_PPAGE) {
            troff += th - 1;
            if (troff > 100000) troff = 100000;
            continue;
        }
        if (k == KEY_NPAGE) {
            troff -= th - 1;
            if (troff < 0) troff = 0;
            continue;
        }
        if (k == '\t' && buf[0] == '/') {
            for (int i = 0; cmds[i]; i++) {
                if (strncmp(buf, cmds[i], strlen(buf)) == 0) {
                    strcpy(buf, cmds[i]);
                    len = cur = strlen(buf);
                    break;
                }
            }
            continue;
        }
        if (k == 18) {
            char q[256] = {0};
            int ql = 0, qk;
            for (;;) {
                char line[280];
                snprintf(line, sizeof(line), "search: %s", q);
                werase(sw);
                mvwaddstr(sw, 0, 0, line);
                wrefresh(sw);
                qk = wgetch(iw);
                if (qk == '\r' || qk == '\n' || qk == KEY_ENTER) break;
                if (qk == 27) {
                    ql = -1;
                    break;
                }
                if ((qk == KEY_BACKSPACE || qk == 127) && ql > 0) q[--ql] = 0;
                else if (qk >= 32 && qk < 127 && ql < 255) {
                    q[ql++] = (char)qk;
                    q[ql] = 0;
                }
            }
            if (ql > 0) {
                for (int i = hcount - 1; i >= 0; i--) {
                    if (strstr(hist[i], q)) {
                        strcpy(buf, hist[i]);
                        len = cur = strlen(buf);
                        break;
                    }
                }
            }
            continue;
        }
        if (k >= 32 && k < 256 && len + 5 < (int)sizeof(buf)) {
            memmove(buf + cur + 1, buf + cur, len - cur + 1);
            buf[cur++] = (char)k;
            len++;
        }
    }

    cleanup();
    chat_free(&c);
    bpe_free(&b);
    wtxt_free(&w);
    free(logits);
    return 0;
}

#else

int tui_chat(int max) {
    (void)max;
    fprintf(stderr, "tui needs ncurses (apt install libncurses-dev)\n");
    return 1;
}

#endif
