#include <locale.h>
#include <pthread.h>
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

static WINDOW *hw, *tw, *sw, *iw;
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

/* wrap s into out (max TR_W each), breaking at spaces; returns line count */
static int wrap(const char *s, int w, char out[][512], int maxl) {
    int n = 0, len = strlen(s);
    int start = 0;
    if (w < 8) w = 8;
    if (w > 511) w = 511;
    while (start < len && n < maxl) {
        int take = len - start;
        if (take > w) {
            take = w;
            int sp = -1;
            for (int i = take; i > 0; i--) {
                if (s[start + i] == ' ') {
                    sp = i;
                    break;
                }
            }
            if (sp > 0) take = sp;
            while (take > 0 && ((unsigned char)s[start + take] & 0xC0) == 0x80) take--;
            if (take <= 0) take = 1;
        }
        memcpy(out[n], s + start, take);
        out[n][take] = 0;
        n++;
        start += take;
        while (s[start] == ' ') start++;
    }
    if (n == 0 && n < maxl) {
        out[n][0] = 0;
        n++;
    }
    return n;
}

static int colw(void) {
    int w = (ww - 2) / 2 - 1;
    return w < 8 ? 8 : w;
}

static int msg_rows(int idx) {
    char lines[512][512];
    int w = tr[idx].tag == 's' ? ww - 2 : colw();
    int n = wrap(tr[idx].text, w, lines, 512);
    return 1 + n;
}

static void putline(int y, int x, int w, const char *s) {
    char line[512];
    int n = strlen(s);
    if (n > w) n = w;
    memcpy(line, s, n);
    line[n] = 0;
    mvwaddstr(tw, y + 1, x, line);
    wclrtoeol(tw);
}

static void draw_msg(int idx, int skip, int maxr, int *pr) {
    char lines[512][512];
    int w = ww - 2, x = 1;
    int row = 0;

    if (tr[idx].tag == 'u') {
        w = colw();
        x = 1;
    } else if (tr[idx].tag == 'a') {
        w = colw();
        x = ww / 2 + 1;
    }
    int n = wrap(tr[idx].text, w, lines, 512);

    if (tr[idx].tag == 's') {
    if (use_color) wattron(tw, COLOR_PAIR(3));
    if (row >= skip) {
        if (*pr < maxr) {
            int len = strlen(tr[idx].text);
            int cx = (ww - len) / 2;
            if (cx < 1) cx = 1;
            mvwaddstr(tw, *pr + 1, cx, tr[idx].text);
            wclrtoeol(tw);
            (*pr)++;
        }
    }
    row++;
    if (use_color) wattroff(tw, COLOR_PAIR(3));
    return;
    }
    if (use_color) wattron(tw, COLOR_PAIR(tr[idx].tag == 'u' ? 1 : 2));
    if (use_color) wattron(tw, A_BOLD);
    if (row >= skip) {
        if (*pr < maxr) {
            mvwaddstr(tw, *pr + 1, x, tr[idx].tag == 'u' ? "you" : "ai");
            wclrtoeol(tw);
            (*pr)++;
        }
    }
    row++;
    if (use_color) wattroff(tw, A_BOLD);
    if (use_color) wattroff(tw, COLOR_PAIR(1));
    if (use_color) wattroff(tw, COLOR_PAIR(2));
    for (int i = 0; i < n; i++) {
        if (row >= skip) {
            if (*pr < maxr) {
                putline(*pr, x, w, lines[i]);
                (*pr)++;
            }
        }
        row++;
    }
}

static void hd_draw(void) {
    werase(hw);
    if (use_color) {
        wattron(hw, COLOR_PAIR(5));
        wattron(hw, A_BOLD);
    }
    mvwaddstr(hw, 0, 1, "crucible");
    if (use_color) {
        wattroff(hw, A_BOLD);
        wattroff(hw, COLOR_PAIR(5));
    }
    mvwaddstr(hw, 0, 11, "GPT-2 124M - pure C");
    char r[32];
    snprintf(r, sizeof(r), "%d cols", ww);
    mvwaddstr(hw, 0, ww - (int)strlen(r) - 1, r);
    wrefresh(hw);
}

static void tr_draw(void) {
    long total = 0;
    int printed = 0;
    long skip, row = 0;
    int maxr;

    werase(tw);
    if (use_color) {
        wattron(tw, COLOR_PAIR(7));
        box(tw, 0, 0);
        wattroff(tw, COLOR_PAIR(7));
    } else {
        box(tw, 0, 0);
    }
    maxr = th - 2;
    if (ww < 10 || maxr < 1) {
        wrefresh(tw);
        return;
    }
    for (int i = 0; i < trn; i++) total += msg_rows(i);
    skip = total - troff - maxr;
    if (skip < 0) skip = 0;
    for (int i = 0; i < trn && printed < maxr; i++) {
        int r = msg_rows(i);
        if (row + r <= skip) {
            row += r;
            continue;
        }
        draw_msg(i, (int)(skip - row), maxr, &printed);
        row += r;
    }
    if (total > maxr) {
        char cnt[32];
        snprintf(cnt, sizeof(cnt), "[%ld/%ld]", total - troff, total);
        mvwaddstr(tw, th - 2, ww - (int)strlen(cnt) - 2, cnt);
    }
    wrefresh(tw);
}

static void st_draw(Chat *c, float temp, int ntok, double secs, int busy) {
    static const char frames[] = "|/-\\";
    static int fi;
    char line[1024];
    werase(sw);
    if (use_color) wbkgd(sw, COLOR_PAIR(4));
    if (busy) {
        snprintf(line, sizeof(line), " [%c] gen %d tok | %d/%d tok +%ld drop | temp %g",
            frames[fi++ % 4], ntok, c->len, CHAT_WINDOW, c->dropped, temp);
    } else {
        snprintf(line, sizeof(line), " %d/%d tok +%ld drop | temp %g | last %d tok %.1fs",
            c->len, CHAT_WINDOW, c->dropped, temp, ntok, secs);
    }
    mvwaddnstr(sw, 0, 0, line, ww - 1);
    wrefresh(sw);
}

static void in_draw(const char *buf, int cur) {
    int w = ww - 6, start = 0;
    char vis[1100];

    werase(iw);
    if (use_color) {
        wattron(iw, COLOR_PAIR(7));
        box(iw, 0, 0);
        wattroff(iw, COLOR_PAIR(7));
    } else {
        box(iw, 0, 0);
    }
    if (use_color) {
        wattron(iw, COLOR_PAIR(6));
        wattron(iw, A_BOLD);
    }
    mvwaddstr(iw, 1, 2, ">");
    if (use_color) {
        wattroff(iw, A_BOLD);
        wattroff(iw, COLOR_PAIR(6));
    }
    if (w < 8) {
        wrefresh(iw);
        return;
    }
    if (cur - start >= w) start = cur - w + 1;
    if (cur < start) start = cur;
    if (buf[0] == 0) {
        if (use_color) wattron(iw, COLOR_PAIR(3));
        mvwaddstr(iw, 1, 4, "Type a message...  (/help)");
        if (use_color) wattroff(iw, COLOR_PAIR(3));
    } else {
        int n = strlen(buf) - start;
        if (n > w) n = w;
        memcpy(vis, buf + start, n);
        vis[n] = 0;
        mvwaddstr(iw, 1, 4, vis);
    }
    wmove(iw, 1, 4 + cpn(buf + start, cur - start));
    wrefresh(iw);
}

static void in_busy(void) {
    static const char frames[] = "|/-\\";
    static int fi;
    char line[64];

    werase(iw);
    if (use_color) {
        wattron(iw, COLOR_PAIR(7));
        box(iw, 0, 0);
        wattroff(iw, COLOR_PAIR(7));
    } else {
        box(iw, 0, 0);
    }
    snprintf(line, sizeof(line), "[%c] responding...", frames[fi++ % 4]);
    if (use_color) wattron(iw, COLOR_PAIR(3));
    mvwaddstr(iw, 1, 2, line);
    if (use_color) wattroff(iw, COLOR_PAIR(3));
    wrefresh(iw);
}

static void layout(void) {
    int H, W;
    getmaxyx(stdscr, H, W);
    if (H < 8) H = 8;
    if (W < 20) W = 20;
    th = H - 5;
    ww = W;
    if (hw) delwin(hw);
    if (tw) delwin(tw);
    if (sw) delwin(sw);
    if (iw) delwin(iw);
    hw = newwin(1, W, 0, 0);
    tw = newwin(th, W, 1, 0);
    sw = newwin(1, W, H - 4, 0);
    iw = newwin(3, W, H - 3, 0);
    scrollok(tw, FALSE);
    keypad(iw, TRUE);
}

static void help_box(void) {
    int H, W, h = 18, w = 48;
    WINDOW *hp;
    getmaxyx(stdscr, H, W);
    if (h > H) h = H;
    if (w > W) w = W;
    hp = newwin(h, w, (H - h) / 2, (W - w) / 2);
    if (!hp) return;
    if (use_color) wattron(hp, COLOR_PAIR(7));
    box(hp, 0, 0);
    if (use_color) wattroff(hp, COLOR_PAIR(7));
    if (use_color) {
        wattron(hp, COLOR_PAIR(5));
        wattron(hp, A_BOLD);
    }
    mvwaddstr(hp, 1, 2, "crucible commands");
    if (use_color) {
        wattroff(hp, A_BOLD);
        wattroff(hp, COLOR_PAIR(5));
    }
    mvwaddstr(hp, 3, 2, "/quit         leave");
    mvwaddstr(hp, 4, 2, "/reset        forget chat");
    mvwaddstr(hp, 5, 2, "/temp N       temperature");
    mvwaddstr(hp, 6, 2, "/tokens       context usage");
    mvwaddstr(hp, 7, 2, "/save FILE    save transcript");
    mvwaddstr(hp, 8, 2, "/load FILE    restore transcript");
    mvwaddstr(hp, 10, 2, "PgUp/PgDn     scroll transcript");
    mvwaddstr(hp, 11, 2, "Up/Down       input history");
    mvwaddstr(hp, 12, 2, "Tab           complete /command");
    mvwaddstr(hp, 13, 2, "Ctrl-R        search history");
    mvwaddstr(hp, 15, 2, "any key closes");
    wrefresh(hp);
    wgetch(hp);
    delwin(hp);
}

typedef struct {
    Chat *c;
    GPT2 *g;
    BPE *b;
    float *logits;
    char msg[1024];
    char reply[2048];
    int max;
    float temp;
    int rc;
    int ntok;
    double secs;
    int active;
    pthread_mutex_t mu;
    char pend[8192];
    int pendlen;
} GenTask;

static void piece_cb(const char *p, void *ctx) {
    GenTask *t = ctx;
    pthread_mutex_lock(&t->mu);
    int n = strlen(p);
    if (t->pendlen + n < (int)sizeof(t->pend) - 1) {
        memcpy(t->pend + t->pendlen, p, n);
        t->pendlen += n;
        t->pend[t->pendlen] = 0;
    }
    t->ntok++;
    pthread_mutex_unlock(&t->mu);
}

static void *gen_run(void *arg) {
    GenTask *t = arg;
    clock_t t0 = clock();
    chat_on_token(piece_cb, t);
    t->rc = chat_turn(t->c, t->g, t->b, t->logits, t->msg, t->reply, sizeof(t->reply), t->max, t->temp);
    chat_on_token(NULL, NULL);
    pthread_mutex_lock(&t->mu);
    t->secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    t->active = 0;
    pthread_mutex_unlock(&t->mu);
    return NULL;
}

/* pump streamed pieces into history; returns worker ntok */
static int drain(GenTask *t) {
    char chunk[8192];
    int n = 0, ntok;
    pthread_mutex_lock(&t->mu);
    if (t->pendlen > 0) {
        memcpy(chunk, t->pend, t->pendlen);
        n = t->pendlen;
        chunk[n] = 0;
        t->pendlen = 0;
        t->pend[0] = 0;
    }
    ntok = t->ntok;
    pthread_mutex_unlock(&t->mu);
    if (n > 0) tr_append('a', chunk);
    return ntok;
}

static int gen_done(GenTask *t) {
    int d;
    pthread_mutex_lock(&t->mu);
    d = !t->active;
    pthread_mutex_unlock(&t->mu);
    return d;
}

static const char *cmds[] = {"/quit", "/reset", "/temp ", "/tokens", "/save ", "/load ", "/help", NULL};

int tui_chat(int max) {
    Chat c;
    char hist[50][1024] = {{0}};
    int hcount = 0, hi = 0;
    char buf[1024] = {0};
    int len = 0, cur = 0;
    float temp = 0.4f;
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
    setlocale(LC_ALL, "");
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
        init_pair(3, COLOR_YELLOW, -1);
        init_pair(4, COLOR_WHITE, COLOR_BLACK);
        init_pair(5, COLOR_YELLOW, -1);
        init_pair(6, COLOR_MAGENTA, -1);
        init_pair(7, COLOR_BLUE, -1);
    }
    layout();
    tr_push('s', ".- crucible - GPT-2 124M in pure C -.");
    tr_push('s', "say hi, or /help for commands, /quit to leave");

    for (;;) {
        int k;
        hd_draw();
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
                    GenTask task;
                    pthread_t gen;
                    clock_t t0;
                    int leave = 0;

                    strcpy(task.msg, buf);
                    task.c = &c;
                    task.g = &g;
                    task.b = &b;
                    task.logits = logits;
                    task.max = max;
                    task.temp = temp;
                    task.rc = 0;
                    task.ntok = 0;
                    task.secs = 0;
                    task.active = 1;
                    task.pendlen = 0;
                    task.pend[0] = 0;
                    pthread_mutex_init(&task.mu, NULL);
                    tr_push('u', task.msg);
                    troff = 0;
                    len = cur = 0;
                    buf[0] = 0;
                    tr_push('a', "");
                    tr_draw();
                    st_draw(&c, temp, 0, 0, 1);
                    in_draw(buf, cur);
                    if (pthread_create(&gen, NULL, gen_run, &task) != 0) {
                        pthread_mutex_destroy(&task.mu);
                        tr_push('s', "turn failed");
                        continue;
                    }
                    t0 = clock();
                    wtimeout(iw, 100);
                    for (;;) {
                        int k2, ntk;
                        ntk = drain(&task);
                        tr_draw();
                        st_draw(&c, temp, ntk, (double)(clock() - t0) / CLOCKS_PER_SEC, 1);
                        in_busy();
                        if (gen_done(&task)) break;
                        k2 = wgetch(iw);
                        if (k2 == KEY_RESIZE) {
                            layout();
                        } else if (k2 == KEY_PPAGE) {
                            troff += th - 1;
                            if (troff > 100000) troff = 100000;
                        } else if (k2 == KEY_NPAGE) {
                            troff -= th - 1;
                            if (troff < 0) troff = 0;
                        } else if (k2 == 4 && len == 0) {
                            leave = 1;
                        }
                    }
                    wtimeout(iw, -1);
                    pthread_join(gen, NULL);
                    pthread_mutex_destroy(&task.mu);
                    ntok = task.ntok;
                    secs = task.secs;
                    if (leave) break;
                    if (task.rc != 0) {
                        tr_push('s', "turn failed");
                    } else {
                        tr_setlast(task.reply[0] ? task.reply : "(no response)");
                        chat_log(task.msg, task.reply);
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
