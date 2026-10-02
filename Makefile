CC = gcc
CFLAGS = -Wall -Wextra -O0 -g -Iinclude
CFLAGS_REL = -Wall -Wextra -O3 -Iinclude
SRC = src/main.c src/wtxt.c src/math.c src/bpe.c src/model.c src/chat.c src/term.c src/tui_nc.c src/tui_tr.c
TARGET = crucible
LDLIBS = -lm -lncurses -pthread

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDLIBS)

release: $(SRC)
	$(CC) $(CFLAGS_REL) -o crucible-opt $(SRC) $(LDLIBS)

test: tests/test_math tests/test_bpe tests/test_model tests/test_chat tests/test_tui
	tests/test_math
	tests/test_bpe
	tests/test_model
	tests/test_chat
	tests/test_tui

tests/test_math: tests/test_math.c src/math.c
	$(CC) $(CFLAGS) -o tests/test_math tests/test_math.c src/math.c -lm

tests/test_bpe: tests/test_bpe.c src/bpe.c
	$(CC) $(CFLAGS) -o tests/test_bpe tests/test_bpe.c src/bpe.c

tests/test_model: tests/test_model.c src/model.c src/math.c src/wtxt.c
	$(CC) $(CFLAGS) -o tests/test_model tests/test_model.c src/model.c src/math.c src/wtxt.c -lm

tests/test_chat: tests/test_chat.c src/chat.c src/model.c src/math.c src/bpe.c src/wtxt.c
	$(CC) $(CFLAGS) -o tests/test_chat tests/test_chat.c src/chat.c src/model.c src/math.c src/bpe.c src/wtxt.c -lm

tests/test_tui: tests/test_tui.c src/tui_tr.c
	$(CC) $(CFLAGS) -o tests/test_tui tests/test_tui.c src/tui_tr.c

clean:
	rm -f $(TARGET) crucible-opt tests/test_math tests/test_bpe tests/test_model tests/test_chat tests/test_tui

.PHONY: clean test release
