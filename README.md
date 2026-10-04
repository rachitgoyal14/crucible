<div align="center">
  <img src="assets/logo.svg" width="150" alt="crucible logo">

# crucible

**GPT-2 (124M), melted down to C you can actually read.**

Tokenizer, weights loader, attention, sampling, ncurses chat —
no PyTorch, no Python at runtime, nothing you can't open and read.

[![language](https://img.shields.io/badge/language-C-00599C?logo=c&logoColor=white)](src)
[![runtime deps](https://img.shields.io/badge/runtime%20deps-0-2ea44f)](#quick-start)
[![tests](https://img.shields.io/badge/make%20test-5%2F5%20suites-2ea44f)](#verification)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

---

Most GPT-2 code hands you a three-line `transformers` call and hides the
model. crucible is the opposite: every tensor, every matmul, every token
of the sampler is a small C file in `src/`. The weights arrive as **148
plain-text files** you can `head` and `diff`. The forward pass is pinned
against byte-exact fixtures, so you can refactor the math and know
immediately when you broke it.

Ported from [gpt2LiveStream](https://github.com/VishwajeetSinghParihar750/gpt2LiveStream)
(reference only — none of its language, deps, or code). This is a reading
project first, a chat box second.

## Quick start

Needs `gcc` or `clang` + `make`. ncurses for the TUI (macOS ships it;
Debian/Ubuntu: `apt install libncurses-dev` — without it everything still
builds, `--tui` just reports unavailable). ~2 GB disk for the model files.

**One-time weight export** (the only place Python is allowed to exist):

```sh
python3 -m venv tools/.venv
tools/.venv/bin/pip install -r tools/requirements.txt
tools/.venv/bin/python tools/export_weights.py models
curl -L -o models/vocab.json https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
curl -L -o models/merges.txt https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt
ls models/*.txt | wc -l   # 149 = 148 weight dumps + merges.txt
```

**Build and go:**

```sh
make
./crucible
```

## What you get

All outputs below are real and byte-exact — they're also pinned in
`tests/fixtures.txt`, so they can't silently drift:

```sh
$ ./crucible encode "Hello, world!"
15496 11 995 0

$ ./crucible forward "Hello"
11 -32.391758 [,]
13 -32.591454 [.]
198 -33.203354 [
]
12 -33.836437 [-]
25 -33.876507 [:]

$ ./crucible generate "Hello" 20
, I'm sorry, but I'm not sure if you're aware of this. I'm not

$ echo hi | ./crucible --no-tui chat 1
you: ai: thinking...
ai:  hi
done - 1 tokens, 2.4s
you:
```

## Usage

| Command | What it does |
|---|---|
| `./crucible` | chat — ncurses TUI on a tty, classic scrolling REPL in a pipe |
| `./crucible --tui chat [max]` | force the TUI |
| `./crucible --no-tui chat [max]` | force the classic REPL (default cap 120 tokens/turn) |
| `./crucible generate "prompt" [n]` | one-shot greedy generation (default 30) |
| `./crucible forward "prompt"` | top-5 next tokens with logits |
| `./crucible encode "text"` | BPE token ids |
| `./crucible dump` | sanity-print of the loaded tensors |

Inside the chat (both UIs behave the same — streaming, history, `chat.log`):

| Command | Effect |
|---|---|
| `/temp N` | sampling temperature, `0` = greedy (default 0.4, floor 1e-5) |
| `/tokens` | context usage: `used / 900 tokens, N dropped` |
| `/save FILE` / `/load FILE` | transcript out / in |
| `/reset` | forget the conversation |
| `/quit` (or `Ctrl-C`/`Ctrl-D` on an empty line) | leave |

The TUI is three panes — scrollable transcript, status line, input — with
full line editing: `Left/Right Home/End`, `Ctrl-A/E/K/U/W`, `Up/Down`
history, `TAB` completes `/`-commands, `Ctrl-R` searches history,
`PgUp/PgDn` scrolls the transcript. `NO_COLOR=1` turns colors off.
The model re-reads at most the last 900 tokens of context; older turns
are dropped and counted.

## How it works

```
one-time (Python)                          every run (C only)
tools/export_weights.py ──▶ models/transformer.h.*.txt   148 text dumps, ~1.9 GB
HF vocab.json + merges.txt ─▶ models/

models/*.txt ── src/wtxt.c ──────┐
models/vocab+merges ── src/bpe ──┤
your text ───────────────────────┘
        │
        ▼
src/model.c: 12 × [ LN → fused [Q|K|V] → causal attn (1/√d, -inf mask)
                     → proj → LN → MLP (tanh GELU) → +residual ]
        → final LN → tied LM head → 50257 logits
        │
        ▼
sampling: top-k 40, temp 0.4 (greedy at 0)
        │
        ▼
src/chat.c · src/term.c · src/tui_nc.c — streamed token by token
```

Details that matter and are pinned by tests:

- **Fused `[Q|K|V]` layout** — one matmul per block, exactly like the
  reference, sliced per head afterwards.
- **Causal mask as `-inf` before softmax**, tanh-approximation GELU,
  LayerNorm eps `1e-5`, LM head tied to `wte`.
- **No KV-cache** — every token is a full recompute over the context,
  exactly like the reference. Slower than caching, but half the code
  and nothing to keep in sync.
- **Sampling** — top-k 40, temp 0.4, floor `1e-5`, greedy at 0, drawn
  with an in-C MT19937 like the reference.
- **Tokenizer** — byte-level BPE verified token-for-token against
  HuggingFace `GPT2Tokenizer`, including multi-space and ` $19.99`
  cases (pinned in `test_bpe`).

## Verification

`make && make test` runs five suites (math ops, BPE, model layout,
chat behavior, TUI transcript buffer). On top of that, the parity
contract: after any change to math or weights,

```sh
./crucible encode "Hello, world!"   # must print: 15496 11 995 0
./crucible forward "Hello"          # must byte-match tests/fixtures.txt
echo hi | ./crucible --no-tui chat 1   # must exit 0
```

If you refactor `src/model.c`, these three lines are your seatbelt.

## Performance, honestly

Measured on one Apple Silicon laptop, debug build (`-O0`), weights
already on disk:

| What | Number |
|---|---|
| Weights parse at startup | ~14 s (1.9 GB of decimal text) |
| Greedy `generate "Hello" 5` | ~16 s total, ~0.4 s/token after parse (KV-cache) |
| `encode` / `forward` after load | instant |

`make release` (`-O3`) speeds the math up further. There are no fused
kernels, no BLAS, no SIMD, no KV-cache — the trade for total readability
is speed, exactly like the reference.

**Why text-file weights?** Every tensor is a file you can `head`, `wc`,
and diff when a layer looks wrong; the loader is ~100 lines of C; and
dump-for-dump it matches the reference implementation, so cross-checking
is file-by-file. Costs, stated plainly: ~3.5x disk vs `safetensors`
(1.9 GB vs 548 MB), a slower startup, and floats round-tripped through
decimal text — verified lossless (fixtures are byte-identical before and
after the switch).

## Repo map

```
Makefile                  make (debug) · make release (-O3) · make test
compile_commands.json     -Iinclude for clangd / your editor
include/                  headers for each src file
src/
  main.c                  CLI dispatch + classic REPL
  wtxt.c                  loads the 148 text dumps (~100 lines)
  bpe.c                   byte-pair encoder/decoder (vocab.json + merges.txt)
  math.c                  matmul, layernorm, softmax, tanh GELU
  model.c                 12-layer forward, MT19937 top-k sampling
  chat.c                  context window, history, save/load
  term.c                  raw-mode line editing for the classic REPL
  tui_nc.c                ncurses panes (transcript / status / input)
  tui_tr.c                tty-free transcript buffer (unit-tested)
tests/                    5 suites + fixtures.txt (the parity pins)
tools/export_weights.py   the only Python: HF weights → txt dumps
models/                   gitignored, export once (~1.9 GB)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `load failed` / `tokenizer load failed` | `models/` missing or truncated — re-export per Quick start |
| `loaded N/148 tensors` | a dump is missing; re-run the exporter |
| `term.h not found` / `History is undefined` | you compiled by hand — use `make`; `compile_commands.json` is at the root for clangd |
| Chat turn is slow | expected at `-O0`; see Performance. Lower the cap: `./crucible chat 40` |
| Terminal echo broken after a crash | `kill -9` can leave raw mode on; run `reset` in your shell |

## Credits

- Model and forward pass: [gpt2LiveStream](https://github.com/VishwajeetSinghParihar750/gpt2LiveStream) (reference only — none of its language, deps, or code)
- C structure and tokenizer approach: [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy
- TUI layout (split message columns, boxed panes, scroll counter): [Term-Chat-TUI](https://github.com/GrandBIRDLizard/Term-Chat-TUI.git)
- Weights: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- Logo: hand-written SVG in `assets/logo.svg`
- License: MIT — see [LICENSE](LICENSE)
