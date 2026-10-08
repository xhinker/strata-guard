# Strata control-span rescue guard

Prevents the Qwen-family premature stop: the model ends its turn with `<|im_end|>`
while it is in the middle of *quoting* a control tag, so the answer arrives
truncated mid-sentence with `finish_reason: stop` and no error.

## Why this lives in the Python server, not the C++ engine

* The engine works in token ids only (it streams `T <id>` lines and has no
  detokenizer) and ends a turn with a post-sample check on its eos ids
  (`generate.cpp`: `eos_ids = {248044, 248046}`, with its own copies in the
  speculative path).  A mask-style guard there needs a precomputed per-id table
  and edits in 3+ places of a 10k-line GPU file - which conflicts with upstream
  on every pull.
* The Python server owns the tokenizer and sees the real text, so the same
  semantics (llama.cpp's "substitution" arm) can be implemented exactly, with
  **zero edits to upstream files**.  The patch exists only for the running
  process.

## Semantics

Identical to the llama.cpp fork's guard:
predicate `inside_span(tail)` = (1) odd number of `"` in the window, or
(2) the tail ends on `|` or on an opening backtick, or (3) a `<` with no `>` or
whitespace after it.  Ported 1:1 from `common/control-span-predicate.h`, including
its notes about tail-local quote parity and about `>` not being an opener.

One difference the tests forced: the tail is built the way llama.cpp builds it -
**CONTROL-type tokens contribute nothing** (`common_token_to_piece(..., special=false)`).
Rendering the stop token as text would let `<|im_end|>` close the very span the
predicate is looking for, and the guard would never fire.

When the engine stops inside a span, the guard appends the escaped spelling of the
stop token (U+200B after the first character, so it cannot re-tokenise as one
reserved id), yields it to the caller, and asks the engine to continue.  At most
`--control-span-rescues` rescues per request, then it lets the turn end.

## Run it

```bash
/mnt/data_2t/github_repos/Strata/.venv/bin/python \
  /mnt/data_2t/github_repos/Strata-guard/serve_with_guard.py \
  --engine strata --config strata-3090x3-iq3s.json --port 8081 \
  --control-span-guard on
```

The wrapper re-execs into Strata's venv by itself, so plain `python3` works too.
Everything after `--control-span-guard` is handed to `serve/server.py` unchanged.

| flag                          | env variable                    | default |
|-------------------------------|---------------------------------|---------|
| `--control-span-guard on\|off` | `STRATA_CONTROL_SPAN_GUARD`     | **off** |
| `--control-span-window N`      | `STRATA_CONTROL_SPAN_WINDOW`    | 32      |
| `--control-span-rescues N`     | `STRATA_CONTROL_SPAN_RESCUES`   | 2       |
| (eos ids)                      | `STRATA_CONTROL_SPAN_EOS`       | 248044,248046 |

It is **off by default**: with the flag absent (and the env unset) the server
behaves exactly as before.

## Logs

```
[control-span] ENABLED (eos ids [248044, 248046], window 32, max rescues 2)
[control-span] rescue 1: stopped inside a span (tail='The closing tag is `') - continuing with the escaped spelling of token 248046 as [27, 15231, 91, 316, 6018, 91, 29]
[control-span] stop inside a span, but the rescue cap (2) is reached - letting it end
```

## Tests (no GPU needed)

```bash
cd /mnt/data_2t/github_repos/Strata-guard
python3 test_control_span.py                                   # predicate + escape helper
/mnt/data_2t/github_repos/Strata/.venv/bin/python test_integration.py   # guard vs MockEngine
```

## Revert

Stop the server and start it with `serve/server.py` as before.  Nothing in the
Strata tree was modified, so a `git pull` needs no re-apply.

## If you want every client covered, not just Python

The engine-side variant (an id-table driven guard in `generate.cpp`) is the only
way to protect non-Python clients.  It needs: a generated per-id table (quotes,
last/previous character, tag-closing flags, escape sequences), a hook in the CLI
parser, a hook at each eos check (main + speculative paths), and a re-apply step
after every `git pull`.  Not implemented here on purpose.
